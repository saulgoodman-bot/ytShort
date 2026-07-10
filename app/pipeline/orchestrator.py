"""End-to-end pipeline: video in -> 5-10 rendered Shorts out.

Every stage caches its artifact under cache/<video>_<hash>/ so interrupted or
re-run jobs resume instead of recomputing. Stage timing is logged.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.ai import cleaner, semantic, transcriber
from app.ai.llm import OllamaClient
from app.analytics import energy as energy_mod
from app.analytics import metadata_gen, scoring
from app.subtitles import generator as subtitle_gen
from app.subtitles.styles import get_style
from app.utils import ffmpeg
from app.utils.cache import StageCache
from app.utils.config import Config, resolve_dir
from app.video import audio as audio_mod
from app.video import cropper, faces, importer, renderer, scenes, thumbnails

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------ run
    def run(self, video_path: Path) -> list[dict]:
        cfg = self.cfg
        t0 = time.time()
        ffmpeg.require_binaries()

        cache = StageCache(resolve_dir(cfg, "cache_dir"), video_path)
        out_dir = resolve_dir(cfg, "output_dir") / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = self._stage_import(cache, video_path)
        wav = self._stage_audio(cache, video_path)
        transcript = self._stage_transcribe(cache, wav)
        cleaned = self._stage_clean(cache, transcript)
        if not cleaned["sentences"]:
            log.error("No usable speech found — cannot generate Shorts.")
            return []

        llm = OllamaClient(
            host=cfg.llm.host,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            timeout=cfg.llm.timeout_seconds,
            max_retries=cfg.llm.max_retries,
        )
        llm_ok = llm.is_available()
        if not llm_ok:
            log.warning(
                "Ollama not reachable at %s — falling back to heuristics. "
                "Start it with: ollama serve", cfg.llm.host,
            )

        analysis = self._stage_understand(cache, llm, cleaned, llm_ok)
        cuts = self._stage_scenes(cache, video_path)
        candidates = self._stage_candidates(cache, llm, cleaned, analysis, llm_ok)
        selected = self._stage_score(cache, llm, candidates, cleaned, wav, cuts, llm_ok)

        results = self._stage_produce(
            video_path, meta, cleaned, analysis, selected, llm, llm_ok, out_dir
        )
        log.info(
            "Pipeline finished: %d Shorts in %.1f min -> %s",
            len(results), (time.time() - t0) / 60.0, out_dir,
        )
        return results

    # --------------------------------------------------------------- stages
    def _timed(self, name: str):
        class _Timer:
            def __enter__(inner):
                inner.t = time.time()
                log.info("stage %s: start", name)
                return inner

            def __exit__(inner, *exc):
                log.info("stage %s: %.1fs", name, time.time() - inner.t)

        return _Timer()

    def _stage_import(self, cache: StageCache, video: Path) -> dict:
        if cache.has("meta.json"):
            return cache.load_json("meta.json")
        with self._timed("import"):
            meta = importer.import_video(video).as_dict()
        cache.save_json("meta.json", meta)
        return meta

    def _stage_audio(self, cache: StageCache, video: Path) -> Path:
        wav = cache.path("audio.wav")
        if wav.exists():
            return wav
        with self._timed("audio"):
            audio_mod.extract_audio(
                video, wav,
                sample_rate=self.cfg.audio.sample_rate,
                denoise=self.cfg.audio.denoise,
                normalize=self.cfg.audio.normalize,
            )
        return wav

    def _stage_transcribe(self, cache: StageCache, wav: Path) -> dict:
        if cache.has("transcript.json"):
            return cache.load_json("transcript.json")
        w = self.cfg.whisper
        with self._timed("transcribe"):
            transcript = transcriber.transcribe(
                wav, model_name=w.model, language=w.get("language"),
                device=w.device, compute_type=w.compute_type,
                word_timestamps=w.word_timestamps, vad_filter=w.vad_filter,
                beam_size=w.beam_size,
            )
        cache.save_json("transcript.json", transcript)
        return transcript

    def _stage_clean(self, cache: StageCache, transcript: dict) -> dict:
        if cache.has("cleaned.json"):
            return cache.load_json("cleaned.json")
        with self._timed("clean"):
            cleaned = cleaner.clean_transcript(transcript)
        cache.save_json("cleaned.json", cleaned)
        return cleaned

    def _stage_understand(
        self, cache: StageCache, llm: OllamaClient, cleaned: dict, llm_ok: bool
    ) -> dict:
        if cache.has("analysis.json"):
            return cache.load_json("analysis.json")
        if not llm_ok:
            return {"main_topic": "unknown", "subtopics": [], "notable_moments": []}
        with self._timed("understand"):
            analysis = semantic.understand_content(llm, cleaned)
        cache.save_json("analysis.json", analysis)
        return analysis

    def _stage_scenes(self, cache: StageCache, video: Path) -> list[float]:
        if cache.has("scenes.json"):
            return cache.load_json("scenes.json")
        s = self.cfg.scenes
        with self._timed("scenes"):
            try:
                cuts = scenes.detect_scenes(
                    video, detector_name=s.detector, threshold=s.threshold,
                    min_scene_len_seconds=s.min_scene_len_seconds,
                )
            except Exception as exc:  # scene detection is enhancing, not critical
                log.warning("Scene detection failed (%s); continuing without", exc)
                cuts = []
        cache.save_json("scenes.json", cuts)
        return cuts

    def _stage_candidates(
        self, cache: StageCache, llm: OllamaClient, cleaned: dict,
        analysis: dict, llm_ok: bool,
    ) -> list[dict]:
        if cache.has("candidates.json"):
            return cache.load_json("candidates.json")
        c = self.cfg.clips
        with self._timed("candidates"):
            if llm_ok:
                cands = semantic.detect_candidates(
                    llm, cleaned, analysis, c.min_duration, c.max_duration
                )
            else:
                cands = semantic.heuristic_candidates(
                    cleaned["sentences"], c.min_duration, c.max_duration
                )
        cache.save_json("candidates.json", cands)
        return cands

    def _stage_score(
        self, cache: StageCache, llm: OllamaClient, candidates: list[dict],
        cleaned: dict, wav: Path, cuts: list[float], llm_ok: bool,
    ) -> list[dict]:
        if cache.has("selection.json"):
            return cache.load_json("selection.json")
        cfg = self.cfg
        all_words = [w for s in cleaned["sentences"] for w in s["words"]]
        with self._timed("score"):
            if llm_ok:
                scoring.llm_semantic_scores(llm, candidates)
            else:
                for cand in candidates:
                    cand["scores"] = {k: 50.0 for k in scoring.LLM_SIGNALS}
                    cand["keywords"] = []

            audio_energy = energy_mod.AudioEnergy(wav)
            trend_keywords = list(cfg.trends.get("keywords") or [])
            for cand in candidates:
                s, e = cand["start"], cand["end"]
                cand["scores"]["energy"] = energy_mod.energy_score(
                    audio_energy, all_words, s, e
                )
                cand["scores"]["visual"] = scoring.visual_score(cuts, s, e)
                cand["scores"]["trend"] = scoring.trend_score(
                    cand["text"], cand.get("keywords", []), trend_keywords
                )
                cand["scores"]["silence_penalty"] = scoring.silence_penalty_score(
                    energy_mod.pause_ratio(all_words, s, e)
                )

            selected = scoring.select_top(
                candidates,
                weights=cfg.scoring.weights.as_dict(),
                penalties=cfg.scoring.penalties.as_dict(),
                max_shorts=cfg.clips.max_shorts,
                max_overlap=cfg.clips.max_overlap,
                duplicate_threshold=cfg.scoring.duplicate_similarity_threshold,
            )
        if len(selected) < cfg.clips.min_shorts:
            log.warning(
                "Only %d clips selected (target >= %d) — video may be short on "
                "distinct moments.", len(selected), cfg.clips.min_shorts,
            )
        for rank, cand in enumerate(selected, 1):
            log.info(
                "clip #%d %.1f-%.1fs score=%.1f hook=%s",
                rank, cand["start"], cand["end"], cand["final_score"],
                cand.get("hook_type"),
            )
        cache.save_json("selection.json", selected)
        return selected

    def _stage_produce(
        self, video: Path, meta: dict, cleaned: dict, analysis: dict,
        selected: list[dict], llm: OllamaClient, llm_ok: bool, out_dir: Path,
    ) -> list[dict]:
        cfg = self.cfg
        style = get_style(cfg.subtitles)
        all_words = [w for s in cleaned["sentences"] for w in s["words"]]
        results: list[dict] = []

        for idx, clip in enumerate(selected, 1):
            name = f"short_{idx:02d}"
            out_mp4 = out_dir / f"{name}.mp4"
            with self._timed(f"produce {name}"):
                try:
                    # 1. Face tracking + crop plan
                    track = faces.track_faces(
                        video, clip["start"], clip["end"],
                        sample_fps=cfg.faces.sample_fps,
                        min_confidence=cfg.faces.min_detection_confidence,
                        smoothing_alpha=cfg.faces.smoothing_alpha,
                    )
                    plan = cropper.plan_crop(
                        meta["width"], meta["height"],
                        track.times, track.centers_x,
                        mode=cfg.crop.mode,
                        keyframe_interval=cfg.crop.dynamic_keyframe_interval,
                        clip_start=clip["start"],
                    )
                    # 2. Subtitles
                    ass_path = out_dir / f"{name}.ass"
                    subtitle_gen.write_ass(
                        ass_path, all_words, clip["start"], clip["end"], style,
                        play_w=cfg.render.width, play_h=cfg.render.height,
                    )
                    # 3. Render
                    renderer.render_short(
                        video, out_mp4, clip["start"], clip["end"], plan, ass_path,
                        width=cfg.render.width, height=cfg.render.height,
                        fps=cfg.render.fps, video_bitrate=cfg.render.video_bitrate,
                        audio_bitrate=cfg.render.audio_bitrate,
                        encoder=cfg.render.encoder,
                    )
                    # 4. Publishing metadata
                    if llm_ok and cfg.metadata.generate_titles:
                        publish = metadata_gen.generate_metadata(
                            llm, clip, analysis.get("main_topic", ""),
                            n_titles=cfg.metadata.n_titles,
                            n_hashtags=cfg.metadata.n_hashtags,
                        )
                    else:
                        publish = {
                            "titles": [clip.get("hook_text", "")[:100]],
                            "description": clip["text"][:300],
                            "hashtags": ["#Shorts", "#Telugu"],
                        }
                    # 5. Thumbnail
                    thumb = None
                    if cfg.thumbnails.enabled:
                        thumb_path = out_dir / f"{name}_thumb.jpg"
                        try:
                            thumb = thumbnails.generate_thumbnail(
                                video, clip["start"], clip["end"],
                                publish["titles"][0] if publish["titles"] else "",
                                thumb_path,
                                overlay_text=cfg.thumbnails.overlay_text,
                                font_size=cfg.thumbnails.font_size,
                                width=cfg.render.width, height=cfg.render.height,
                            )
                        except Exception as exc:
                            log.warning("Thumbnail failed for %s: %s", name, exc)

                    record = {
                        "file": str(out_mp4),
                        "thumbnail": str(thumb) if thumb else None,
                        "start": clip["start"],
                        "end": clip["end"],
                        "duration": round(clip["end"] - clip["start"], 2),
                        "final_score": clip.get("final_score"),
                        "scores": clip.get("scores", {}),
                        "hook_type": clip.get("hook_type"),
                        "hook_text": clip.get("hook_text"),
                        "face_detection_rate": track.detection_rate,
                        **publish,
                    }
                    (out_dir / f"{name}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    results.append(record)
                except Exception as exc:
                    log.error("Failed to produce %s: %s", name, exc, exc_info=True)
        return results
