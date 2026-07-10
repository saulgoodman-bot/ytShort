# AutoShorts — Local AI YouTube Shorts Generator (Telugu)

Converts a 10–20 minute Telugu YouTube video into 5–10 vertical, subtitled,
publish-ready Shorts — fully offline on macOS, using only free open-source
models (faster-whisper + Ollama + MediaPipe + PySceneDetect + FFmpeg).

## How it works

```
video ─► audio (ffmpeg) ─► Telugu transcript (faster-whisper, word timestamps)
      ─► cleanup (fillers, sentence merge) ─► LLM understanding (Ollama)
      ─► candidate clips (semantic hooks, sentence-snapped, 20–60 s)
      ─► multi-signal scoring:
            LLM: emotion · curiosity · storytelling · engagement · keywords
            code: speaking energy · scene-cut density · silence · trend · dup
      ─► top 5–10 ─► face tracking (MediaPipe) ─► smart 9:16 crop
      ─► ASS subtitles (karaoke word-highlight) ─► ffmpeg render 1080×1920
      ─► titles / description / hashtags / thumbnail per Short
```

Every stage caches its output under `cache/<video>_<hash>/`, so an interrupted
run resumes where it stopped, and re-runs with different settings skip
transcription entirely.

## Install (macOS)

```bash
./setup.sh                       # ffmpeg, ollama, venv, pip deps, qwen3:8b
source .venv/bin/activate
python -m app check              # verify everything is green
```

Optional Telugu subtitle font (recommended):
`brew install --cask font-noto-sans-telugu`

## Use

```bash
# start the local LLM once per session
ollama serve &

python -m app generate input/my_video.mp4
python -m app generate input/*.mp4 --max-shorts 6 --subtitle-style bold
python -m app generate video.mp4 --whisper-model small   # faster, less accurate
```

Outputs land in `output/<video_name>/`:

```
short_01.mp4          rendered 1080×1920 Short
short_01.ass          burned-in subtitle source
short_01.json         titles, description, hashtags, scores, timings
short_01_thumb.jpg    thumbnail suggestion
```

## Configuration

Defaults live in `config/default.yaml`. Create `config/local.yaml` to
override anything (it is merged on top). Most-used knobs:

| Key | Meaning |
|---|---|
| `whisper.model` | tiny/base/small/medium/large-v3 — accuracy vs speed |
| `llm.model` | any Ollama model (qwen3:8b, gemma3:12b, llama3.1:8b) |
| `clips.min/max_duration` | Short length window (default 20–60 s) |
| `clips.max_shorts` | how many Shorts to render |
| `scoring.weights` | signal weights for ranking |
| `crop.mode` | center / smart_static / smart_dynamic |
| `subtitles.style` | classic / modern / bold / minimal / tiktok |
| `trends.keywords` | manual trending keywords that boost matching clips |

## Realistic performance expectations (honest numbers)

On an M-series MacBook Air, the dominant costs for a 10-minute video are:

- **Whisper**: CTranslate2 runs CPU-only on macOS. `medium`+int8 ≈ roughly
  real-time or slower on an Air; `small` is ~2–3× faster. `large-v3` is the
  most accurate for Telugu but expect several× real-time. If transcription is
  too slow, drop to `small` — the LLM sees cleaned text either way.
- **LLM**: qwen3:8b on 16 GB Apple Silicon is comfortable; on 8 GB machines
  use a 4B-class model (`gemma3:4b`, `qwen3:4b`) or Ollama will swap heavily.
- **Rendering**: hardware-encoded via `h264_videotoolbox` — fast.

The "10-minute video in under 10 minutes" target is achievable with
`whisper.model: small` + a 4B LLM; with `medium` + 8B expect 15–25 minutes on
an Air. An M-series Pro/Max or any machine with more cores shortens this.

## Degradation ladder (nothing hard-fails)

- **No Ollama running** → heuristic sliding-window candidates, neutral
  semantic scores; energy/visual/silence signals still rank clips; generic
  metadata. You still get Shorts.
- **Scene detection fails** → visual score neutral, pipeline continues.
- **No face detected** → center crop fallback.
- **LLM returns malformed JSON** → tolerant parser, then per-batch fallback.

## Project layout

```
app/
  pipeline/orchestrator.py   stage sequencing, caching, resume
  ai/transcriber.py          faster-whisper wrapper
  ai/cleaner.py              filler removal, sentence merge (pure)
  ai/llm.py                  Ollama client + tolerant JSON parsing
  ai/semantic.py             content understanding + clip detection
  analytics/scoring.py       multi-signal scoring + ranked selection (pure math)
  analytics/energy.py        loudness/speech-rate/pause analysis (stdlib)
  analytics/metadata_gen.py  titles, descriptions, hashtags
  video/importer.py          probe + validation
  video/audio.py             WAV extraction, denoise, loudnorm
  video/scenes.py            PySceneDetect wrapper
  video/faces.py             MediaPipe face tracking + EMA smoothing
  video/cropper.py           9:16 crop geometry (pure)
  video/renderer.py          ffmpeg render with videotoolbox
  video/thumbnails.py        best-frame + text overlay
  subtitles/styles.py        5 presets + config overrides
  subtitles/generator.py     ASS builder, karaoke word highlighting (pure)
  utils/                     config, cache, logging, ffmpeg, errors, text
  ui/                        (reserved for the desktop GUI phase)
```

Modules marked (pure) have no heavy dependencies and are covered by unit
tests: `pytest tests/`.

## Extending

- **New scoring signal**: add a function in `analytics/`, write into
  `cand["scores"]["<name>"]` in `orchestrator._stage_score`, add a weight in
  `config/default.yaml`. Nothing else changes.
- **New subtitle style**: add a preset in `subtitles/styles.py`.
- **New export target (Reels/TikTok)**: reuse `renderer.render_short` with a
  different resolution/bitrate profile.
- **Swap Whisper backend** (e.g. whisper.cpp with Metal): implement the same
  return shape as `ai/transcriber.transcribe` — everything downstream only
  reads that dict.

## Roadmap (not in this build)

Desktop GUI (PySide6), batch queue UI, speaker diarization, online trend
fetching, animated caption effects, B-roll insertion.
