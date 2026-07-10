"""Module 3: speech recognition with faster-whisper (word-level timestamps)."""
from __future__ import annotations

import logging
from pathlib import Path

from app.utils.errors import TranscriptionError

log = logging.getLogger(__name__)


def transcribe(
    audio_path: Path,
    model_name: str = "medium",
    language: str | None = "te",
    device: str = "auto",
    compute_type: str = "int8",
    word_timestamps: bool = True,
    vad_filter: bool = True,
    beam_size: int = 5,
) -> dict:
    """Return {"language", "language_probability", "segments": [...]}.

    Each segment: {text, start, end, confidence, words: [{word,start,end,prob}]}
    """
    try:
        from faster_whisper import WhisperModel  # lazy heavy import
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    resolved_device = "cpu" if device == "auto" else device
    log.info(
        "Loading Whisper model '%s' (device=%s, compute=%s)…",
        model_name, resolved_device, compute_type,
    )
    try:
        model = WhisperModel(model_name, device=resolved_device, compute_type=compute_type)
    except Exception as exc:
        raise TranscriptionError(f"Could not load Whisper model '{model_name}': {exc}") from exc

    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    segments = []
    for seg in segments_iter:
        words = [
            {
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "prob": round(w.probability, 3),
            }
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append(
            {
                "text": seg.text.strip(),
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                # avg_logprob in [-inf,0]; map to a rough 0..1 confidence
                "confidence": round(min(1.0, max(0.0, 1.0 + seg.avg_logprob)), 3),
                "words": words,
            }
        )
        log.debug("[%.1f-%.1f] %s", seg.start, seg.end, seg.text.strip())

    log.info(
        "Transcribed %d segments (lang=%s p=%.2f)",
        len(segments), info.language, info.language_probability,
    )
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "segments": segments,
    }
