"""Module 4: transcript cleanup — pure Python, unit-testable.

Removes fillers, merges broken fragments into sentences, preserves timestamps.
Output "sentences" are the atomic units clip boundaries snap to.
"""
from __future__ import annotations

import logging

from app.utils.textnorm import ends_sentence, is_filler, normalize_spaces

log = logging.getLogger(__name__)

MAX_SENTENCE_SECONDS = 18.0  # force a break on run-on segments
MERGE_GAP_SECONDS = 0.9      # fragments closer than this get merged


def clean_transcript(raw: dict) -> dict:
    """raw: transcriber output. Returns {"language", "sentences": [...]}.

    Sentence: {text, start, end, confidence, words: [...]}
    """
    sentences: list[dict] = []
    current_words: list[dict] = []
    current_texts: list[str] = []
    current_start: float | None = None
    current_conf: list[float] = []

    def flush() -> None:
        nonlocal current_words, current_texts, current_start, current_conf
        if not current_texts or current_start is None:
            current_words, current_texts, current_conf = [], [], []
            return
        text = normalize_spaces(" ".join(current_texts))
        if text:
            end = current_words[-1]["end"] if current_words else current_start
            sentences.append(
                {
                    "text": text,
                    "start": current_start,
                    "end": end,
                    "confidence": round(
                        sum(current_conf) / max(len(current_conf), 1), 3
                    ),
                    "words": current_words,
                }
            )
        current_words, current_texts, current_conf = [], [], []
        current_start = None

    for seg in raw.get("segments", []):
        seg_words = seg.get("words") or _synthesize_words(seg)
        kept = [w for w in seg_words if not is_filler(w["word"])]
        if not kept:
            continue

        if current_start is not None and current_words:
            gap = kept[0]["start"] - current_words[-1]["end"]
            duration = current_words[-1]["end"] - current_start
            if gap > MERGE_GAP_SECONDS or duration > MAX_SENTENCE_SECONDS:
                flush()

        if current_start is None:
            current_start = kept[0]["start"]
        current_words.extend(kept)
        current_texts.append(" ".join(w["word"] for w in kept))
        current_conf.append(seg.get("confidence", 0.5))

        if ends_sentence(seg["text"]):
            flush()

    flush()
    log.info("Cleaned transcript: %d sentences", len(sentences))
    return {"language": raw.get("language"), "sentences": sentences}


def _synthesize_words(seg: dict) -> list[dict]:
    """Fallback when word timestamps are missing: distribute words evenly."""
    tokens = seg["text"].split()
    if not tokens:
        return []
    span = max(seg["end"] - seg["start"], 0.01)
    step = span / len(tokens)
    return [
        {
            "word": tok,
            "start": round(seg["start"] + i * step, 3),
            "end": round(seg["start"] + (i + 1) * step, 3),
            "prob": seg.get("confidence", 0.5),
        }
        for i, tok in enumerate(tokens)
    ]


def transcript_text(cleaned: dict, numbered: bool = True) -> str:
    """Render sentences for LLM prompts; numbering lets the LLM reference them."""
    lines = []
    for i, s in enumerate(cleaned["sentences"]):
        prefix = f"[{i}] " if numbered else ""
        lines.append(f"{prefix}({s['start']:.1f}-{s['end']:.1f}s) {s['text']}")
    return "\n".join(lines)
