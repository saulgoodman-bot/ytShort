"""Modules 5+6: LLM semantic understanding and intelligent clip detection.

The LLM reads the numbered, timestamped sentence list and proposes candidate
clips by sentence index. Boundaries are then snapped to sentences and forced
into the [min, max] duration window — never cutting mid-sentence.

If the LLM fails or returns garbage, a heuristic sliding-window fallback
guarantees the pipeline still produces candidates.
"""
from __future__ import annotations

import logging

from app.ai.cleaner import transcript_text
from app.ai.llm import OllamaClient
from app.utils.errors import LLMError

log = logging.getLogger(__name__)

_UNDERSTAND_SYSTEM = (
    "You are an expert Telugu/English video editor who finds viral moments in "
    "long videos. You read timestamped transcripts and answer ONLY with JSON."
)

_UNDERSTAND_PROMPT = """Analyze this transcript of a Telugu video. Sentences are numbered [i] with (start-end s) timestamps.

TRANSCRIPT:
{transcript}

Return JSON exactly in this shape:
{{
  "main_topic": "...",
  "subtopics": ["..."],
  "language_mix": "telugu|english|mixed",
  "notable_moments": [
    {{"sentence": 3, "type": "hook|joke|fact|quote|advice|emotional|question|cta|viral", "note": "..."}}
  ]
}}"""

_CLIPS_PROMPT = """You are selecting YouTube Shorts candidates from this Telugu video transcript. Sentences are numbered [i] with (start-end s) timestamps.

TRANSCRIPT:
{transcript}

VIDEO CONTEXT: main topic: {main_topic}

Rules:
- Each clip must START at a strong hook: a question, surprising fact, emotional statement, or curiosity trigger.
- Each clip must END after a complete thought, before the topic changes.
- Target duration {min_dur:.0f}-{max_dur:.0f} seconds. Use the timestamps.
- Propose {n_candidates} candidates covering DIFFERENT topics/moments.

Return JSON exactly in this shape:
{{
  "clips": [
    {{
      "start_sentence": 4,
      "end_sentence": 9,
      "hook_type": "question|fact|emotional|curiosity|viral",
      "hook_text": "first line of the clip",
      "reason": "why this would perform as a Short"
    }}
  ]
}}"""


def understand_content(llm: OllamaClient, cleaned: dict) -> dict:
    try:
        result = llm.generate_json(
            _UNDERSTAND_PROMPT.format(transcript=transcript_text(cleaned)),
            system=_UNDERSTAND_SYSTEM,
        )
        if isinstance(result, dict):
            return result
    except LLMError as exc:
        log.warning("Content understanding failed: %s", exc)
    return {"main_topic": "unknown", "subtopics": [], "notable_moments": []}


def detect_candidates(
    llm: OllamaClient,
    cleaned: dict,
    analysis: dict,
    min_duration: float,
    max_duration: float,
    n_candidates: int = 16,
) -> list[dict]:
    sentences = cleaned["sentences"]
    candidates: list[dict] = []
    try:
        result = llm.generate_json(
            _CLIPS_PROMPT.format(
                transcript=transcript_text(cleaned),
                main_topic=analysis.get("main_topic", "unknown"),
                min_dur=min_duration,
                max_dur=max_duration,
                n_candidates=n_candidates,
            ),
            system=_UNDERSTAND_SYSTEM,
        )
        raw_clips = result.get("clips", []) if isinstance(result, dict) else []
        for rc in raw_clips:
            cand = _snap_candidate(rc, sentences, min_duration, max_duration)
            if cand:
                candidates.append(cand)
    except LLMError as exc:
        log.warning("LLM clip detection failed (%s); using heuristic fallback", exc)

    if len(candidates) < 5:
        log.info("Only %d LLM candidates; adding heuristic windows", len(candidates))
        candidates.extend(
            heuristic_candidates(sentences, min_duration, max_duration)
        )
    candidates = _dedupe_ranges(candidates)
    log.info("Candidate clips: %d", len(candidates))
    return candidates


def _snap_candidate(
    rc: dict, sentences: list[dict], min_dur: float, max_dur: float
) -> dict | None:
    try:
        i = max(0, min(int(rc["start_sentence"]), len(sentences) - 1))
        j = max(i, min(int(rc["end_sentence"]), len(sentences) - 1))
    except (KeyError, TypeError, ValueError):
        return None

    # Expand forward sentence-by-sentence until min duration is met.
    while sentences[j]["end"] - sentences[i]["start"] < min_dur and j < len(sentences) - 1:
        j += 1
    # Trim from the end until under max duration (keep at least one sentence).
    while sentences[j]["end"] - sentences[i]["start"] > max_dur and j > i:
        j -= 1

    duration = sentences[j]["end"] - sentences[i]["start"]
    if not (min_dur * 0.8 <= duration <= max_dur * 1.05):
        return None
    return {
        "start": sentences[i]["start"],
        "end": sentences[j]["end"],
        "start_sentence": i,
        "end_sentence": j,
        "text": " ".join(s["text"] for s in sentences[i : j + 1]),
        "hook_type": rc.get("hook_type", "unknown"),
        "hook_text": rc.get("hook_text", sentences[i]["text"][:120]),
        "reason": rc.get("reason", ""),
        "source": "llm",
    }


def heuristic_candidates(
    sentences: list[dict], min_dur: float, max_dur: float
) -> list[dict]:
    """Sliding windows over sentences hitting the duration band.

    Guarantees candidates even with no working LLM. Windows advance ~half a
    window at a time so they overlap enough to cover hooks near boundaries.
    """
    out: list[dict] = []
    i = 0
    n = len(sentences)
    while i < n:
        j = i
        while j < n - 1 and sentences[j]["end"] - sentences[i]["start"] < min_dur:
            j += 1
        while j > i and sentences[j]["end"] - sentences[i]["start"] > max_dur:
            j -= 1
        duration = sentences[j]["end"] - sentences[i]["start"]
        if min_dur * 0.8 <= duration <= max_dur * 1.05:
            out.append(
                {
                    "start": sentences[i]["start"],
                    "end": sentences[j]["end"],
                    "start_sentence": i,
                    "end_sentence": j,
                    "text": " ".join(s["text"] for s in sentences[i : j + 1]),
                    "hook_type": "window",
                    "hook_text": sentences[i]["text"][:120],
                    "reason": "heuristic sliding window",
                    "source": "heuristic",
                }
            )
        advance = max((j - i + 1) // 2, 1)
        i += advance
    return out


def _dedupe_ranges(candidates: list[dict], tolerance: float = 2.0) -> list[dict]:
    """Drop candidates whose (start, end) nearly duplicate an earlier one."""
    kept: list[dict] = []
    for c in candidates:
        duplicate = any(
            abs(c["start"] - k["start"]) < tolerance
            and abs(c["end"] - k["end"]) < tolerance
            for k in kept
        )
        if not duplicate:
            kept.append(c)
    return kept
