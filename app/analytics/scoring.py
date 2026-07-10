"""Module 7: multi-signal clip scoring and final ranking.

LLM supplies semantic scores (emotion, curiosity, storytelling, engagement,
keywords). Deterministic code supplies energy, visual, silence, trend, and
duplicate signals. Pure ranking math lives in weighted_total/select_top so it
is unit-testable without any models.
"""
from __future__ import annotations

import logging

from app.ai.llm import OllamaClient
from app.utils.errors import LLMError
from app.utils.textnorm import jaccard_similarity

log = logging.getLogger(__name__)

_SCORE_SYSTEM = (
    "You are a viral-content analyst for Telugu YouTube Shorts. "
    "Score honestly and answer ONLY with JSON."
)

_SCORE_PROMPT = """Score each candidate clip below for YouTube Shorts potential. All scores are integers 0-100.

Definitions:
- emotion: excitement, passion, surprise, humor, anger, happiness expressed
- curiosity: hooks like "do you know", "you won't believe", open questions
- storytelling: problem->solution, before->after, conflict->resolution, examples, personal stories
- engagement: questions to audience, calls to action, predictions, opinions, debates
- keywords: presence of important named entities/topics (people, places, cinema, politics, tech, finance, AI)

CLIPS:
{clips_block}

Return JSON exactly:
{{"scores": [
  {{"id": 0, "emotion": 0, "curiosity": 0, "storytelling": 0, "engagement": 0, "keywords": 0, "keyword_list": ["..."]}}
]}}"""

LLM_SIGNALS = ("emotion", "curiosity", "storytelling", "engagement", "keywords")
_BATCH = 6


def llm_semantic_scores(llm: OllamaClient, candidates: list[dict]) -> None:
    """Attach LLM scores to each candidate in place (neutral 50s on failure)."""
    for base in range(0, len(candidates), _BATCH):
        batch = candidates[base : base + _BATCH]
        block = "\n\n".join(
            f"[{i}] ({c['end'] - c['start']:.0f}s) {c['text'][:600]}"
            for i, c in enumerate(batch)
        )
        scores_by_id: dict[int, dict] = {}
        try:
            result = llm.generate_json(
                _SCORE_PROMPT.format(clips_block=block), system=_SCORE_SYSTEM
            )
            for item in (result or {}).get("scores", []):
                try:
                    scores_by_id[int(item["id"])] = item
                except (KeyError, TypeError, ValueError):
                    continue
        except LLMError as exc:
            log.warning("LLM scoring failed for batch at %d: %s", base, exc)

        for i, cand in enumerate(batch):
            item = scores_by_id.get(i, {})
            for signal in LLM_SIGNALS:
                cand.setdefault("scores", {})[signal] = _clamp_score(
                    item.get(signal, 50)
                )
            cand["keywords"] = [
                str(k) for k in item.get("keyword_list", []) if isinstance(k, (str, int))
            ][:10]


def _clamp_score(value) -> float:
    try:
        return max(0.0, min(float(value), 100.0))
    except (TypeError, ValueError):
        return 50.0


def visual_score(scene_cuts: list[float], start: float, end: float) -> float:
    """0-100 from scene-cut density: ~1 cut / 8s scores well; flat talking-head scores low."""
    n = sum(1 for c in scene_cuts if start <= c <= end)
    per_10s = n / max((end - start) / 10.0, 0.1)
    return round(min(per_10s / 1.25, 1.0) * 100.0, 1)


def trend_score(text: str, keywords: list[str], trend_keywords: list[str]) -> float:
    """0-100 for overlap between clip content and trending keyword list."""
    if not trend_keywords:
        return 50.0  # neutral when trends are unavailable (offline default)
    haystack = (text + " " + " ".join(keywords)).lower()
    hits = sum(1 for k in trend_keywords if k.lower() in haystack)
    return round(min(hits / 3.0, 1.0) * 100.0, 1)


def silence_penalty_score(pause_fraction: float) -> float:
    """0-100 penalty magnitude from the fraction of clip spent silent."""
    return round(min(pause_fraction / 0.4, 1.0) * 100.0, 1)


def weighted_total(scores: dict, weights: dict, penalties: dict) -> float:
    """Final 0-100 composite. `scores` holds all signal values incl. penalties."""
    total = 0.0
    for signal, weight in weights.items():
        total += weight * scores.get(signal, 50.0)
    total -= penalties.get("silence", 0.0) * scores.get("silence_penalty", 0.0)
    total -= penalties.get("duplicate", 0.0) * scores.get("duplicate_penalty", 0.0)
    return round(max(total, 0.0), 2)


def overlap_fraction(a: dict, b: dict) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return inter / max(shorter, 0.01)


def select_top(
    candidates: list[dict],
    weights: dict,
    penalties: dict,
    max_shorts: int,
    max_overlap: float = 0.15,
    duplicate_threshold: float = 0.55,
    hard_duplicate_threshold: float = 0.85,
) -> list[dict]:
    """Greedy ranked selection with overlap rejection and topic-dup handling.

    Similarity above `duplicate_threshold` applies the soft penalty; above
    `hard_duplicate_threshold` the candidate is rejected outright (a penalty
    alone cannot stop a near-identical clip with a high base score).
    Re-scores remaining candidates after each pick so penalties reflect what
    is already selected.
    """
    pool = list(candidates)
    selected: list[dict] = []
    while pool and len(selected) < max_shorts:
        for cand in pool:
            dup = 0.0
            for chosen in selected:
                sim = jaccard_similarity(cand["text"], chosen["text"])
                if sim > duplicate_threshold:
                    dup = max(dup, min((sim - duplicate_threshold) / 0.45, 1.0) * 100.0)
            cand["scores"]["duplicate_penalty"] = round(dup, 1)
            cand["final_score"] = weighted_total(cand["scores"], weights, penalties)

        pool.sort(key=lambda c: c["final_score"], reverse=True)
        best = pool.pop(0)
        if any(overlap_fraction(best, s) > max_overlap for s in selected):
            continue
        if any(
            jaccard_similarity(best["text"], s["text"]) >= hard_duplicate_threshold
            for s in selected
        ):
            continue
        selected.append(best)
    selected.sort(key=lambda c: c["final_score"], reverse=True)
    return selected
