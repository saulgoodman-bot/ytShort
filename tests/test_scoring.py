from app.analytics.scoring import (
    overlap_fraction, select_top, silence_penalty_score, trend_score,
    visual_score, weighted_total,
)

WEIGHTS = {
    "emotion": 0.20, "curiosity": 0.20, "storytelling": 0.15, "keywords": 0.10,
    "energy": 0.10, "visual": 0.10, "trend": 0.10, "engagement": 0.05,
}
PENALTIES = {"silence": 0.05, "duplicate": 0.05}


def _cand(start, end, text, base=70.0):
    return {
        "start": start, "end": end, "text": text,
        "scores": {k: base for k in WEIGHTS} | {"silence_penalty": 0.0},
    }


def test_weighted_total_perfect_scores():
    scores = {k: 100.0 for k in WEIGHTS}
    scores["silence_penalty"] = 0.0
    scores["duplicate_penalty"] = 0.0
    assert weighted_total(scores, WEIGHTS, PENALTIES) == 100.0


def test_penalties_reduce_score():
    scores = {k: 100.0 for k in WEIGHTS}
    scores["silence_penalty"] = 100.0
    scores["duplicate_penalty"] = 100.0
    assert weighted_total(scores, WEIGHTS, PENALTIES) == 90.0


def test_select_top_rejects_overlap():
    a = _cand(0, 30, "unique topic alpha beta gamma", base=90)
    b = _cand(10, 40, "different words entirely here now", base=85)  # overlaps a
    c = _cand(100, 130, "third completely separate subject matter", base=60)
    picked = select_top([a, b, c], WEIGHTS, PENALTIES, max_shorts=3, max_overlap=0.15)
    spans = [(p["start"], p["end"]) for p in picked]
    assert (0, 30) in spans and (100, 130) in spans
    assert (10, 40) not in spans


def test_select_top_penalizes_duplicates():
    a = _cand(0, 30, "the market crashed because of interest rates today", base=90)
    dup = _cand(60, 90, "the market crashed because of interest rates today", base=89)
    other = _cand(120, 150, "cooking biryani needs patience and good rice", base=70)
    picked = select_top([a, dup, other], WEIGHTS, PENALTIES, max_shorts=2)
    texts = [p["text"] for p in picked]
    assert "biryani" in texts[0] + texts[1]


def test_overlap_fraction():
    a = {"start": 0.0, "end": 30.0}
    b = {"start": 15.0, "end": 45.0}
    assert abs(overlap_fraction(a, b) - 0.5) < 1e-9
    assert overlap_fraction(a, {"start": 40.0, "end": 60.0}) == 0.0


def test_visual_score_bounds():
    assert visual_score([], 0, 30) == 0.0
    dense = [float(t) for t in range(0, 30, 2)]
    assert visual_score(dense, 0, 30) == 100.0


def test_trend_score_neutral_when_offline():
    assert trend_score("anything", [], []) == 50.0
    assert trend_score("AI stocks rally", ["ai"], ["ai", "stocks", "rally"]) == 100.0


def test_silence_penalty_scaling():
    assert silence_penalty_score(0.0) == 0.0
    assert silence_penalty_score(0.4) == 100.0
    assert silence_penalty_score(1.0) == 100.0
