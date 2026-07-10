"""Small text utilities shared across modules (pure Python, testable)."""
from __future__ import annotations

import re

# Telugu + English filler tokens commonly produced by ASR.
FILLERS = {
    "um", "uh", "uhh", "umm", "hmm", "mmm", "aa", "ah", "haan", "matlab",
    "ante", "అంటే", "మ్", "ఆఁ", "అఀ", "హా",
}

_WORD_RE = re.compile(r"[\wఀ-౿']+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_filler(word: str) -> bool:
    return word.strip().strip(".,!?…").lower() in FILLERS


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_SENTENCE_END = re.compile(r"[.!?…।]\s*$")


def ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))
