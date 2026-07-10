"""Speaking-energy analysis from the 16-bit mono WAV (stdlib only)."""
from __future__ import annotations

import array
import logging
import math
import wave
from pathlib import Path

log = logging.getLogger(__name__)

_FRAME_SECONDS = 0.25


class AudioEnergy:
    """Loads per-frame RMS once; answers per-range energy queries."""

    def __init__(self, wav_path: Path):
        self.frame_rms: list[float] = []
        self.frame_seconds = _FRAME_SECONDS
        try:
            with wave.open(str(wav_path), "rb") as wf:
                rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                if sampwidth != 2:
                    log.warning("Expected 16-bit WAV, got %d bytes/sample", sampwidth)
                    return
                chunk = int(rate * _FRAME_SECONDS) * n_channels
                while True:
                    raw = wf.readframes(chunk // n_channels)
                    if not raw:
                        break
                    samples = array.array("h", raw)
                    if not samples:
                        break
                    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                    self.frame_rms.append(rms)
        except (OSError, wave.Error) as exc:
            log.warning("Could not analyze audio energy: %s", exc)

    def _slice(self, start: float, end: float) -> list[float]:
        i = int(start / self.frame_seconds)
        j = max(i + 1, int(end / self.frame_seconds))
        return self.frame_rms[i:j]

    def loudness_stats(self, start: float, end: float) -> tuple[float, float]:
        """(mean_rms, coefficient_of_variation) for the range."""
        frames = self._slice(start, end)
        if not frames:
            return 0.0, 0.0
        mean = sum(frames) / len(frames)
        if mean <= 0:
            return 0.0, 0.0
        var = sum((f - mean) ** 2 for f in frames) / len(frames)
        return mean, math.sqrt(var) / mean


def speech_rate(words: list[dict], start: float, end: float) -> float:
    """Words per second within [start, end]."""
    n = sum(1 for w in words if start <= w["start"] < end)
    span = max(end - start, 0.01)
    return n / span


def pause_ratio(words: list[dict], start: float, end: float, gap: float = 0.7) -> float:
    """Fraction of the clip spent in silences longer than `gap` seconds."""
    inside = sorted(
        (w for w in words if start <= w["start"] < end), key=lambda w: w["start"]
    )
    if len(inside) < 2:
        return 1.0 if not inside else 0.0
    silence = 0.0
    lead = inside[0]["start"] - start
    if lead > gap:
        silence += lead
    for prev, nxt in zip(inside, inside[1:]):
        g = nxt["start"] - prev["end"]
        if g > gap:
            silence += g
    tail = end - inside[-1]["end"]
    if tail > gap:
        silence += tail
    return min(silence / max(end - start, 0.01), 1.0)


def energy_score(
    audio: AudioEnergy, words: list[dict], start: float, end: float
) -> float:
    """0-100 speaking-energy score: rate + loudness variation - dead air."""
    rate = speech_rate(words, start, end)
    # 2.2-3.5 words/sec is lively speech for Telugu/mixed content.
    rate_score = max(0.0, min((rate - 1.0) / 2.5, 1.0))
    _mean, cov = audio.loudness_stats(start, end)
    variation_score = max(0.0, min(cov / 0.8, 1.0))
    pauses = pause_ratio(words, start, end)
    score = 100.0 * (0.5 * rate_score + 0.3 * variation_score + 0.2 * (1.0 - pauses))
    return round(score, 1)
