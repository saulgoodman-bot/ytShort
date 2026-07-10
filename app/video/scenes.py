"""Scene detection via PySceneDetect. Returns cut timestamps in seconds."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def detect_scenes(
    video: Path,
    detector_name: str = "content",
    threshold: float = 27.0,
    min_scene_len_seconds: float = 1.0,
) -> list[float]:
    """Return sorted list of scene-cut timestamps (seconds)."""
    from scenedetect import AdaptiveDetector, ContentDetector, detect  # lazy import

    if detector_name == "adaptive":
        detector = AdaptiveDetector()
    else:
        detector = ContentDetector(threshold=threshold)

    scene_list = detect(str(video), detector)
    cuts: list[float] = []
    for start, _end in scene_list:
        t = start.get_seconds()
        if t > 0 and (not cuts or t - cuts[-1] >= min_scene_len_seconds):
            cuts.append(round(t, 3))
    log.info("Scene detection: %d cuts", len(cuts))
    return cuts


def cuts_in_range(cuts: list[float], start: float, end: float) -> list[float]:
    return [c for c in cuts if start <= c <= end]
