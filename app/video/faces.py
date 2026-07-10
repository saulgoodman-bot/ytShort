"""Face detection & tracking with MediaPipe.

Samples frames inside a clip's time range and returns a smoothed horizontal
face-center track (normalized 0..1) used by the cropper.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    times: list[float]        # absolute timestamps (s)
    centers_x: list[float]    # normalized 0..1 face center per sample
    centers_y: list[float]
    detection_rate: float     # fraction of sampled frames with a face


def ema_smooth(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average; pure function so it is unit-testable."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * out[-1] + (1.0 - alpha) * v)
    return out


def track_faces(
    video: Path,
    start: float,
    end: float,
    sample_fps: float = 2.0,
    min_confidence: float = 0.5,
    smoothing_alpha: float = 0.85,
) -> FaceTrack:
    import cv2  # lazy heavy imports
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.warning("Could not open video for face tracking; using center fallback")
        return FaceTrack([start], [0.5], [0.42], 0.0)

    times: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    detected = 0
    total = 0
    last_x, last_y = 0.5, 0.42  # slight upper bias: faces sit above center

    step = 1.0 / max(sample_fps, 0.1)
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_confidence
    ) as detector:
        t = start
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            if result.detections:
                # Choose the largest face (the active speaker heuristic).
                best = max(
                    result.detections,
                    key=lambda d: d.location_data.relative_bounding_box.width
                    * d.location_data.relative_bounding_box.height,
                )
                box = best.location_data.relative_bounding_box
                last_x = min(max(box.xmin + box.width / 2.0, 0.0), 1.0)
                last_y = min(max(box.ymin + box.height / 2.0, 0.0), 1.0)
                detected += 1
            times.append(round(t, 3))
            xs.append(last_x)
            ys.append(last_y)
            t += step
    cap.release()

    if not times:
        return FaceTrack([start], [0.5], [0.42], 0.0)
    return FaceTrack(
        times=times,
        centers_x=ema_smooth(xs, smoothing_alpha),
        centers_y=ema_smooth(ys, smoothing_alpha),
        detection_rate=detected / max(total, 1),
    )
