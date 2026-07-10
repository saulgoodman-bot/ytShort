"""Smart 16:9 -> 9:16 crop planning.

Pure geometry lives here (unit-testable). The renderer consumes CropPlan.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

TARGET_ASPECT = 9.0 / 16.0


@dataclass
class CropPlan:
    crop_w: int
    crop_h: int
    # Static crop: single x offset. Dynamic: list of (time, x) keyframes.
    x: int
    keyframes: list[tuple[float, int]] | None = None

    @property
    def is_dynamic(self) -> bool:
        return bool(self.keyframes and len(self.keyframes) > 1)


def _even(v: int) -> int:
    return v - (v % 2)


def crop_dimensions(src_w: int, src_h: int) -> tuple[int, int]:
    """Largest 9:16 window that fits inside the source frame."""
    crop_h = src_h
    crop_w = _even(min(src_w, int(round(crop_h * TARGET_ASPECT))))
    if crop_w > src_w:  # narrow sources
        crop_w = _even(src_w)
        crop_h = _even(int(round(crop_w / TARGET_ASPECT)))
    return crop_w, min(crop_h, src_h)


def clamp_x(center_x_norm: float, src_w: int, crop_w: int) -> int:
    """Convert a normalized face center to a clamped, even x offset."""
    x = int(round(center_x_norm * src_w - crop_w / 2.0))
    x = max(0, min(x, src_w - crop_w))
    return _even(x)


def plan_crop(
    src_w: int,
    src_h: int,
    face_times: list[float],
    face_centers_x: list[float],
    mode: str = "smart_static",
    keyframe_interval: float = 2.0,
    clip_start: float = 0.0,
) -> CropPlan:
    crop_w, crop_h = crop_dimensions(src_w, src_h)

    if mode == "center" or not face_centers_x:
        return CropPlan(crop_w, crop_h, clamp_x(0.5, src_w, crop_w))

    if mode == "smart_dynamic" and len(face_centers_x) > 1:
        keyframes: list[tuple[float, int]] = []
        last_t = -1e9
        for t, cx in zip(face_times, face_centers_x):
            if t - last_t >= keyframe_interval:
                keyframes.append((round(t - clip_start, 3), clamp_x(cx, src_w, crop_w)))
                last_t = t
        median_x = clamp_x(statistics.median(face_centers_x), src_w, crop_w)
        return CropPlan(crop_w, crop_h, median_x, keyframes=keyframes)

    # smart_static: median of the smoothed track — robust to outliers/jitter.
    median_x = clamp_x(statistics.median(face_centers_x), src_w, crop_w)
    return CropPlan(crop_w, crop_h, median_x)


def dynamic_x_expression(plan: CropPlan) -> str:
    """Build an ffmpeg crop-x expression that linearly interpolates keyframes.

    Produces nested if(lt(t,k),...) expressions. Kept bounded by the
    keyframe interval so expressions stay manageable.
    """
    if not plan.is_dynamic:
        return str(plan.x)
    kfs = plan.keyframes or []
    expr = str(kfs[-1][1])  # after last keyframe hold final x
    for (t0, x0), (t1, x1) in zip(reversed(kfs[:-1]), reversed(kfs[1:])):
        span = max(t1 - t0, 1e-3)
        lerp = f"({x0}+({x1}-{x0})*(t-{t0})/{span:.3f})"
        expr = f"if(lt(t,{t1}),{lerp},{expr})"
    first_t, first_x = kfs[0]
    return f"if(lt(t,{first_t}),{first_x},{expr})"
