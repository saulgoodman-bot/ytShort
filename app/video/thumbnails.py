"""Thumbnail generation: pick the sharpest face frame, overlay hook text."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/NotoSansTelugu-Bold.ttf",
    "/Library/Fonts/NotoSansTelugu-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int):
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def best_frame(video: Path, start: float, end: float, samples: int = 8):
    """Return the sharpest sampled frame (BGR ndarray) within [start, end]."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    best, best_score = None, -1.0
    if not cap.isOpened():
        return None
    for i in range(samples):
        t = start + (end - start) * (i + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness > best_score:
            best, best_score = frame, sharpness
    cap.release()
    return best


def generate_thumbnail(
    video: Path,
    start: float,
    end: float,
    text: str,
    out_path: Path,
    overlay_text: bool = True,
    font_size: int = 110,
    width: int = 1080,
    height: int = 1920,
) -> Path | None:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    frame = best_frame(video, start, end)
    if frame is None:
        log.warning("No frame available for thumbnail")
        return None

    # Center-crop to 9:16 then resize.
    h, w = frame.shape[:2]
    crop_w = min(w, int(h * 9 / 16))
    x0 = (w - crop_w) // 2
    frame = frame[:, x0 : x0 + crop_w]
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if overlay_text and text:
        draw = ImageDraw.Draw(img)
        font = _load_font(font_size)
        words = text.split()
        lines, line = [], ""
        for word in words:
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=font) > width * 0.9 and line:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        lines = lines[:3]
        y = int(height * 0.72)
        for ln in lines:
            tw = draw.textlength(ln, font=font)
            x = (width - tw) / 2
            # Outline for readability.
            for dx in (-4, 0, 4):
                for dy in (-4, 0, 4):
                    draw.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0))
            draw.text((x, y), ln, font=font, fill=(255, 212, 0))
            y += int(font_size * 1.15)

    img.save(out_path, quality=92)
    log.info("Thumbnail -> %s", out_path.name)
    return out_path
