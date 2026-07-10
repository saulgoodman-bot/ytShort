"""Final Short rendering: trim -> crop -> scale -> burn subtitles -> encode."""
from __future__ import annotations

import logging
from pathlib import Path

from app.utils import ffmpeg
from app.video.cropper import CropPlan, dynamic_x_expression

log = logging.getLogger(__name__)


def _escape_filter_path(path: Path) -> str:
    # ffmpeg filter args need ':' and '\' escaped; keep it simple for POSIX.
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def render_short(
    source: Path,
    out_path: Path,
    start: float,
    end: float,
    crop: CropPlan,
    subtitle_file: Path | None,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    video_bitrate: str = "8M",
    audio_bitrate: str = "192k",
    encoder: str = "auto",
) -> Path:
    if crop.is_dynamic:
        # Quote the expression: it contains commas, which otherwise split
        # the filtergraph. Single quotes protect them inside -vf.
        x_expr = "'" + dynamic_x_expression(crop) + "'"
    else:
        x_expr = str(crop.x)
    vf_parts = [
        f"crop={crop.crop_w}:{crop.crop_h}:{x_expr}:0",
        f"scale={width}:{height}:flags=lanczos",
        f"fps={fps}",
    ]
    if subtitle_file is not None:
        vf_parts.append(f"ass='{_escape_filter_path(subtitle_file)}'")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(source),
        "-vf", ",".join(vf_parts),
        *ffmpeg.pick_encoder(encoder),
        "-b:v", video_bitrate,
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(out_path),
    ]
    ffmpeg.run(cmd, desc=f"render {out_path.name}")
    log.info("Rendered %s (%.1fs)", out_path.name, end - start)
    return out_path
