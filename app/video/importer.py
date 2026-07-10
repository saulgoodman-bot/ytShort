"""Module 1: video import + metadata extraction."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from app.utils import ffmpeg
from app.utils.errors import VideoImportError

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class VideoMeta:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    video_codec: str
    audio_codec: str | None
    audio_channels: int
    audio_sample_rate: int

    def as_dict(self) -> dict:
        return asdict(self)


def import_video(path: Path) -> VideoMeta:
    if not path.exists():
        raise VideoImportError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoImportError(
            f"Unsupported container '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    try:
        info = ffmpeg.probe(path)
    except Exception as exc:
        raise VideoImportError(f"Could not read '{path}': {exc}") from exc

    vstream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    astream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if vstream is None:
        raise VideoImportError(f"No video stream in '{path}'")
    if astream is None:
        raise VideoImportError(f"No audio stream in '{path}' — nothing to transcribe")

    num, _, den = (vstream.get("avg_frame_rate") or "30/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0

    meta = VideoMeta(
        path=str(path),
        duration=float(info["format"].get("duration", 0.0)),
        fps=round(fps, 3),
        width=int(vstream["width"]),
        height=int(vstream["height"]),
        video_codec=vstream.get("codec_name", "unknown"),
        audio_codec=astream.get("codec_name"),
        audio_channels=int(astream.get("channels", 1)),
        audio_sample_rate=int(astream.get("sample_rate", 0)),
    )
    log.info(
        "Imported %s: %.1fs %dx%d @%.2ffps (%s/%s)",
        path.name, meta.duration, meta.width, meta.height, meta.fps,
        meta.video_codec, meta.audio_codec,
    )
    return meta
