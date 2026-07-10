"""Thin wrapper around ffmpeg/ffprobe subprocess calls."""
from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
from pathlib import Path

from app.utils.errors import DependencyError, RenderError

log = logging.getLogger(__name__)


def require_binaries() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise DependencyError(
                f"'{binary}' not found on PATH. Install with: brew install ffmpeg"
            )


def run(cmd: list[str], desc: str = "ffmpeg") -> str:
    """Run a command, raising RenderError with stderr tail on failure."""
    log.debug("run: %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-15:])
        raise RenderError(f"{desc} failed (exit {proc.returncode}):\n{tail}")
    return proc.stdout


def probe(path: Path) -> dict:
    out = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        desc="ffprobe",
    )
    return json.loads(out)


def has_encoder(name: str) -> bool:
    try:
        out = run(["ffmpeg", "-hide_banner", "-encoders"], desc="ffmpeg -encoders")
    except RenderError:
        return False
    return name in out


def pick_encoder(preference: str = "auto") -> list[str]:
    """Return the video-encoder args for this platform."""
    if preference != "auto":
        return ["-c:v", preference]
    if platform.system() == "Darwin" and has_encoder("h264_videotoolbox"):
        # Hardware encode on Apple Silicon: fast and power-efficient.
        return ["-c:v", "h264_videotoolbox", "-allow_sw", "1"]
    return ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
