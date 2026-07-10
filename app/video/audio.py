"""Module 2: audio extraction for ASR (16 kHz mono WAV, normalized, denoised)."""
from __future__ import annotations

import logging
from pathlib import Path

from app.utils import ffmpeg

log = logging.getLogger(__name__)


def extract_audio(
    video: Path,
    target: Path,
    sample_rate: int = 16000,
    denoise: bool = True,
    normalize: bool = True,
) -> Path:
    filters = []
    if denoise:
        filters.append("afftdn=nf=-25")
    if normalize:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_s16le", str(target)]
    ffmpeg.run(cmd, desc="audio extraction")
    log.info("Extracted audio -> %s", target.name)
    return target
