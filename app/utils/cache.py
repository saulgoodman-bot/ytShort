"""Content-addressed cache for intermediate pipeline artifacts.

Every input video gets a cache directory keyed by a hash of its first/last
bytes + size (fast, avoids hashing multi-GB files fully). Stages read/write
JSON artifacts there so re-runs resume instead of recompute.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB


def video_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(_CHUNK))
        if size > 2 * _CHUNK:
            fh.seek(-_CHUNK, 2)
            h.update(fh.read(_CHUNK))
    return h.hexdigest()[:16]


class StageCache:
    def __init__(self, cache_root: Path, video_path: Path):
        self.dir = cache_root / f"{video_path.stem}_{video_fingerprint(video_path)}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def load_json(self, name: str) -> Any:
        with open(self.path(name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save_json(self, name: str, data: Any) -> Path:
        target = self.path(name)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        tmp.replace(target)
        log.debug("cache write: %s", target)
        return target
