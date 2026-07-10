"""Central logging: console + per-run logfile under logs/."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(logs_dir: Path, verbose: bool = False) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / time.strftime("autoshorts_%Y%m%d_%H%M%S.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    root.addHandler(console)

    filehandler = logging.FileHandler(logfile, encoding="utf-8")
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(filehandler)

    # Quiet noisy third-party loggers.
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logfile
