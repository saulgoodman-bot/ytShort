"""AutoShorts CLI.

  python -m app check                      # verify environment
  python -m app generate input/video.mp4   # produce Shorts
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from app.utils.config import load_config, resolve_dir
from app.utils.logging_setup import setup_logging


def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config()
    ok = True

    def report(label: str, good: bool, hint: str = "") -> None:
        nonlocal ok
        mark = "OK " if good else "FAIL"
        print(f"[{mark}] {label}" + (f"  -> {hint}" if not good and hint else ""))
        ok = ok and good

    report("ffmpeg on PATH", shutil.which("ffmpeg") is not None, "brew install ffmpeg")
    report("ffprobe on PATH", shutil.which("ffprobe") is not None, "brew install ffmpeg")

    for module, hint in [
        ("faster_whisper", "pip install faster-whisper"),
        ("scenedetect", "pip install 'scenedetect[opencv]'"),
        ("mediapipe", "pip install mediapipe"),
        ("cv2", "pip install opencv-python"),
        ("PIL", "pip install pillow"),
        ("yaml", "pip install pyyaml"),
        ("requests", "pip install requests"),
    ]:
        try:
            __import__(module)
            report(f"python: {module}", True)
        except ImportError:
            report(f"python: {module}", False, hint)

    from app.ai.llm import OllamaClient

    llm = OllamaClient(host=cfg.llm.host, model=cfg.llm.model)
    reachable = llm.is_available()
    report(f"ollama at {cfg.llm.host}", reachable, "install from https://ollama.com then run: ollama serve")
    if reachable:
        report(
            f"ollama model '{cfg.llm.model}'", llm.has_model(),
            f"ollama pull {cfg.llm.model}",
        )
    print("\nEnvironment " + ("looks good." if ok else "has problems — fix the FAIL lines above."))
    return 0 if ok else 1


def cmd_generate(args: argparse.Namespace) -> int:
    overrides = {
        "clips.max_shorts": args.max_shorts,
        "clips.min_duration": args.min_duration,
        "clips.max_duration": args.max_duration,
        "subtitles.style": args.subtitle_style,
        "llm.model": args.llm_model,
        "whisper.model": args.whisper_model,
        "crop.mode": args.crop_mode,
    }
    cfg = load_config({k: v for k, v in overrides.items() if v is not None})
    logfile = setup_logging(resolve_dir(cfg, "logs_dir"), verbose=args.verbose)
    print(f"Log: {logfile}")

    from app.pipeline.orchestrator import Pipeline
    from app.utils.errors import AutoShortsError

    videos = [Path(v) for v in args.videos]
    exit_code = 0
    for video in videos:
        print(f"\n=== {video} ===")
        try:
            results = Pipeline(cfg).run(video)
        except AutoShortsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        if not results:
            print("No Shorts produced.")
            exit_code = 1
            continue
        print(f"\nProduced {len(results)} Shorts:")
        for r in results:
            title = (r.get("titles") or [""])[0]
            print(f"  {r['file']}  ({r['duration']}s, score {r['final_score']})")
            if title:
                print(f"    title: {title}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoshorts",
        description="Local AI YouTube Shorts generator for Telugu content.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify ffmpeg, python deps, ollama, models").set_defaults(
        func=cmd_check
    )

    g = sub.add_parser("generate", help="generate Shorts from one or more videos")
    g.add_argument("videos", nargs="+", help="input video file(s)")
    g.add_argument("--max-shorts", type=int, default=None)
    g.add_argument("--min-duration", type=float, default=None)
    g.add_argument("--max-duration", type=float, default=None)
    g.add_argument(
        "--subtitle-style", choices=["classic", "modern", "bold", "minimal", "tiktok"],
        default=None,
    )
    g.add_argument("--llm-model", default=None, help="ollama model name")
    g.add_argument("--whisper-model", default=None, help="tiny|base|small|medium|large-v3")
    g.add_argument(
        "--crop-mode", choices=["center", "smart_static", "smart_dynamic"], default=None
    )
    g.add_argument("-v", "--verbose", action="store_true")
    g.set_defaults(func=cmd_generate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
