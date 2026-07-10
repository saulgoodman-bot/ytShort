"""ASS subtitle generation from word-level timestamps (pure Python).

Two modes:
- plain lines (classic/minimal): one Dialogue per line of N words
- karaoke   (tiktok/modern):     one Dialogue per WORD, re-rendering the full
  line with the active word in the highlight color — the word-pop style.
Timestamps are shifted so 0:00 == clip start.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.subtitles.styles import SubtitleStyle, hex_to_ass

log = logging.getLogger(__name__)


def _ass_time(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _header(style: SubtitleStyle, play_w: int, play_h: int) -> str:
    bold = -1 if style.bold else 0
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font},{style.font_size},{hex_to_ass(style.primary_color)},{hex_to_ass(style.highlight_color)},{hex_to_ass(style.outline_color)},&H80000000,{bold},0,0,0,100,100,0,0,1,{style.outline},{style.shadow},2,60,60,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def group_words_into_lines(words: list[dict], max_words: int) -> list[list[dict]]:
    """Split a word stream into subtitle lines of at most max_words."""
    lines: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        current.append(w)
        if len(current) >= max_words:
            lines.append(current)
            current = []
    if current:
        lines.append(current)
    return lines


def _display(word: str, style: SubtitleStyle) -> str:
    text = word.upper() if style.uppercase else word
    return text.replace("{", "(").replace("}", ")")


def build_ass(
    words: list[dict],
    clip_start: float,
    clip_end: float,
    style: SubtitleStyle,
    play_w: int = 1080,
    play_h: int = 1920,
) -> str:
    """Render the full ASS document for one clip."""
    clip_words = [
        w for w in words if clip_start <= w["start"] < clip_end and w["word"].strip()
    ]
    events: list[str] = []
    highlight = hex_to_ass(style.highlight_color).replace("&H", "&H00")

    for line in group_words_into_lines(clip_words, style.max_words_per_line):
        line_start = line[0]["start"] - clip_start
        line_end = min(line[-1]["end"], clip_end) - clip_start
        if line_end <= line_start:
            continue

        if style.karaoke:
            # One event per word: whole line shown, active word highlighted.
            for k, active in enumerate(line):
                w_start = active["start"] - clip_start
                w_end = (line[k + 1]["start"] - clip_start) if k + 1 < len(line) else line_end
                if w_end <= w_start:
                    continue
                parts = []
                for m, w in enumerate(line):
                    token = _display(w["word"], style)
                    if m == k:
                        parts.append(
                            "{\\c" + highlight + "\\fscx108\\fscy108}" + token + "{\\r}"
                        )
                    else:
                        parts.append(token)
                events.append(
                    f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},"
                    f"Default,,0,0,0,,{' '.join(parts)}"
                )
        else:
            text = " ".join(_display(w["word"], style) for w in line)
            events.append(
                f"Dialogue: 0,{_ass_time(line_start)},{_ass_time(line_end)},"
                f"Default,,0,0,0,,{text}"
            )

    return _header(style, play_w, play_h) + "\n".join(events) + "\n"


def write_ass(
    out_path: Path,
    words: list[dict],
    clip_start: float,
    clip_end: float,
    style: SubtitleStyle,
    play_w: int = 1080,
    play_h: int = 1920,
) -> Path:
    doc = build_ass(words, clip_start, clip_end, style, play_w, play_h)
    out_path.write_text(doc, encoding="utf-8")
    log.info("Subtitles -> %s", out_path.name)
    return out_path
