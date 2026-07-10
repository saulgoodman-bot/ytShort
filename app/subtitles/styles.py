"""Subtitle style presets. Colors are #RRGGBB; converted to ASS &HBBGGRR&."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SubtitleStyle:
    name: str
    font: str
    font_size: int
    primary_color: str
    highlight_color: str
    outline_color: str
    outline: int
    shadow: int
    bold: bool
    uppercase: bool
    karaoke: bool
    max_words_per_line: int
    margin_v: int


PRESETS: dict[str, SubtitleStyle] = {
    "classic": SubtitleStyle(
        "classic", "Noto Sans Telugu", 56, "#FFFFFF", "#FFFFFF", "#000000",
        2, 1, False, False, False, 7, 220,
    ),
    "modern": SubtitleStyle(
        "modern", "Noto Sans Telugu", 62, "#FFFFFF", "#4DD0E1", "#101010",
        3, 0, True, False, True, 5, 250,
    ),
    "bold": SubtitleStyle(
        "bold", "Noto Sans Telugu", 72, "#FFFFFF", "#FF5252", "#000000",
        4, 2, True, True, False, 4, 260,
    ),
    "minimal": SubtitleStyle(
        "minimal", "Noto Sans Telugu", 50, "#F5F5F5", "#F5F5F5", "#202020",
        1, 0, False, False, False, 8, 200,
    ),
    "tiktok": SubtitleStyle(
        "tiktok", "Noto Sans Telugu", 64, "#FFFFFF", "#FFD400", "#000000",
        3, 1, True, False, True, 4, 260,
    ),
}


def get_style(cfg_subtitles) -> SubtitleStyle:
    """Resolve preset then apply user config overrides."""
    preset = PRESETS.get(cfg_subtitles.get("style", "tiktok"), PRESETS["tiktok"])
    return replace(
        preset,
        font=cfg_subtitles.get("font", preset.font),
        font_size=int(cfg_subtitles.get("font_size", preset.font_size)),
        primary_color=cfg_subtitles.get("primary_color", preset.primary_color),
        highlight_color=cfg_subtitles.get("highlight_color", preset.highlight_color),
        outline_color=cfg_subtitles.get("outline_color", preset.outline_color),
        outline=int(cfg_subtitles.get("outline", preset.outline)),
        shadow=int(cfg_subtitles.get("shadow", preset.shadow)),
        karaoke=bool(cfg_subtitles.get("karaoke", preset.karaoke)),
        max_words_per_line=int(
            cfg_subtitles.get("max_words_per_line", preset.max_words_per_line)
        ),
        margin_v=int(cfg_subtitles.get("margin_v", preset.margin_v)),
    )


def hex_to_ass(color: str, alpha: str = "00") -> str:
    """#RRGGBB -> &HAABBGGRR (ASS is little-endian BGR with alpha)."""
    c = color.lstrip("#")
    r, g, b = c[0:2], c[2:4], c[4:6]
    return f"&H{alpha}{b}{g}{r}".upper()
