from app.subtitles.generator import build_ass, group_words_into_lines
from app.subtitles.styles import PRESETS, hex_to_ass


def _words():
    return [
        {"word": "hello", "start": 10.0, "end": 10.4, "prob": 0.9},
        {"word": "world", "start": 10.5, "end": 10.9, "prob": 0.9},
        {"word": "this", "start": 11.0, "end": 11.3, "prob": 0.9},
        {"word": "rocks", "start": 11.4, "end": 11.8, "prob": 0.9},
        {"word": "outside", "start": 99.0, "end": 99.5, "prob": 0.9},  # beyond clip
    ]


def test_hex_to_ass_reverses_channels():
    assert hex_to_ass("#FFD400") == "&H0000D4FF"


def test_grouping():
    lines = group_words_into_lines(_words(), 2)
    assert [len(l) for l in lines] == [2, 2, 1]


def test_build_ass_shifts_times_and_filters_range():
    doc = build_ass(_words(), clip_start=10.0, clip_end=12.0, style=PRESETS["classic"])
    assert "outside" not in doc
    assert "0:00:00.00" in doc  # first word shifted to clip-relative zero
    assert "[Events]" in doc and "Dialogue:" in doc


def test_karaoke_emits_per_word_events():
    plain = build_ass(_words(), 10.0, 12.0, PRESETS["classic"])
    karaoke = build_ass(_words(), 10.0, 12.0, PRESETS["tiktok"])
    assert karaoke.count("Dialogue:") > plain.count("Dialogue:")
    assert "\\c&H00" in karaoke  # highlight override present


def test_uppercase_style():
    doc = build_ass(_words(), 10.0, 12.0, PRESETS["bold"])
    assert "HELLO" in doc
