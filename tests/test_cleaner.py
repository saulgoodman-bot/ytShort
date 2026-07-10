from app.ai.cleaner import clean_transcript, transcript_text


def _seg(text, start, end, conf=0.9, words=None):
    return {"text": text, "start": start, "end": end, "confidence": conf,
            "words": words or []}


def test_fillers_removed_and_sentences_merged():
    raw = {
        "language": "te",
        "segments": [
            _seg("um this is", 0.0, 1.0, words=[
                {"word": "um", "start": 0.0, "end": 0.2, "prob": 0.9},
                {"word": "this", "start": 0.3, "end": 0.5, "prob": 0.9},
                {"word": "is", "start": 0.5, "end": 0.7, "prob": 0.9},
            ]),
            _seg("a test.", 1.0, 2.0, words=[
                {"word": "a", "start": 1.0, "end": 1.1, "prob": 0.9},
                {"word": "test.", "start": 1.1, "end": 1.5, "prob": 0.9},
            ]),
        ],
    }
    cleaned = clean_transcript(raw)
    assert len(cleaned["sentences"]) == 1
    s = cleaned["sentences"][0]
    assert "um" not in s["text"].lower().split()
    assert s["start"] == 0.3
    assert s["end"] == 1.5


def test_gap_splits_sentences():
    raw = {
        "language": "te",
        "segments": [
            _seg("first part", 0.0, 1.0, words=[
                {"word": "first", "start": 0.0, "end": 0.4, "prob": 0.9},
                {"word": "part", "start": 0.4, "end": 0.8, "prob": 0.9},
            ]),
            _seg("second part.", 5.0, 6.0, words=[
                {"word": "second", "start": 5.0, "end": 5.4, "prob": 0.9},
                {"word": "part.", "start": 5.4, "end": 5.8, "prob": 0.9},
            ]),
        ],
    }
    cleaned = clean_transcript(raw)
    assert len(cleaned["sentences"]) == 2


def test_word_synthesis_when_missing():
    raw = {"language": "te", "segments": [_seg("one two three.", 0.0, 3.0)]}
    cleaned = clean_transcript(raw)
    words = cleaned["sentences"][0]["words"]
    assert len(words) == 3
    assert words[0]["start"] == 0.0
    assert words[-1]["end"] == 3.0


def test_transcript_text_numbering():
    raw = {"language": "te", "segments": [_seg("hello world.", 0.0, 1.0)]}
    text = transcript_text(clean_transcript(raw))
    assert text.startswith("[0] (0.0-1.0s)")
