from app.analytics.energy import pause_ratio, speech_rate


def _words(times):
    return [{"word": "w", "start": t, "end": t + 0.2, "prob": 0.9} for t in times]


def test_speech_rate():
    words = _words([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert abs(speech_rate(words, 0.0, 3.0) - 2.0) < 1e-9


def test_pause_ratio_dense_speech():
    words = _words([i * 0.4 for i in range(25)])
    assert pause_ratio(words, 0.0, 10.0) == 0.0


def test_pause_ratio_dead_air():
    words = _words([0.0, 9.0])
    ratio = pause_ratio(words, 0.0, 10.0)
    assert ratio > 0.8


def test_pause_ratio_empty():
    assert pause_ratio([], 0.0, 10.0) == 1.0
