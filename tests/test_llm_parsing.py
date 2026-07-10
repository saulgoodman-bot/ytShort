import pytest

from app.ai.llm import parse_json_loosely
from app.utils.errors import LLMError


def test_clean_json():
    assert parse_json_loosely('{"a": 1}') == {"a": 1}


def test_fenced_json():
    raw = 'Here you go:\n```json\n{"clips": []}\n```\nDone!'
    assert parse_json_loosely(raw) == {"clips": []}


def test_embedded_json():
    raw = 'Sure! {"scores": [{"id": 0}]} hope that helps'
    assert parse_json_loosely(raw) == {"scores": [{"id": 0}]}


def test_garbage_raises():
    with pytest.raises(LLMError):
        parse_json_loosely("no json here at all")
