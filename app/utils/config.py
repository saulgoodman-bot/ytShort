"""Configuration loading and access.

Loads config/default.yaml, overlays config/local.yaml if present, and exposes
a dot-accessible Config object. CLI flags may further override values.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG = PROJECT_ROOT / "config" / "local.yaml"


class Config:
    """Dot-accessible, read-mostly wrapper around a nested dict."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as exc:
            raise AttributeError(f"No config key '{key}'") from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a value by dotted path, e.g. 'clips.max_shorts'."""
        node = self._data
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(overrides: dict[str, Any] | None = None) -> Config:
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, "r", encoding="utf-8") as fh:
            local = yaml.safe_load(fh) or {}
        data = _deep_merge(data, local)
    cfg = Config(data)
    for dotted_key, value in (overrides or {}).items():
        if value is not None:
            cfg.set(dotted_key, value)
    return cfg


def resolve_dir(cfg: Config, name: str) -> Path:
    """Resolve an app.* directory relative to the project root and create it."""
    path = PROJECT_ROOT / getattr(cfg.app, name)
    path.mkdir(parents=True, exist_ok=True)
    return path
