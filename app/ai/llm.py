"""Ollama client: JSON-oriented prompting against a local model."""
from __future__ import annotations

import json
import logging
import re

import requests

from app.utils.errors import LLMError

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        temperature: float = 0.4,
        timeout: int = 300,
        max_retries: int = 2,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    # ---- health -----------------------------------------------------------
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def has_model(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            names = [m["name"] for m in r.json().get("models", [])]
            base = self.model.split(":")[0]
            return any(n == self.model or n.split(":")[0] == base for n in names)
        except requests.RequestException:
            return False

    # ---- generation -------------------------------------------------------
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = True) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    f"{self.host}/api/generate", json=payload, timeout=self.timeout
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except requests.RequestException as exc:
                last_error = exc
                log.warning("Ollama call failed (attempt %d): %s", attempt + 1, exc)
        raise LLMError(
            f"Ollama unreachable or failing at {self.host} (model {self.model}): {last_error}"
        )

    def generate_json(self, prompt: str, system: str | None = None) -> dict | list:
        raw = self.generate(prompt, system=system, json_mode=True)
        return parse_json_loosely(raw)


def parse_json_loosely(raw: str) -> dict | list:
    """Parse LLM output into JSON, tolerating chatter around the payload."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # strip code fences
    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # first {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Could not parse JSON from LLM output: {raw[:300]!r}")
