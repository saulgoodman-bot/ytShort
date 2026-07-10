"""Per-Short publishing metadata: titles, description, hashtags via local LLM."""
from __future__ import annotations

import logging

from app.ai.llm import OllamaClient
from app.utils.errors import LLMError

log = logging.getLogger(__name__)

_SYSTEM = (
    "You write high-CTR YouTube Shorts metadata for Telugu audiences. "
    "Mix Telugu and English naturally. Answer ONLY with JSON."
)

_PROMPT = """Create publishing metadata for this YouTube Short.

CLIP TRANSCRIPT:
{text}

VIDEO TOPIC: {topic}
CLIP HOOK: {hook}

Return JSON exactly:
{{
  "titles": ["5 options, each under 100 characters, curiosity-driven, no clickbait lies"],
  "description": "2-4 sentences, SEO keywords woven in, ends with a call to action",
  "hashtags": ["{n_hashtags} tags mixing #Telugu general tags, topic tags, and English tags, no spaces"]
}}"""


def generate_metadata(
    llm: OllamaClient,
    clip: dict,
    topic: str,
    n_titles: int = 5,
    n_hashtags: int = 15,
) -> dict:
    fallback = {
        "titles": [clip.get("hook_text", "")[:100] or "Telugu Short"],
        "description": clip["text"][:300],
        "hashtags": ["#Shorts", "#Telugu", "#TeluguShorts", "#Trending", "#Viral"],
    }
    try:
        result = llm.generate_json(
            _PROMPT.format(
                text=clip["text"][:1500],
                topic=topic,
                hook=clip.get("hook_text", ""),
                n_hashtags=n_hashtags,
            ),
            system=_SYSTEM,
        )
    except LLMError as exc:
        log.warning("Metadata generation failed: %s", exc)
        return fallback
    if not isinstance(result, dict):
        return fallback

    titles = [str(t)[:100] for t in result.get("titles", []) if str(t).strip()][:n_titles]
    hashtags = []
    for tag in result.get("hashtags", []):
        tag = str(tag).strip().replace(" ", "")
        if tag and not tag.startswith("#"):
            tag = "#" + tag
        if tag and tag not in hashtags:
            hashtags.append(tag)
    return {
        "titles": titles or fallback["titles"],
        "description": str(result.get("description", "")) or fallback["description"],
        "hashtags": hashtags[:n_hashtags] or fallback["hashtags"],
    }
