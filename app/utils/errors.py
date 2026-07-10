"""Pipeline exception types."""


class AutoShortsError(Exception):
    """Base class for all pipeline errors."""


class DependencyError(AutoShortsError):
    """A required external tool or model is missing."""


class VideoImportError(AutoShortsError):
    """Input video is unreadable, corrupted, or unsupported."""


class TranscriptionError(AutoShortsError):
    """Speech recognition failed."""


class LLMError(AutoShortsError):
    """Local LLM call failed or returned unusable output."""


class RenderError(AutoShortsError):
    """FFmpeg rendering failed."""
