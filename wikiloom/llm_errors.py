"""Exceptions raised by the LLM provider abstraction.

Mirrors the pattern in ``wikiloom/ingest/errors.py``: a typed base
class so callers can branch on category, with subclasses carrying
structured context as instance attributes.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all ``wikiloom.llm`` failures."""


class LLMProviderError(LLMError):
    """litellm / upstream provider raised during a completion call.

    Wraps the original exception so callers can retry or fail with a
    consistent type regardless of which provider is behind litellm.
    Inspect ``self.original`` for the underlying litellm exception
    (e.g. ``litellm.RateLimitError``, ``litellm.AuthenticationError``).
    """

    def __init__(
        self,
        model: str,
        call_type: str,
        original: BaseException,
    ) -> None:
        super().__init__(
            f"LLM provider error during {call_type} with model "
            f"{model!r}: {type(original).__name__}: {original}"
        )
        self.model = model
        self.call_type = call_type
        self.original = original


class LLMResponseFormatError(LLMError):
    """Model returned text that couldn't be parsed as the expected format.

    Raised by ``LLMClient.synthesize`` when JSON-mode output arrives
    but isn't a valid JSON object. The raw text is preserved on the
    exception so debuggers can see exactly what the model produced.
    """

    def __init__(
        self,
        model: str,
        raw_text: str,
        parse_error: str,
    ) -> None:
        super().__init__(
            f"Model {model!r} returned unparseable structured output: "
            f"{parse_error}. Raw text preserved on exception "
            f"(raw_text attribute)."
        )
        self.model = model
        self.raw_text = raw_text
        self.parse_error = parse_error
