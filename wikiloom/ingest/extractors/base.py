"""Base extractor interface and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractedContent:
    """Content extracted from a source file or URL.

    Fields:
        text: Full extracted text. For multi-page sources, all pages
            are concatenated. For code, includes a language context prefix.
        metadata: Extractor-specific metadata. PDF extractors MUST populate
            metadata["pages"] as list[str] (one entry per page) so the
            chunker can fall back to page-boundary chunking. Code extractors
            populate metadata["language"] and metadata["filename"].
        source_path: The original source path (or URL).
        content_type: A short identifier like "markdown", "pdf", "code",
            "office", "image", "web".
        extraction_method: How the text was obtained, for diagnostics.
        token_estimate: Rough token count for budgeting.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    content_type: str = ""
    extraction_method: str = ""
    token_estimate: int = 0


class BaseExtractor:
    """Abstract base class for all extractors."""

    def can_handle(self, path: Path) -> bool:
        """Return True if this extractor handles the given file."""
        raise NotImplementedError

    def extract(self, path: Path) -> ExtractedContent:
        """Extract text content from the file."""
        raise NotImplementedError
