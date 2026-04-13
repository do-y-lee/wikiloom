"""Markdown / plain text extractor."""

from __future__ import annotations

from pathlib import Path

from wikiloom.ingest.extractors.base import BaseExtractor, ExtractedContent
from wikiloom.utils import estimate_tokens

MARKDOWN_EXTENSIONS = {".md", ".txt", ".rst"}


class MarkdownExtractor(BaseExtractor):
    """Reads markdown, plain text, and reStructuredText files."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in MARKDOWN_EXTENSIONS

    def extract(self, path: Path) -> ExtractedContent:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractedContent(
            text=text,
            metadata={"filename": path.name},
            source_path=path,
            content_type="markdown",
            extraction_method="plain-text",
            token_estimate=estimate_tokens(text),
        )
