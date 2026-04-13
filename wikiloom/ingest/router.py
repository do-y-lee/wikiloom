"""File type detection and extractor dispatch."""

from __future__ import annotations

from pathlib import Path

from wikiloom.ingest.extractors import (
    BaseExtractor,
    CodeExtractor,
    ImageExtractor,
    MarkdownExtractor,
    OfficeExtractor,
    PDFExtractor,
    WebExtractor,
)

EXTENSION_MAP: dict[str, type[BaseExtractor]] = {
    # Markdown / text
    ".md": MarkdownExtractor,
    ".txt": MarkdownExtractor,
    ".rst": MarkdownExtractor,
    # PDF
    ".pdf": PDFExtractor,
    # Images
    ".png": ImageExtractor,
    ".jpg": ImageExtractor,
    ".jpeg": ImageExtractor,
    ".webp": ImageExtractor,
    ".gif": ImageExtractor,
    # Office
    ".docx": OfficeExtractor,
    ".pptx": OfficeExtractor,
    # Code — all read as plain text with language context
    ".py": CodeExtractor,
    ".js": CodeExtractor,
    ".ts": CodeExtractor,
    ".tsx": CodeExtractor,
    ".jsx": CodeExtractor,
    ".go": CodeExtractor,
    ".rs": CodeExtractor,
    ".java": CodeExtractor,
    ".rb": CodeExtractor,
    ".cs": CodeExtractor,
    ".cpp": CodeExtractor,
    ".c": CodeExtractor,
    ".sh": CodeExtractor,
    ".sql": CodeExtractor,
    ".tf": CodeExtractor,
    ".hcl": CodeExtractor,
    ".proto": CodeExtractor,
    ".graphql": CodeExtractor,
    # Config files — also code extractor with context
    ".yaml": CodeExtractor,
    ".yml": CodeExtractor,
    ".json": CodeExtractor,
    ".toml": CodeExtractor,
    ".dockerfile": CodeExtractor,
}


def route(path: Path | str) -> BaseExtractor:
    """Return an extractor instance for the given path or URL.

    Falls back to MarkdownExtractor for unknown extensions so that any
    text-bearing file can still be ingested as plain text.
    """
    if isinstance(path, str) and path.startswith(("http://", "https://")):
        return WebExtractor()

    p = Path(path) if isinstance(path, str) else path

    if str(p).startswith(("http://", "https://")):
        return WebExtractor()

    if p.name.lower() == "dockerfile":
        return CodeExtractor()

    extractor_cls = EXTENSION_MAP.get(p.suffix.lower(), MarkdownExtractor)
    return extractor_cls()


def can_handle(path: Path | str) -> bool:
    """Return True if any extractor (other than the markdown fallback) handles this path."""
    if isinstance(path, str) and path.startswith(("http://", "https://")):
        return True
    p = Path(path) if isinstance(path, str) else path
    if p.name.lower() == "dockerfile":
        return True
    return p.suffix.lower() in EXTENSION_MAP
