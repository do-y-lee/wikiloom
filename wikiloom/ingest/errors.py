"""Exceptions raised by the ingest pipeline boundary guards.

These are raised early, before any expensive work (raw copy,
chunking, LLM synthesis) so a user running ``wikiloom ingest`` on
bad input sees a clear message and no partial state on disk.

The CLI layer (``wikiloom/cli.py``) catches these and re-raises as
``click.ClickException`` for a clean stderr presentation. Library
callers of ``ingest()`` see the typed exceptions directly.
"""

from __future__ import annotations


class IngestError(Exception):
    """Base class for ingest-boundary failures."""


class FileTooLargeError(IngestError):
    """Source file exceeds ``[ingest] max_file_size_mb``."""

    def __init__(self, path: str, size_mb: float, limit_mb: int) -> None:
        super().__init__(
            f"Source {path!r} is {size_mb:.1f} MB, which exceeds the "
            f"configured limit of {limit_mb} MB. Raise "
            f"[ingest] max_file_size_mb in wikiloom.toml to allow it."
        )
        self.path = path
        self.size_mb = size_mb
        self.limit_mb = limit_mb


class EmptyExtractionError(IngestError):
    """Extractor returned no usable text.

    Typically a scanned PDF with no text layer, a blank document, or
    a URL whose fetched body had no extractable content.
    """

    def __init__(self, path: str, content_type: str, extracted_chars: int) -> None:
        super().__init__(
            f"Extractor produced only {extracted_chars} character(s) of "
            f"text from {path!r} (content_type={content_type!r}). This "
            f"often means a scanned PDF with no text layer or an empty "
            f"document. Skipping ingest to avoid wasting LLM tokens on "
            f"empty input."
        )
        self.path = path
        self.content_type = content_type
        self.extracted_chars = extracted_chars
