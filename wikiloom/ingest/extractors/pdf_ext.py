"""PDF extractor using pymupdf."""

from __future__ import annotations

from pathlib import Path

from wikiloom.ingest.extractors.base import BaseExtractor, ExtractedContent
from wikiloom.utils import estimate_tokens


class PDFExtractor(BaseExtractor):
    """Extract text from PDF files using pymupdf.

    Populates metadata["pages"] as list[str] (one entry per page) so the
    chunker can fall back to page-boundary chunking for oversized PDFs.
    """

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(self, path: Path) -> ExtractedContent:
        import pymupdf  # imported lazily so the package can be optional at import time

        pages: list[str] = []
        doc = pymupdf.open(str(path))
        try:
            for page in doc:
                pages.append(page.get_text())
        finally:
            doc.close()

        full_text = "\n\n".join(pages)

        return ExtractedContent(
            text=full_text,
            metadata={
                "filename": path.name,
                "pages": pages,
                "page_count": len(pages),
            },
            source_path=path,
            content_type="pdf",
            extraction_method="pymupdf",
            token_estimate=estimate_tokens(full_text),
        )
