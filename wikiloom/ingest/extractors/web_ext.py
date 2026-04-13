"""Web extractor using trafilatura."""

from __future__ import annotations

from pathlib import Path

from wikiloom.ingest.extractors.base import BaseExtractor, ExtractedContent
from wikiloom.utils import estimate_tokens


class WebExtractor(BaseExtractor):
    """Fetch a URL and extract its main content as markdown via trafilatura."""

    def can_handle(self, path: Path) -> bool:
        return str(path).startswith(("http://", "https://"))

    def extract(self, path: Path) -> ExtractedContent:
        import trafilatura

        url = str(path)
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            raise RuntimeError(f"Failed to fetch URL: {url}")

        text = trafilatura.extract(
            downloaded,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        )
        if not text:
            text = ""

        return ExtractedContent(
            text=text,
            metadata={"url": url},
            source_path=Path(url),
            content_type="web",
            extraction_method="trafilatura",
            token_estimate=estimate_tokens(text),
        )
