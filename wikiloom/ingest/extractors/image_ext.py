"""Image extractor — produces a placeholder text payload + base64 for LLM vision."""

from __future__ import annotations

import base64
from pathlib import Path

from wikiloom.ingest.extractors.base import BaseExtractor, ExtractedContent
from wikiloom.utils import estimate_tokens

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class ImageExtractor(BaseExtractor):
    """Image extractor.

    Images don't have textual content to extract — they're handled by an
    LLM vision call downstream. This extractor produces a small text
    placeholder describing the image and stores the base64-encoded bytes
    in metadata under "image_base64" so the LLM client can pass them to a
    vision-capable model.
    """

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def extract(self, path: Path) -> ExtractedContent:
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")

        text = f"[Image: {path.name} | {len(raw)} bytes | requires LLM vision]"

        return ExtractedContent(
            text=text,
            metadata={
                "filename": path.name,
                "image_base64": encoded,
                "byte_size": len(raw),
                "mime_type": _mime_for(path.suffix.lower()),
            },
            source_path=path,
            content_type="image",
            extraction_method="base64-passthrough",
            token_estimate=estimate_tokens(text),
        )


def _mime_for(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "application/octet-stream")
