"""Tests for chunker boundary strategies and parent_heading capture."""

from __future__ import annotations

from pathlib import Path

from wikiloom.ingest.chunker import BudgetPlan, Chunker, _first_heading
from wikiloom.ingest.extractors.base import ExtractedContent


def _content(text: str, content_type: str = "markdown") -> ExtractedContent:
    return ExtractedContent(
        text=text,
        metadata={"filename": "test.md"},
        source_path=Path("test.md"),
        content_type=content_type,
        extraction_method="plain-text",
        token_estimate=len(text) // 4 + 1,
    )


# ----------------------------------------------------------------------
# _first_heading helper
# ----------------------------------------------------------------------


def test_first_heading_picks_first_atx_line() -> None:
    text = "preamble line\n# Auth\nbody"
    assert _first_heading(text) == "Auth"


def test_first_heading_handles_levels_h1_to_h6() -> None:
    for level in range(1, 7):
        prefix = "#" * level
        assert _first_heading(f"{prefix} Title\nbody") == "Title"


def test_first_heading_returns_none_when_absent() -> None:
    assert _first_heading("just paragraphs of text\n\nno headings here") is None


def test_first_heading_strips_trailing_hashes() -> None:
    # ATX closing-hash variant: "# Title #" → "Title"
    assert _first_heading("# Title #\nbody") == "Title"


# ----------------------------------------------------------------------
# Chunker.split — multi-section markdown gets per-section parent_heading
# ----------------------------------------------------------------------


def test_chunker_stamps_parent_heading_on_each_section() -> None:
    text = (
        "# Auth\n"
        "auth body paragraph one.\n\n"
        "auth body paragraph two.\n\n"
        "# Login\n"
        "login body paragraph one.\n\n"
        "# Logout\n"
        "logout body paragraph one.\n"
    )
    # Budget large enough that each section fits in one chunk → triggers
    # the heading-split strategy.
    chunks = Chunker().split(_content(text), BudgetPlan(available_for_source=1000, needs_chunking=True))

    headings = [c.metadata.get("parent_heading") for c in chunks]
    assert headings == ["Auth", "Login", "Logout"]


def test_chunker_stamps_parent_heading_for_paragraph_fallback_when_chunk_includes_heading() -> None:
    # Force the paragraph-fallback path by giving content with no
    # h1/h2 headings (so _split_on_headings returns None) but with
    # an h3 inside one paragraph.
    text = (
        "Intro paragraph with no top-level heading.\n\n"
        "### Subsection\nA paragraph under the subsection.\n\n"
        "Another paragraph still under it.\n"
    )
    chunks = Chunker().split(
        _content(text),
        BudgetPlan(available_for_source=1000, needs_chunking=True),
    )
    # All paragraphs fit in one fallback chunk; that chunk's
    # parent_heading should be the first heading found in it.
    assert len(chunks) == 1
    assert chunks[0].metadata.get("parent_heading") == "Subsection"


def test_chunker_omits_parent_heading_when_no_heading_in_chunk() -> None:
    text = "paragraph one.\n\nparagraph two.\n\nparagraph three.\n"
    chunks = Chunker().split(
        _content(text, content_type="text"),
        BudgetPlan(available_for_source=1000, needs_chunking=True),
    )
    assert len(chunks) == 1
    # No heading anywhere → metadata key absent (downstream callers
    # treat absence as None).
    assert "parent_heading" not in chunks[0].metadata
