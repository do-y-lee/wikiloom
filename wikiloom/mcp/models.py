"""Pydantic output models for the WikiLoom MCP tools.

Internal types stay frozen dataclasses (``Citation``, ``ContextResult``,
``PageHit``, ``StoredChunk``). These models live at the MCP boundary so
the agent sees a precise JSON schema with per-field semantics — better
chaining decisions, fewer "guessed the field name" failures.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Summary preview length for router payloads (~30 tokens). Full body
# stays behind get_pages so the cheap-router stays cheap.
_SUMMARY_MAX_CHARS = 120


def short_summary(s: str, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    """Word-boundary truncation with ellipsis. Empty/short inputs pass through."""
    if not s or len(s) <= max_chars:
        return s
    head = s[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:!?")
    return head + "…" if head else s[: max_chars - 1] + "…"


class PageHitOut(BaseModel):
    page_id: str = Field(
        description="Stable page id; pass to get_pages or get_outbound_links."
    )
    type: str = Field(description="Page type, e.g. 'concept' or 'entity'.")
    title: str = Field(description="Human-readable page title.")
    summary: str = Field(
        description="≤30-token router preview — NOT the full body. "
                    "Call get_pages(ids) for the full markdown."
    )
    similarity: float = Field(
        description="Cosine similarity to the goal embedding, in [0,1]."
    )


class CitationOut(BaseModel):
    chunk_id: str = Field(
        description="Stable chunk id; pass to get_chunks for full text."
    )
    page_id: str | None = Field(
        default=None,
        description="The synthesized page this chunk belongs to, if any.",
    )
    source_path: str | None = Field(
        default=None,
        description="Original source file path the chunk was extracted from.",
    )
    parent_heading: str | None = Field(
        default=None,
        description="Nearest markdown heading above the chunk in its source.",
    )
    snippet: str = Field(
        description="~200-char preview; NOT the full chunk text. "
                    "Call get_chunks(ids) for the full text."
    )
    score: float = Field(description="RRF-fused BM25 + vector score.")
    token_estimate: int | None = Field(
        default=None,
        description="Approximate token count; usable for budgeting.",
    )


class ContextResultOut(BaseModel):
    pages: list[PageHitOut] = Field(
        description="Router-picked pages with truncated summaries — "
                    "explains which topics the chunks were drawn from."
    )
    citations: list[CitationOut] = Field(
        description="Top-ranked chunks within the routed pages, "
                    "optionally clamped by a token budget."
    )


class StoredChunkOut(BaseModel):
    chunk_id: str = Field(description="Stable chunk id.")
    text: str = Field(
        description="Full chunk text — the expensive payload. "
                    "Provenance fields (page_id, source_path) come from "
                    "the earlier search_chunks call."
    )
    token_estimate: int | None = Field(
        default=None, description="Approximate token count."
    )
    content_type: str = Field(
        description="Source content type, e.g. 'markdown', 'pdf'."
    )


class PageBodyOut(BaseModel):
    page_id: str
    title: str = Field(description="Human-readable page title.")
    type: str = Field(description="Page type, e.g. 'concept' or 'entity'.")
    status: str = Field(
        description="Lifecycle label, e.g. 'active', 'stale', 'deprecated'."
    )
    summary: str = Field(description="Full summary (not truncated here).")
    body: str = Field(
        description="Full synthesized page markdown — the expensive payload."
    )


class OutboundLinkOut(BaseModel):
    source_page: str = Field(description="Page whose body contains the link.")
    target_page: str = Field(
        description="Page being linked to; pass to get_pages to follow the hop."
    )
