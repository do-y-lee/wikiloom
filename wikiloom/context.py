"""Hybrid context lane: page router + scoped chunk rerank.

Default agent path for goal-shaped queries. Embeds a goal once, asks
the page router for the top-N most similar synthesized pages, then
reranks chunks within those pages via ``search_chunks`` scoped to
the selected page_ids. Returns both surfaces — pages for
explainability, chunks for the actual context payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wikiloom.cache import SQLiteCache
from wikiloom.retrieval import Citation, search_chunks


@dataclass(frozen=True)
class PageHit:
    """A page surfaced by the router with its similarity score."""

    page_id: str
    type: str
    title: str
    summary: str
    similarity: float


@dataclass(frozen=True)
class ContextResult:
    """Routed pages and the chunks reranked within them."""

    pages: list[PageHit]
    citations: list[Citation]


def get_context(
    cache: SQLiteCache,
    embedder: Any,
    goal: str,
    *,
    top_pages: int = 5,
    k: int = 20,
    embedder_provider: str = "",
    embedder_model: str = "",
) -> ContextResult:
    """Return top-K chunks from the top-N pages most similar to ``goal``.

    Two-stage retrieval:

    1. Page router via ``cache.semantic_search`` picks ``top_pages``
       pages by cosine similarity, excluding deprecated.
    2. ``search_chunks(..., page_ids=...)`` reranks chunks within
       those pages via BM25 + vector + RRF.

    ``pages`` is included so callers can see which pages the router
    picked and with what confidence — useful for explainability and
    for any agent-facing tool that wraps this function.

    Returns an empty ``ContextResult`` for empty goals, missing
    embedder, no embedded pages, or zero ``k``/``top_pages``. Raises
    on embedder fingerprint mismatch.
    """
    if not goal.strip() or k <= 0 or top_pages <= 0:
        return ContextResult(pages=[], citations=[])
    if embedder is None:
        return ContextResult(pages=[], citations=[])

    query_vecs = embedder.embed_texts([goal])
    if not query_vecs:
        return ContextResult(pages=[], citations=[])
    query_vec = list(query_vecs[0])

    hits = cache.semantic_search(
        query_vec,
        limit=top_pages,
        exclude_statuses=("deprecated",),
    )
    if not hits:
        return ContextResult(pages=[], citations=[])

    pages = [
        PageHit(
            page_id=h["page_id"],
            type=h["type"],
            title=h["title"],
            summary=h.get("summary") or "",
            similarity=h["similarity"],
        )
        for h in hits
    ]
    page_ids = [p.page_id for p in pages]

    citations = search_chunks(
        cache,
        embedder,
        goal,
        k=k,
        embedder_provider=embedder_provider,
        embedder_model=embedder_model,
        page_ids=page_ids,
        query_vec=query_vec,
    )

    return ContextResult(pages=pages, citations=citations)
