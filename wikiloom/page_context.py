"""Per-chunk page context retrieval for synthesis.

Embeds each chunk and queries the SQLite cache for the top-K most
similar existing pages, so the synthesis prompt can include
chunk-relevant candidates instead of a generic manifest snapshot.
Reduces duplicate page creation by letting the LLM see what already
exists on the chunk's topic before deciding UPDATE vs CREATE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wikiloom.cache import SQLiteCache


@dataclass
class PageCandidate:
    """A page surfaced as potentially relevant to a chunk."""

    page_id: str
    type: str
    title: str
    summary: str
    similarity: float
    status: str = "active"  # "active" or "dormant" — annotates the rendered table


def retrieve_candidates_for_chunk(
    chunk_text: str,
    project_root: Path,
    embedder: Any,
    top_k: int = 10,
    min_similarity: float = 0.60,
    *,
    cache: "SQLiteCache | None" = None,
) -> list[PageCandidate]:
    """Return the top-K existing pages most similar to ``chunk_text``.

    Embeds the chunk and delegates ranking to ``SQLiteCache.semantic_search``,
    which reuses a process-cached embedding matrix and runs one matmul per
    call instead of a Python loop over every page.

    Deprecated pages are excluded; dormant pages are kept since they are
    valid update targets that the LLM should see to avoid duplicate
    creation.

    ``cache`` may be passed in so concurrent workers share a single
    matrix (built once on the first call). When omitted, a transient
    cache is created and closed for this call.

    Returns an empty list gracefully when:

    - the cache is missing (fresh project)
    - no pages have embeddings yet (before first rebuild-cache)
    - the embedder fails on the chunk (surfaces as empty list, not crash)
    """
    db_path = project_root / "_registry" / "wiki.db"
    if not db_path.exists():
        return []

    try:
        vectors = embedder.embed_texts([chunk_text])
    except Exception:
        return []
    if not vectors:
        return []
    query_vec = list(vectors[0])

    from wikiloom.cache import SQLiteCache

    owns_cache = cache is None
    if cache is None:
        cache = SQLiteCache(db_path)
    try:
        hits = cache.semantic_search(
            query_vec,
            limit=top_k,
            exclude_statuses=("deprecated",),
        )
    finally:
        if owns_cache:
            cache.close()

    return [
        PageCandidate(
            page_id=h["page_id"],
            type=h["type"],
            title=h["title"],
            summary=h.get("summary") or "",
            similarity=h["similarity"],
            status=h.get("status") or "active",
        )
        for h in hits
        if h["similarity"] >= min_similarity
    ]


def render_candidates(candidates: list[PageCandidate]) -> str:
    """Render retrieved candidates as a markdown table for prompt injection.

    Returns a placeholder line when empty so the prompt section always
    has content (makes the LLM's instruction about reading "the list
    below" well-defined even on first ingest of an empty wiki).
    """
    if not candidates:
        return "(no semantically related pages found — this chunk's topic is likely new)"
    lines = [
        "| page_id | type | title | status | summary |",
        "|---|---|---|---|---|",
    ]
    for c in candidates:
        summary = c.summary.replace("\n", " ").replace("|", "\\|")
        if len(summary) > 100:
            summary = summary[:97] + "..."
        title = c.title.replace("|", "\\|")
        # Mark dormant pages so the LLM knows they are valid update
        # targets that may benefit from a freshening UPDATE.
        status_marker = "[dormant]" if c.status == "dormant" else "active"
        lines.append(
            f"| {c.page_id} | {c.type} | {title} | {status_marker} | {summary} |"
        )
    return "\n".join(lines)
