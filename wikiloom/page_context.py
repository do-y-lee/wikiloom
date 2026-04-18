"""Per-chunk page context retrieval for synthesis.

Embeds each chunk and queries the SQLite cache for the top-K most
similar existing pages, so the synthesis prompt can include
chunk-relevant candidates instead of a generic manifest snapshot.
Reduces duplicate page creation by letting the LLM see what already
exists on the chunk's topic before deciding UPDATE vs CREATE.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wikiloom.embeddings import cosine_similarity, deserialize_embedding


@dataclass
class PageCandidate:
    """A page surfaced as potentially relevant to a chunk."""

    page_id: str
    type: str
    title: str
    summary: str
    similarity: float


def retrieve_candidates_for_chunk(
    chunk_text: str,
    project_root: Path,
    embedder: Any,
    top_k: int = 10,
    min_similarity: float = 0.60,
) -> list[PageCandidate]:
    """Return the top-K existing pages most similar to ``chunk_text``.

    Embeds the chunk, scores every page in the cache by cosine
    similarity, and returns the top-K above ``min_similarity``.
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

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT page_id, type, title, summary, embedding "
            "FROM pages WHERE status = 'active' AND embedding IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    scored: list[PageCandidate] = []
    for row in rows:
        blob = row["embedding"]
        if blob is None:
            continue
        try:
            page_vec = deserialize_embedding(blob)
        except Exception:
            continue
        score = cosine_similarity(query_vec, page_vec)
        if score < min_similarity:
            continue
        scored.append(
            PageCandidate(
                page_id=row["page_id"],
                type=row["type"],
                title=row["title"],
                summary=row["summary"] or "",
                similarity=score,
            )
        )

    scored.sort(key=lambda c: c.similarity, reverse=True)
    return scored[:top_k]


def render_candidates(candidates: list[PageCandidate]) -> str:
    """Render retrieved candidates as a markdown table for prompt injection.

    Returns a placeholder line when empty so the prompt section always
    has content (makes the LLM's instruction about reading "the list
    below" well-defined even on first ingest of an empty wiki).
    """
    if not candidates:
        return "(no semantically related pages found — this chunk's topic is likely new)"
    lines = [
        "| page_id | type | title | summary |",
        "|---|---|---|---|",
    ]
    for c in candidates:
        summary = c.summary.replace("\n", " ").replace("|", "\\|")
        if len(summary) > 100:
            summary = summary[:97] + "..."
        title = c.title.replace("|", "\\|")
        lines.append(f"| {c.page_id} | {c.type} | {title} | {summary} |")
    return "\n".join(lines)
