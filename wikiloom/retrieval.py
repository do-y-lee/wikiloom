"""Hybrid BM25 + vector retrieval over the chunks table."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from wikiloom.cache import (
    SQLiteCache,
    _build_fts_match,
    chunk_vec_exists,
    get_embedder_fingerprint,
)
from wikiloom.embeddings import serialize_embedding

# Standard RRF constant; controls how aggressively top ranks dominate.
_RRF_K = 60

# Per-lane candidate multiplier; fusion needs more than k inputs.
_LANE_MULTIPLIER = 3

# Snippet preview length: ≈40-50 tokens. Cheap-router payload size.
_SNIPPET_MAX_CHARS = 200


@dataclass(frozen=True)
class Citation:
    """A single chunk hit with provenance + fused retrieval score."""

    chunk_id: str
    page_id: str | None
    source_path: str | None
    parent_heading: str | None
    snippet: str
    score: float


def _make_snippet(text: str) -> str:
    """First N chars of chunk text, single-line, with ellipsis on truncation."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= _SNIPPET_MAX_CHARS:
        return flat
    return flat[: _SNIPPET_MAX_CHARS - 1].rstrip() + "…"


def search_chunks(
    cache: SQLiteCache,
    embedder: Any | None,
    query: str,
    *,
    k: int = 10,
    min_score: float | None = None,
    embedder_provider: str = "",
    embedder_model: str = "",
    page_ids: list[str] | None = None,
) -> list[Citation]:
    """Return top-k chunks for ``query`` via BM25 + vector + RRF fusion.

    ``embedder`` may be None to run BM25 only. When given, its identity
    is checked against the stored fingerprint and a mismatch raises.

    ``page_ids`` scopes both lanes to chunks belonging to the given
    pages (used by the 1.2 hybrid lane). ``None`` = unscoped (1.1's
    default); ``[]`` = scoped to nothing → empty result.
    """
    if not query.strip() or k <= 0:
        return []
    if page_ids is not None and not page_ids:
        return []

    candidates = max(k * _LANE_MULTIPLIER, k)

    with cache._connect() as conn:
        bm25_ranks = _bm25_lane(conn, query, candidates, page_ids=page_ids)
        vector_ranks = _vector_lane(
            conn, embedder, query, candidates,
            embedder_provider, embedder_model,
            page_ids=page_ids,
        )

        fused = _rrf_fuse(bm25_ranks, vector_ranks)
        if not fused:
            return []

        ranked = sorted(fused.items(), key=lambda x: -x[1])
        if min_score is not None:
            ranked = [(rid, s) for rid, s in ranked if s >= min_score]
        ranked = ranked[:k]
        if not ranked:
            return []

        return _hydrate(conn, ranked)


# ----------------------------------------------------------------------
# Lanes
# ----------------------------------------------------------------------


def _bm25_lane(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    page_ids: list[str] | None = None,
) -> dict[int, int]:
    """Return rowid -> 1-based rank for the BM25 lane.

    When ``page_ids`` is given, restrict matches to chunks whose
    ``page_id`` is in the set via an FTS5 JOIN — FTS5 ``MATCH``
    composes with ``WHERE``, so this stays a single SQL query.
    """
    match_expr = _build_fts_match(query)
    if not match_expr:
        return {}
    if page_ids is not None and not page_ids:
        return {}
    try:
        if page_ids is None:
            rows = conn.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (match_expr, limit),
            ).fetchall()
        else:
            placeholders = ",".join("?" * len(page_ids))
            rows = conn.execute(
                f"SELECT chunks_fts.rowid FROM chunks_fts "
                f"JOIN chunks ON chunks.rowid = chunks_fts.rowid "
                f"WHERE chunks_fts MATCH ? "
                f"AND chunks.page_id IN ({placeholders}) "
                f"ORDER BY rank LIMIT ?",
                (match_expr, *page_ids, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {rid: i + 1 for i, (rid,) in enumerate(rows)}


def _vector_lane(
    conn: sqlite3.Connection,
    embedder: Any | None,
    query: str,
    limit: int,
    expected_provider: str,
    expected_model: str,
    *,
    page_ids: list[str] | None = None,
) -> dict[int, int]:
    """Return rowid -> 1-based rank for the vector lane.

    Empty when the embedder is missing, the index has no fingerprint,
    or (unscoped only) chunk_vec hasn't been created yet. Raises on a
    real fingerprint mismatch — better to fail than to return
    cross-space garbage.

    When ``page_ids`` is given, falls back to in-memory cosine over
    ``chunks.embedding`` for that subset — sqlite-vec's ``MATCH`` is
    terminal and can't compose with ``WHERE``. The subset is small
    (top-N pages × chunks-per-page), so a numpy matmul is microseconds.
    """
    if embedder is None:
        return {}
    if page_ids is not None and not page_ids:
        return {}
    fp = get_embedder_fingerprint(conn)
    if fp is None:
        return {}
    # Unscoped path needs the chunk_vec index; the scoped path reads
    # chunks.embedding directly so it doesn't.
    if page_ids is None and not chunk_vec_exists(conn):
        return {}

    query_vecs = embedder.embed_texts([query])
    if not query_vecs:
        return {}
    query_vec = query_vecs[0]
    current_dim = len(query_vec)

    stored_provider, stored_model, stored_dim = fp
    if current_dim != stored_dim:
        raise RuntimeError(
            f"Embedder dim mismatch: stored {stored_dim}, "
            f"active {current_dim}. Re-embed the index before querying."
        )
    if (
        expected_provider
        and expected_model
        and (expected_provider, expected_model) != (stored_provider, stored_model)
    ):
        raise RuntimeError(
            f"Embedder fingerprint mismatch: stored "
            f"{(stored_provider, stored_model)}, active "
            f"{(expected_provider, expected_model)}. "
            f"Re-embed the index before querying."
        )

    if page_ids is not None:
        return _vector_lane_scoped(conn, query_vec, limit, page_ids)

    blob = serialize_embedding(query_vec)
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunk_vec WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (blob, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {rid: i + 1 for i, (rid,) in enumerate(rows)}


def _vector_lane_scoped(
    conn: sqlite3.Connection,
    query_vec: list[float],
    limit: int,
    page_ids: list[str],
) -> dict[int, int]:
    """In-memory cosine over chunks.embedding for the page-scoped subset."""
    placeholders = ",".join("?" * len(page_ids))
    rows = conn.execute(
        f"SELECT rowid, embedding FROM chunks "
        f"WHERE page_id IN ({placeholders}) AND embedding IS NOT NULL",
        page_ids,
    ).fetchall()
    if not rows:
        return {}

    rowids = [r[0] for r in rows]
    matrix = np.stack(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    )
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return {}
    norms = np.linalg.norm(matrix, axis=1)
    # cosine = (M · q) / (||M_i|| * ||q||); +1e-12 guards zero-norm rows.
    scores = (matrix @ q) / (norms * q_norm + 1e-12)

    k = min(limit, len(scores))
    if k == 0:
        return {}
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return {rowids[i]: rank + 1 for rank, i in enumerate(top_idx)}


# ----------------------------------------------------------------------
# Fusion + hydration
# ----------------------------------------------------------------------


def _rrf_fuse(*lanes: dict[int, int]) -> dict[int, float]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank_i)."""
    fused: dict[int, float] = {}
    for lane in lanes:
        for rid, rank in lane.items():
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
    return fused


def _hydrate(
    conn: sqlite3.Connection, ranked: list[tuple[int, float]]
) -> list[Citation]:
    """One SELECT to attach provenance fields to each fused rowid."""
    rowids = [rid for rid, _ in ranked]
    scores = {rid: s for rid, s in ranked}
    placeholders = ",".join("?" * len(rowids))
    rows = conn.execute(
        f"SELECT rowid AS rid, chunk_id, page_id, source_path, "
        f"parent_heading, text FROM chunks WHERE rowid IN ({placeholders})",
        rowids,
    ).fetchall()
    by_rowid = {row["rid"]: row for row in rows}

    out: list[Citation] = []
    for rid in rowids:
        row = by_rowid.get(rid)
        if row is None:
            continue
        out.append(
            Citation(
                chunk_id=row["chunk_id"],
                page_id=row["page_id"],
                source_path=row["source_path"],
                parent_heading=row["parent_heading"],
                snippet=_make_snippet(row["text"] or ""),
                score=scores[rid],
            )
        )
    return out


