"""Chunk persistence for provenance click-through.

Persists extracted chunks to the SQLite chunks table so synthesized
pages can reference them via stable chunk_ids. Users run
wikiloom source <chunk_id> to see the exact text the LLM saw.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from wikiloom.cache import (
    SQLiteCache,
    _embed_in_batches,
    ensure_chunk_vec,
    get_embedder_fingerprint,
    set_embedder_fingerprint,
)
from wikiloom.utils import now_iso

if TYPE_CHECKING:
    from wikiloom.ingest.extractors.base import ExtractedContent

# 12 hex chars = 48 bits of entropy. Collision risk is negligible for
# any realistic wiki (would need >10M chunks from one source to hit).
CHUNK_ID_LENGTH = 12

# float32 = 4 bytes per dim; chunk_vec dim is len(blob)//4.
_FLOAT32_BYTES = 4


@dataclass(frozen=True)
class StoredChunk:
    """A chunks-table row hydrated into a dataclass for readable returns."""

    chunk_id: str
    source_hash: str
    chunk_index: int
    chunk_total: int
    content_type: str
    text: str
    token_estimate: int
    created_at: str


def derive_chunk_id(source_hash: str, chunk_index: int) -> str:
    """Stable chunk_id for a given (source_hash, chunk_index)."""
    payload = f"{source_hash}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:CHUNK_ID_LENGTH]


class ChunkStore:
    """Reads/writes the ``chunks`` table via the shared ``SQLiteCache``."""

    def __init__(self, cache_or_path: SQLiteCache | Path) -> None:
        if isinstance(cache_or_path, SQLiteCache):
            self._cache = cache_or_path
            self._owns_cache = False
        else:
            self._cache = SQLiteCache(Path(cache_or_path))
            self._owns_cache = True
        self.db_path = self._cache.db_path

    def close(self) -> None:
        if self._owns_cache:
            self._cache.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def persist_chunks(
        self,
        source_hash: str,
        chunks: Iterable["ExtractedContent"],
        *,
        embedder: Any | None = None,
        embedder_provider: str = "",
        embedder_model: str = "",
    ) -> list[StoredChunk]:
        """Insert chunks for a source, replacing any prior rows."""
        chunk_list = list(chunks)
        created = now_iso()
        stored: list[StoredChunk] = []

        # Embed outside the SQLite lock — embedder calls can be slow.
        embeddings: list[bytes | None] = [None] * len(chunk_list)
        dim: int | None = None
        if embedder is not None and chunk_list:
            texts = [c.text for c in chunk_list]
            embeddings = _embed_in_batches(embedder, texts)
            for blob in embeddings:
                if blob is not None:
                    dim = len(blob) // _FLOAT32_BYTES
                    break

        rows: list[tuple] = []
        for i, chunk in enumerate(chunk_list):
            chunk_index = int(chunk.metadata.get("chunk_index", 0))
            chunk_total = int(
                chunk.metadata.get("chunk_total", len(chunk_list))
            )
            chunk_id = derive_chunk_id(source_hash, chunk_index)
            source_path_str = (
                str(chunk.source_path) if chunk.source_path is not None else None
            )
            parent_heading = chunk.metadata.get("parent_heading")
            record = StoredChunk(
                chunk_id=chunk_id,
                source_hash=source_hash,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                content_type=chunk.content_type,
                text=chunk.text,
                token_estimate=chunk.token_estimate,
                created_at=created,
            )
            rows.append(
                (
                    chunk_id,
                    source_hash,
                    chunk_index,
                    chunk_total,
                    chunk.content_type,
                    chunk.text,
                    chunk.token_estimate,
                    created,
                    source_path_str,
                    parent_heading,
                    None,  # page_id populated post-synthesis by page_writer
                    embeddings[i],
                )
            )
            stored.append(record)

        with self._cache._connect() as conn:
            # Refuse if the embedder identity has changed since the
            # index was first built — mixing vector spaces silently
            # corrupts retrieval.
            if dim is not None:
                existing_fp = get_embedder_fingerprint(conn)
                if existing_fp is not None and existing_fp != (
                    embedder_provider, embedder_model, dim,
                ):
                    raise RuntimeError(
                        f"Embedder fingerprint mismatch: stored {existing_fp}, "
                        f"active {(embedder_provider, embedder_model, dim)}. "
                        f"Re-embedding the index requires wiping chunks first."
                    )

            old_rowids = [
                r[0]
                for r in conn.execute(
                    "SELECT rowid FROM chunks WHERE source_hash = ?",
                    (source_hash,),
                ).fetchall()
            ]
            if old_rowids and _chunk_vec_exists(conn):
                conn.executemany(
                    "DELETE FROM chunk_vec WHERE rowid = ?",
                    [(rid,) for rid in old_rowids],
                )
            conn.execute(
                "DELETE FROM chunks WHERE source_hash = ?", (source_hash,)
            )

            if not rows:
                return stored

            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, source_hash, chunk_index, chunk_total,
                    content_type, text, token_estimate, created_at,
                    source_path, parent_heading, page_id, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            if dim is not None:
                ensure_chunk_vec(conn, dim)
                if get_embedder_fingerprint(conn) is None:
                    set_embedder_fingerprint(
                        conn, embedder_provider, embedder_model, dim
                    )
                placeholders = ",".join("?" * len(rows))
                rowid_by_id = {
                    r[0]: r[1]
                    for r in conn.execute(
                        f"SELECT chunk_id, rowid FROM chunks "
                        f"WHERE chunk_id IN ({placeholders})",
                        [r[0] for r in rows],
                    ).fetchall()
                }
                vec_rows = [
                    (rowid_by_id[row[0]], row[11])
                    for row in rows
                    if row[11] is not None
                ]
                if vec_rows:
                    conn.executemany(
                        "INSERT INTO chunk_vec(rowid, embedding) VALUES (?, ?)",
                        vec_rows,
                    )

        return stored

    def set_page_ids(self, mapping: dict[str, str]) -> int:
        """Stamp ``page_id`` on each chunk in ``mapping``. Returns rows updated."""
        if not mapping:
            return 0
        with self._cache._connect() as conn:
            cur = conn.executemany(
                "UPDATE chunks SET page_id = ? WHERE chunk_id = ?",
                [(page_id, cid) for cid, page_id in mapping.items()],
            )
            return cur.rowcount or 0

    def delete_by_source(self, source_hash: str) -> int:
        """Remove every chunk for a given source. Returns rows affected."""
        with self._cache._connect() as conn:
            old_rowids = [
                r[0]
                for r in conn.execute(
                    "SELECT rowid FROM chunks WHERE source_hash = ?",
                    (source_hash,),
                ).fetchall()
            ]
            if old_rowids and _chunk_vec_exists(conn):
                conn.executemany(
                    "DELETE FROM chunk_vec WHERE rowid = ?",
                    [(rid,) for rid in old_rowids],
                )
            cursor = conn.execute(
                "DELETE FROM chunks WHERE source_hash = ?", (source_hash,)
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        """Fetch a single chunk by its id, or None if not present."""
        with self._cache._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return _row_to_stored_chunk(row) if row is not None else None

    def get_chunks_for_source(self, source_hash: str) -> list[StoredChunk]:
        """All chunks for a source, ordered by chunk_index."""
        with self._cache._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE source_hash = ?
                ORDER BY chunk_index ASC
                """,
                (source_hash,),
            ).fetchall()
        return [_row_to_stored_chunk(row) for row in rows]

    def count(self) -> int:
        """Total chunk rows. Used by tests and ``wikiloom status``."""
        with self._cache._connect() as conn:
            (total,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(total)


def _chunk_vec_exists(conn: Any) -> bool:
    # Created lazily by ensure_chunk_vec on first persist with an embedder.
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_vec'"
    ).fetchone()
    return row is not None


def _row_to_stored_chunk(row: Any) -> StoredChunk:
    return StoredChunk(
        chunk_id=row["chunk_id"],
        source_hash=row["source_hash"],
        chunk_index=int(row["chunk_index"]),
        chunk_total=int(row["chunk_total"]),
        content_type=row["content_type"] or "",
        text=row["text"] or "",
        token_estimate=int(row["token_estimate"] or 0),
        created_at=row["created_at"] or "",
    )
