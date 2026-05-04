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

from wikiloom.cache import init_cache
from wikiloom.utils import now_iso

if TYPE_CHECKING:
    from wikiloom.ingest.extractors.base import ExtractedContent

# 12 hex chars = 48 bits of entropy. Collision risk is negligible for
# any realistic wiki (would need >10M chunks from one source to hit).
CHUNK_ID_LENGTH = 12


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
    """Return the stable chunk_id for a given (source_hash, chunk_index).

    Deterministic across runs — the same source file re-ingested with
    ``--force`` produces the exact same chunk_ids, which means page
    frontmatter references remain valid after re-ingest.
    """
    payload = f"{source_hash}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:CHUNK_ID_LENGTH]


class ChunkStore:
    """Reads/writes the ``chunks`` table in the SQLite cache."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        # Cheap idempotent schema ensure — matches SQLiteCache.__init__.
        # If the cache db doesn't exist yet (e.g., a new project without
        # a rebuild-cache run), this creates it.
        init_cache(self.db_path)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def persist_chunks(
        self,
        source_hash: str,
        chunks: Iterable["ExtractedContent"],
    ) -> list[StoredChunk]:
        """Insert every chunk for a source, replacing any prior rows.

        Returns the list of ``StoredChunk`` records in the order they
        were inserted so the synthesis loop can pair each LLM call with
        the corresponding chunk_id without a second lookup.
        """
        import sqlite3

        chunk_list = list(chunks)
        created = now_iso()
        stored: list[StoredChunk] = []

        conn = sqlite3.connect(str(self.db_path))
        try:
            # Wipe any prior chunks for this source so re-ingests don't
            # accumulate stale rows. Same-index chunks with same bytes
            # will reproduce the same chunk_id, so references stay
            # stable even through the delete/insert cycle.
            conn.execute("DELETE FROM chunks WHERE source_hash = ?", (source_hash,))

            rows: list[tuple] = []
            for chunk in chunk_list:
                chunk_index = int(chunk.metadata.get("chunk_index", 0))
                chunk_total = int(
                    chunk.metadata.get("chunk_total", len(chunk_list))
                )
                chunk_id = derive_chunk_id(source_hash, chunk_index)
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
                        record.chunk_id,
                        record.source_hash,
                        record.chunk_index,
                        record.chunk_total,
                        record.content_type,
                        record.text,
                        record.token_estimate,
                        record.created_at,
                    )
                )
                stored.append(record)
            if rows:
                conn.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, source_hash, chunk_index, chunk_total,
                        content_type, text, token_estimate, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.commit()
        finally:
            conn.close()

        return stored

    def delete_by_source(self, source_hash: str) -> int:
        """Remove every chunk for a given source. Returns rows affected."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE source_hash = ?", (source_hash,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        """Fetch a single chunk by its id, or None if not present."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        finally:
            conn.close()
        return _row_to_stored_chunk(row) if row is not None else None

    def get_chunks_for_source(self, source_hash: str) -> list[StoredChunk]:
        """All chunks for a source, ordered by chunk_index."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE source_hash = ?
                ORDER BY chunk_index ASC
                """,
                (source_hash,),
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_stored_chunk(row) for row in rows]

    def count(self) -> int:
        """Total chunk rows. Used by tests and ``wikiloom status``."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        try:
            (total,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        finally:
            conn.close()
        return int(total)


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
