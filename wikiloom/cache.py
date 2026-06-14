"""SQLite query cache.

Derived read-side index at _registry/wiki.db. Mirrors manifest,
backlinks, and page content for FTS5 search, stats, and embedding
similarity queries. Regenerable via wikiloom rebuild-cache.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

# Batch size for embedding-model calls. Small enough that no single
# call exceeds fastembed/sentence-transformer backend limits on
# long-body pages; large enough that the per-call overhead amortizes.
# Used by both full_rebuild and _incremental_sync.
_EMBED_BATCH_SIZE = 32


def _embed_in_batches(
    embedder: Any,
    texts: list[str],
    progress: "Callable[[int, int], None] | None" = None,
) -> list[bytes | None]:
    """Compute embeddings in fixed-size batches, one batch at a time.

    Serializes successful batches and returns one entry per input (None
    for batches that raised). Surfaces any batch failure to stderr so
    the user knows embeddings silently degraded — previous silent-
    catch behavior meant a single long page could cause the entire
    wiki to end up with NULL embeddings with no feedback.

    ``progress`` is optional; when provided it's called as
    ``progress(done, total)`` after each batch so the CLI can render
    incremental status on large wikis.
    """
    from wikiloom.embeddings import serialize_embedding

    total = len(texts)
    out: list[bytes | None] = [None] * total
    for start in range(0, total, _EMBED_BATCH_SIZE):
        end = min(start + _EMBED_BATCH_SIZE, total)
        batch = texts[start:end]
        try:
            vectors = embedder.embed_texts(batch)
        except Exception as exc:
            sys.stderr.write(
                f"Warning: embedder.embed_texts failed on batch "
                f"{start}–{end - 1} ({type(exc).__name__}: {exc}). "
                f"Those pages will have NULL embeddings. Run "
                f"`wikiloom rebuild-cache` to retry.\n"
            )
            sys.stderr.flush()
            if progress is not None:
                progress(end, total)
            continue
        for offset, v in enumerate(vectors):
            out[start + offset] = serialize_embedding(v)
        if progress is not None:
            progress(end, total)
    return out

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.registry import Registry

SQLITE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS pages (
    page_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    summary TEXT,
    created TEXT NOT NULL,
    modified TEXT NOT NULL,
    source_count INTEGER DEFAULT 0,
    inbound_links INTEGER DEFAULT 0,
    outbound_links INTEGER DEFAULT 0,
    confidence TEXT DEFAULT 'medium',
    human_edited BOOLEAN DEFAULT 0,
    content_hash TEXT,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT NOT NULL,
    page_id TEXT NOT NULL,
    FOREIGN KEY (page_id) REFERENCES pages(page_id),
    PRIMARY KEY (alias, page_id)
);

CREATE TABLE IF NOT EXISTS backlinks (
    source_page TEXT NOT NULL,
    target_page TEXT NOT NULL,
    context TEXT,
    confidence TEXT DEFAULT 'high',
    linked_at TEXT NOT NULL,
    PRIMARY KEY (source_page, target_page)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT,
    pages_created TEXT,
    pages_updated TEXT,
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    git_hash TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(page_id, title, summary, body, tokenize='porter ascii');

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_total INTEGER NOT NULL,
    content_type TEXT,
    text TEXT NOT NULL,
    token_estimate INTEGER,
    created_at TEXT NOT NULL,
    source_path TEXT,
    parent_heading TEXT,
    page_id TEXT,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);

-- Many-to-many chunk<->page edges. One chunk can feed several pages,
-- which chunks.page_id (a single scalar) can't represent. Materialized
-- projection of each page's frontmatter sources.chunk_ids; rebuilt by
-- full_rebuild / _incremental_sync. Retrieval scopes via this, not the
-- scalar.
CREATE TABLE IF NOT EXISTS chunk_pages (
    chunk_id TEXT NOT NULL,
    page_id  TEXT NOT NULL,
    PRIMARY KEY (chunk_id, page_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_pages_page ON chunk_pages(page_id);

-- External-content FTS5 over chunks.text. content_rowid='rowid' lets
-- BM25 results join straight back to chunks by rowid; the triggers
-- below keep chunks_fts in lockstep with chunks on every write.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid', tokenize='porter ascii'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Generic key/value store for cache-wide flags. First user is the
-- embedder fingerprint (provider, model, dim) that the chunk_vec
-- table is keyed to; retrieval refuses to query if the active
-- embedder doesn't match what was stamped at index time.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


_FTS_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "may", "me", "might", "my", "no", "not", "of", "on",
    "or", "our", "shall", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "would", "you", "your",
})


def _build_fts_match(query: str) -> str:
    """Turn a free-text query into an FTS5 MATCH expression.

    Strips stop words and punctuation so a natural-language question
    like "What is overdraft protection?" becomes ``"overdraft"
    "protection"`` instead of ``"What" "is" "overdraft"
    "protection?"`` (which would fail because no page title contains
    "What"). Each remaining term is quoted so FTS5 operator
    characters survive as literals.
    """
    import re

    terms: list[str] = []
    for raw in query.split():
        cleaned = re.sub(r"[^\w-]", "", raw).lower()
        if cleaned and cleaned not in _FTS_STOP_WORDS:
            escaped = cleaned.replace('"', '""')
            terms.append(f'"{escaped}"')
    return " OR ".join(terms)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Register the sqlite-vec extension on a connection.

    Required before any reference to the ``vec0`` virtual table module
    (chunk_vec). The platform's stock python-sqlite must be built with
    extension loading enabled — system Python on macOS and some Linux
    distros ships with it disabled. We surface a clear ImportError so
    the user knows to switch interpreters rather than chase a cryptic
    ``no such module: vec0`` later.
    """
    try:
        import sqlite_vec
    except ImportError as exc:
        raise ImportError(
            "sqlite-vec is required for chunk-level retrieval. "
            "Install with: pip install sqlite-vec"
        ) from exc
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        raise RuntimeError(
            "Your Python sqlite3 was built without extension loading. "
            "Use a Homebrew/pyenv interpreter (or any build with "
            "SQLITE_ENABLE_LOAD_EXTENSION) to use chunk retrieval."
        ) from exc


def init_cache(db_path: Path) -> None:
    """Create the SQLite cache file and install the schema.

    Idempotent — safe to call on an existing db. Used by
    ``scaffold.init_project`` during ``wikiloom init``.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SQLITE_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def ensure_chunk_vec(conn: sqlite3.Connection, dim: int) -> None:
    """Create ``chunk_vec`` virtual table if missing, sized to ``dim``.

    sqlite-vec's ``vec0`` requires a literal dimension at CREATE time,
    so we can't ship the table in ``SQLITE_SCHEMA`` — we don't know
    the embedder's dimension until the first persist. Stamp ``dim``
    into ``meta`` alongside provider/model so retrieval can verify.
    """
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec "
        f"USING vec0(embedding float[{int(dim)}])"
    )


_FP_KEYS = ("embedder_provider", "embedder_model", "embedder_dim")


def set_embedder_fingerprint(
    conn: sqlite3.Connection, provider: str, model: str, dim: int
) -> None:
    """Stamp the embedder identity used to populate ``chunk_vec`` into ``meta``."""
    conn.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [
            ("embedder_provider", provider),
            ("embedder_model", model),
            ("embedder_dim", str(int(dim))),
        ],
    )


def chunk_vec_exists(conn: sqlite3.Connection) -> bool:
    """True if ``chunk_vec`` has been created (lazy on first persist)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_vec'"
    ).fetchone()
    return row is not None


def get_embedder_fingerprint(
    conn: sqlite3.Connection,
) -> tuple[str, str, int] | None:
    """Return ``(provider, model, dim)`` from ``meta`` or ``None`` if unset."""
    placeholders = ",".join("?" * len(_FP_KEYS))
    rows = conn.execute(
        f"SELECT key, value FROM meta WHERE key IN ({placeholders})",
        _FP_KEYS,
    ).fetchall()
    found = {r[0]: r[1] for r in rows}
    if set(found) != set(_FP_KEYS):
        return None
    try:
        return found["embedder_provider"], found["embedder_model"], int(found["embedder_dim"])
    except (TypeError, ValueError):
        return None


class SQLiteCache:
    """Read-mostly query cache over the wiki's derived state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        init_cache(self.db_path)
        # Long-lived connection — every read/write reuses it instead of
        # paying the open/close cost per call. ``check_same_thread=False``
        # is safe because ``self._lock`` serializes every access.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Cache is regenerable via ``wikiloom rebuild-cache`` — durability
        # tradeoff of synchronous=NORMAL is acceptable.
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA mmap_size=268435456;
            """
        )
        # sqlite-vec must be registered on every new connection — the
        # extension can't be persisted in the .db file. Loading at init
        # means the long-lived ``self._conn`` can reference ``chunk_vec``
        # in any later read/write without re-registration cost.
        _load_sqlite_vec(self._conn)
        self._lock = threading.RLock()
        # Lazily-loaded embedding matrix for ``semantic_search``. Marked
        # dirty on every page-write path so the next query reloads.
        self._emb_matrix: np.ndarray | None = None
        self._emb_ids: list[str] = []
        self._emb_statuses: list[str] = []
        self._emb_norms: np.ndarray | None = None
        self._emb_dirty = True

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __del__(self) -> None:
        # Best-effort cleanup if the caller didn't call close().
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Connection plumbing
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _invalidate_embeddings(self) -> None:
        """Mark the cached embedding matrix stale. Called from write paths."""
        self._emb_dirty = True

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def full_rebuild(
        self,
        project_root: Path,
        embedder: Any | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Wipe and repopulate every table from on-disk state.

        Reads the manifest for page rows + aliases, and the backlinks
        file for the edge table. If ``embedder`` is provided, computes
        and stores embeddings for each page (title + summary + body).
        Returns the number of pages loaded. Callers should hold the
        project FileLock.

        ``progress`` is optional; when provided it is called as
        ``progress(done, total)`` after each batch of embeddings so
        the CLI can render incremental progress on large wikis. The
        batch size is chosen internally — callers don't need to know.
        """
        project_root = Path(project_root)
        registry_dir = project_root / "_registry"
        wiki_dir = project_root / "wiki"

        registry = Registry(registry_dir, wiki_dir=wiki_dir)
        backlinks = BacklinkRegistry(registry_dir, wiki_dir=wiki_dir)

        with self._connect() as conn:
            # Wipe and repopulate. FTS5 doesn't support TRUNCATE but
            # DELETE works on both regular and virtual tables. The FTS
            # table is dropped and recreated so schema changes (e.g.
            # adding the Porter stemmer tokenizer) take effect on
            # rebuild without requiring a manual db delete.
            conn.execute("DROP TABLE IF EXISTS pages")
            conn.execute("DROP TABLE IF EXISTS aliases")
            conn.execute("DELETE FROM backlinks")
            conn.execute("DELETE FROM chunk_pages")
            conn.execute("DROP TABLE IF EXISTS pages_fts")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pages (
                    page_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    summary TEXT,
                    created TEXT NOT NULL,
                    modified TEXT NOT NULL,
                    source_count INTEGER DEFAULT 0,
                    inbound_links INTEGER DEFAULT 0,
                    outbound_links INTEGER DEFAULT 0,
                    confidence TEXT DEFAULT 'medium',
                    human_edited BOOLEAN DEFAULT 0,
                    content_hash TEXT,
                    embedding BLOB
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    alias TEXT NOT NULL,
                    page_id TEXT NOT NULL,
                    FOREIGN KEY (page_id) REFERENCES pages(page_id),
                    PRIMARY KEY (alias, page_id)
                );
                """
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts "
                "USING fts5(page_id, title, summary, body, tokenize='porter ascii')"
            )

            page_count = 0
            page_entries: list[tuple] = []
            chunk_page_rows: list[tuple[str, str]] = []
            for page_id, entry in registry.pages.items():
                # Read the page body from disk so FTS can index the
                # full content, not just the title + summary. This is
                # what makes "overdrawn" match the overdraft-protection
                # page — the word appears in the body even when the
                # title and summary use "overdraft".
                body_text = ""
                page_path = wiki_dir / f"{page_id}.md"
                if page_path.exists():
                    from wikiloom.frontmatter import read_page
                    fm, body_text = read_page(page_path)
                    # Frontmatter sources.chunk_ids is the authoritative
                    # chunk<->page mapping; project it into chunk_pages.
                    if fm is not None:
                        chunk_page_rows.extend(
                            (cid, page_id) for cid in fm.all_chunk_ids()
                        )

                # Collect text for batch embedding after the loop
                embed_text = f"{entry.title}\n{entry.summary or ''}\n{body_text}"
                page_entries.append((page_id, entry, body_text, embed_text))
                page_count += 1

            # Batch-compute embeddings if an embedder was provided.
            # Delegates to _embed_in_batches so full_rebuild and the
            # incremental sync path share batching and error handling.
            embedding_blobs: list[bytes | None] = [None] * len(page_entries)
            if embedder is not None and page_entries:
                texts = [e[3] for e in page_entries]
                embedding_blobs = _embed_in_batches(
                    embedder, texts, progress=progress
                )

            page_rows = [
                (
                    page_id,
                    entry.title,
                    entry.type,
                    entry.status,
                    entry.summary,
                    entry.created,
                    entry.modified,
                    entry.source_count,
                    entry.inbound_link_count,
                    entry.outbound_link_count,
                    entry.confidence,
                    1 if entry.human_edited else 0,
                    None,
                    embedding_blobs[i],
                )
                for i, (page_id, entry, _, _) in enumerate(page_entries)
            ]
            fts_rows = [
                (page_id, entry.title, entry.summary or "", body_text)
                for page_id, entry, body_text, _ in page_entries
            ]
            alias_rows = [
                (alias.lower(), page_id)
                for page_id, entry, _, _ in page_entries
                for alias in entry.aliases
            ]
            backlink_rows = [
                (
                    edge.source,
                    edge.target,
                    edge.context,
                    edge.confidence,
                    edge.linked_at,
                )
                for edge in backlinks.edges
            ]

            conn.executemany(
                """
                INSERT INTO pages (
                    page_id, title, type, status, summary,
                    created, modified, source_count,
                    inbound_links, outbound_links,
                    confidence, human_edited, content_hash,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                page_rows,
            )
            conn.executemany(
                "INSERT INTO pages_fts (page_id, title, summary, body) VALUES (?, ?, ?, ?)",
                fts_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO aliases (alias, page_id) VALUES (?, ?)",
                alias_rows,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO backlinks (
                    source_page, target_page, context, confidence, linked_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                backlink_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO chunk_pages (chunk_id, page_id) "
                "VALUES (?, ?)",
                chunk_page_rows,
            )

        self._invalidate_embeddings()
        return page_count

    def sync_from_files(
        self,
        project_root: Path,
        changed_files: list[Path] | None = None,
        embedder: Any | None = None,
    ) -> None:
        """Refresh the cache after a write-side commit.

        When ``changed_files`` is ``None``, falls back to a full
        rebuild — the safe default for callers that don't know the
        scope of their changes (e.g., ``wikiloom rebuild-cache``,
        post-merge relink). When ``changed_files`` is a list of page
        paths, only those rows are re-embedded and upserted; every
        other row's embedding stays intact. An empty list is a
        no-op.

        ``embedder`` is forwarded so auto-syncs preserve embeddings
        instead of wiping them to NULL. Non-page paths in
        ``changed_files`` (e.g., ``_registry/manifest.json``) are
        silently ignored — the backlinks table is always refreshed
        from ``backlinks.json`` regardless, since that's cheap and
        keeps graph queries correct after edits that rewrite edges.
        """
        if changed_files is None:
            self.full_rebuild(project_root, embedder=embedder)
            return
        if not changed_files:
            return
        self._incremental_sync(
            project_root, changed_files, embedder=embedder
        )

    def _incremental_sync(
        self,
        project_root: Path,
        changed_files: list[Path],
        embedder: Any | None = None,
    ) -> None:
        """Upsert only the rows for the given page files.

        See ``sync_from_files`` for semantics. Keeps the page table
        intact for every page not in ``changed_files`` so the 99%
        of unchanged pages don't need re-embedding on a single-page
        edit — the fix for the multi-second ``related``, ``merge``,
        ``save`` tax on large wikis.
        """
        from wikiloom.frontmatter import read_page

        project_root = Path(project_root)
        registry_dir = project_root / "_registry"
        wiki_dir = project_root / "wiki"

        registry = Registry(registry_dir, wiki_dir=wiki_dir)
        backlinks = BacklinkRegistry(registry_dir, wiki_dir=wiki_dir)

        # Translate file paths → page_ids. Ignore non-page paths so
        # callers can safely pass the whole staged list (manifest,
        # backlinks.json, etc.) without special-casing.
        changed_page_ids: list[str] = []
        for raw_path in changed_files:
            path = Path(raw_path)
            try:
                rel = path.relative_to(wiki_dir)
            except ValueError:
                continue
            if rel.suffix != ".md":
                continue
            if rel.name in {"log.md", "index.md"}:
                continue
            changed_page_ids.append(str(rel.with_suffix("")))
        # Dedupe while preserving first-seen order. A multi-pair merge
        # batch (``wikiloom duplicates --review``) hands us a list
        # where the same page can appear as winner of one pair and
        # loser of another — or as the surviving winner across
        # multiple pairs. Without this dedup, the page lands in
        # ``page_rows`` twice and trips the executemany INSERT's
        # UNIQUE constraint on ``pages.page_id``.
        changed_page_ids = list(dict.fromkeys(changed_page_ids))

        # Split into rows we refresh vs rows we drop. Only delete when
        # the registry entry is gone (truly retired). When the entry
        # still exists but the file moved to ``archive/`` (a deprecate
        # or merge just happened), keep the row with empty body —
        # mirrors ``full_rebuild`` so both sync paths produce the same
        # cache state and ``wikiloom status`` keeps an accurate
        # deprecated count without a manual rebuild.
        to_upsert: list[tuple[str, Any, str, str]] = []
        to_delete: list[str] = []
        chunk_page_rows: list[tuple[str, str]] = []
        for page_id in changed_page_ids:
            entry = registry.pages.get(page_id)
            if entry is None:
                to_delete.append(page_id)
                continue
            page_path = wiki_dir / f"{page_id}.md"
            if page_path.exists():
                fm, body_text = read_page(page_path)
                # Re-project this page's authoritative chunk<->page edges.
                if fm is not None:
                    chunk_page_rows.extend(
                        (cid, page_id) for cid in fm.all_chunk_ids()
                    )
            else:
                body_text = ""
            embed_text = f"{entry.title}\n{entry.summary or ''}\n{body_text}"
            to_upsert.append((page_id, entry, body_text, embed_text))

        # Batch-embed only the changed subset — this is where the
        # incremental path earns back its cost over full_rebuild.
        # Batched in _EMBED_BATCH_SIZE chunks so a single over-length
        # page body doesn't poison the whole call.
        embedding_blobs: list[bytes | None] = [None] * len(to_upsert)
        if embedder is not None and to_upsert:
            texts = [t[3] for t in to_upsert]
            embedding_blobs = _embed_in_batches(embedder, texts)

        delete_ids = [(page_id,) for page_id in to_delete] + [
            (page_id,) for page_id, _, _, _ in to_upsert
        ]
        page_rows = [
            (
                page_id,
                entry.title,
                entry.type,
                entry.status,
                entry.summary,
                entry.created,
                entry.modified,
                entry.source_count,
                entry.inbound_link_count,
                entry.outbound_link_count,
                entry.confidence,
                1 if entry.human_edited else 0,
                None,
                embedding_blobs[i],
            )
            for i, (page_id, entry, _, _) in enumerate(to_upsert)
        ]
        fts_rows = [
            (page_id, entry.title, entry.summary or "", body_text)
            for page_id, entry, body_text, _ in to_upsert
        ]
        alias_rows = [
            (alias.lower(), page_id)
            for page_id, entry, _, _ in to_upsert
            for alias in entry.aliases
        ]
        backlink_rows = [
            (
                edge.source,
                edge.target,
                edge.context,
                edge.confidence,
                edge.linked_at,
            )
            for edge in backlinks.edges
        ]

        with self._connect() as conn:
            conn.executemany("DELETE FROM pages WHERE page_id = ?", delete_ids)
            conn.executemany("DELETE FROM aliases WHERE page_id = ?", delete_ids)
            conn.executemany(
                "DELETE FROM pages_fts WHERE page_id = ?", delete_ids
            )
            conn.executemany(
                "DELETE FROM chunk_pages WHERE page_id = ?", delete_ids
            )

            conn.executemany(
                """
                INSERT INTO pages (
                    page_id, title, type, status, summary,
                    created, modified, source_count,
                    inbound_links, outbound_links,
                    confidence, human_edited, content_hash,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                page_rows,
            )
            conn.executemany(
                "INSERT INTO pages_fts (page_id, title, summary, body) VALUES (?, ?, ?, ?)",
                fts_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO aliases (alias, page_id) VALUES (?, ?)",
                alias_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO chunk_pages (chunk_id, page_id) "
                "VALUES (?, ?)",
                chunk_page_rows,
            )

            # Refresh backlinks. Cheap — just a JSON read + row re-insert.
            # Keeps graph-walking queries (backlinks, related, orphans)
            # current after merge/rename/link edits.
            conn.execute("DELETE FROM backlinks")
            conn.executemany(
                """
                INSERT OR REPLACE INTO backlinks (
                    source_page, target_page, context, confidence, linked_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                backlink_rows,
            )

        self._invalidate_embeddings()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """FTS5 search over page title + summary.

        Returns a list of page row dicts ordered by FTS rank. Empty or
        whitespace queries return no results (FTS5 raises on empty
        MATCH strings, which isn't a useful error here).

        User queries are tokenized on whitespace and each term is
        wrapped in double quotes so FTS5 special characters (``/``,
        ``-``, ``:``, etc.) are treated as literals instead of
        operators. Multi-term queries become implicit AND.
        """
        if not query or not query.strip():
            return []
        match_expr = _build_fts_match(query)
        if not match_expr:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*
                FROM pages p
                JOIN pages_fts f ON p.page_id = f.page_id
                WHERE pages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def semantic_search(
        self,
        query_vector: list[float],
        limit: int = 5,
        *,
        exclude_statuses: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search against page embeddings.

        Returns the top ``limit`` pages by similarity, excluding pages
        with no embedding. Falls back gracefully: if no embeddings
        exist, returns an empty list.

        ``exclude_statuses`` masks rows with the given statuses (e.g.,
        ``("deprecated",)``); fewer than ``limit`` rows may be returned
        if the filter is restrictive.

        Backed by a lazily-loaded numpy matrix cached on the instance,
        invalidated by every page-write path.
        """
        self._ensure_embedding_matrix()
        if self._emb_matrix is None or not self._emb_ids:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        # cosine = (M · q) / (||M_i|| * ||q||)
        assert self._emb_norms is not None  # set whenever _emb_matrix is
        scores = (self._emb_matrix @ q) / (self._emb_norms * q_norm + 1e-12)

        if exclude_statuses:
            excluded = set(exclude_statuses)
            mask = np.array(
                [s not in excluded for s in self._emb_statuses], dtype=bool
            )
            scores = np.where(mask, scores, -np.inf)

        k = min(limit, len(scores))
        if k == 0:
            return []
        # argpartition is O(n); a full sort would be O(n log n).
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        top_idx = top_idx[np.isfinite(scores[top_idx])]
        if len(top_idx) == 0:
            return []
        top_ids = [self._emb_ids[i] for i in top_idx]

        placeholders = ",".join("?" * len(top_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pages WHERE page_id IN ({placeholders})",
                top_ids,
            ).fetchall()
        by_id = {r["page_id"]: dict(r) for r in rows}

        out: list[dict[str, Any]] = []
        for idx in top_idx:
            pid = self._emb_ids[idx]
            row_dict = by_id.get(pid)
            if row_dict is None:
                continue
            row_dict.pop("embedding", None)
            row_dict["similarity"] = float(scores[idx])
            out.append(row_dict)
        return out

    def _ensure_embedding_matrix(self) -> None:
        """Lazily build the cached embedding matrix used by ``semantic_search``."""
        if not self._emb_dirty and self._emb_matrix is not None:
            return
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT page_id, status, embedding FROM pages "
                "WHERE embedding IS NOT NULL"
            ).fetchall()
        if not rows:
            self._emb_matrix = None
            self._emb_ids = []
            self._emb_statuses = []
            self._emb_norms = None
            self._emb_dirty = False
            return
        self._emb_ids = [r["page_id"] for r in rows]
        self._emb_statuses = [(r["status"] or "active") for r in rows]
        # ``np.frombuffer`` is zero-copy — keeps the on-disk BLOB
        # format unchanged while exposing it as a numpy view.
        self._emb_matrix = np.stack([
            np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
        ])
        self._emb_norms = np.linalg.norm(self._emb_matrix, axis=1)
        self._emb_dirty = False

    def get_stale(self, window_days: int = 90) -> list[dict[str, Any]]:
        """Active pages whose ``modified`` is older than ``window_days``.

        The manifest carries per-page ``staleness_window_days`` but the
        cache schema doesn't mirror it, so v1 uses a single global
        window. Callers that need per-page windows should consult the
        ``Registry`` directly (see ``registry.get_stale_pages``).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM pages
                WHERE status = 'active'
                  AND julianday('now') - julianday(modified) > ?
                ORDER BY modified ASC
                """,
                (window_days,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> dict[str, Any]:
        """High-level counts for status / dashboard commands."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            by_type = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT type, COUNT(*) FROM pages GROUP BY type"
                ).fetchall()
            }
            by_status = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT status, COUNT(*) FROM pages GROUP BY status"
                ).fetchall()
            }
            human_edited = conn.execute(
                "SELECT COUNT(*) FROM pages WHERE human_edited = 1"
            ).fetchone()[0]
            backlink_count = conn.execute(
                "SELECT COUNT(*) FROM backlinks"
            ).fetchone()[0]
            alias_count = conn.execute(
                "SELECT COUNT(*) FROM aliases"
            ).fetchone()[0]
        return {
            "total_pages": total,
            "by_type": by_type,
            "by_status": by_status,
            "human_edited": human_edited,
            "backlinks": backlink_count,
            "aliases": alias_count,
        }

    def get_page(self, page_id: str) -> dict[str, Any] | None:
        """Fetch a single page row by id, or None if missing."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pages WHERE page_id = ?", (page_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_pages(self, page_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Bulk-fetch page rows for the given ids. Missing ids are omitted.

        One ``IN`` query replaces N ``get_page`` calls when callers
        already have a list of ids — used by the query path's linked-
        page ranker.
        """
        if not page_ids:
            return {}
        placeholders = ",".join("?" * len(page_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pages WHERE page_id IN ({placeholders})",
                page_ids,
            ).fetchall()
        return {r["page_id"]: dict(r) for r in rows}

    def get_outbound_edges(
        self, source_page_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Return backlink edges originating from any of ``source_page_ids``.

        Mirrors the data ``BacklinkRegistry.edges`` would expose for the
        same sources, but reads it from the cache's ``backlinks`` table
        so the JSON file doesn't have to be parsed on every query. The
        sync path keeps both stores aligned.
        """
        if not source_page_ids:
            return []
        placeholders = ",".join("?" * len(source_page_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT source_page, target_page "
                f"FROM backlinks WHERE source_page IN ({placeholders})",
                source_page_ids,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_inbound_edges(
        self, target_page_ids: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return backlink edges pointing to any of ``target_page_ids``.

        Mirrors the data ``BacklinkRegistry.edges`` would expose for the
        same targets, but reads it from the cache's ``backlinks`` table
        so the JSON file doesn't have to be parsed on every query. The
        sync path keeps both stores aligned. ``limit`` caps the total
        number of rows returned across all targets.
        """
        if not target_page_ids:
            return []
        placeholders = ",".join("?" * len(target_page_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT source_page, target_page "
                f"FROM backlinks WHERE target_page IN ({placeholders}) "
                f"LIMIT ?",
                (*target_page_ids, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def load_page_embeddings(self) -> dict[str, list[float]]:
        """Return ``{page_id: vector}`` for every page with an embedding.

        Used by the hybrid linker to pre-load all page embeddings once
        per link session. Pages with NULL embeddings are omitted —
        they simply can't be reranked against, so callers treat them
        as out-of-scope for the cosine pass.
        """
        from wikiloom.embeddings import deserialize_embedding

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT page_id, embedding FROM pages "
                "WHERE embedding IS NOT NULL"
            ).fetchall()
        out: dict[str, list[float]] = {}
        for row in rows:
            blob = row["embedding"]
            if blob:
                out[row["page_id"]] = deserialize_embedding(blob)
        return out
