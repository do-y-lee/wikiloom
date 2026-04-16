"""SQLite query cache.

The cache is a derived read-side index over the wiki: it mirrors
``manifest.json`` (pages, aliases), ``backlinks.json`` (graph edges),
and ``wiki/log.md`` (event history) into a single SQLite database at
``_registry/wiki.db``. Query paths (``search``, ``get_stale``,
``get_stats``) hit the db instead of reparsing JSON and walking the
filesystem.

Design notes
------------
- ``wiki.db`` is git-ignored. It is *derived* state — if it's missing,
  corrupt, or out of sync, ``wikiloom rebuild-cache`` regenerates it
  from on-disk files, which remain the source of truth.
- ``full_rebuild`` wipes and repopulates every table. It's cheap at
  today's scale (dozens to low thousands of pages) and side-steps
  every incremental-upsert edge case.
- ``sync_from_files`` is the per-commit hook writers call after a
  successful commit. v1 implementation is "full_rebuild under the
  hood" — correct, dumb, safe. Incremental upsert by changed file is
  a later optimization and can slot in without changing the signature.
- Schema lives here (not scaffold) so there is one source of truth for
  the table definitions. ``scaffold.init_project`` calls ``init_cache``
  to create the empty db at init time.
- The FTS5 virtual table (``pages_fts``) is populated alongside
  ``pages``. Title + summary cover the common search paths; body
  indexing is deliberately out of scope for v1 — ``wikiloom search``
  can fall back to grep for body matches.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
    content_hash TEXT
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
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_hash);
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


class SQLiteCache:
    """Read-mostly query cache over the wiki's derived state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        init_cache(self.db_path)

    # ------------------------------------------------------------------
    # Connection plumbing
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def full_rebuild(self, project_root: Path) -> int:
        """Wipe and repopulate every table from on-disk state.

        Reads the manifest for page rows + aliases, and the backlinks
        file for the edge table. Returns the number of pages loaded.
        Callers should hold the project FileLock.
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
            conn.execute("DELETE FROM pages")
            conn.execute("DELETE FROM aliases")
            conn.execute("DELETE FROM backlinks")
            conn.execute("DROP TABLE IF EXISTS pages_fts")
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts "
                "USING fts5(page_id, title, summary, body, tokenize='porter ascii')"
            )

            page_count = 0
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
                    _, body_text = read_page(page_path)

                conn.execute(
                    """
                    INSERT INTO pages (
                        page_id, title, type, status, summary,
                        created, modified, source_count,
                        inbound_links, outbound_links,
                        confidence, human_edited, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
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
                    ),
                )
                conn.execute(
                    "INSERT INTO pages_fts (page_id, title, summary, body) VALUES (?, ?, ?, ?)",
                    (page_id, entry.title, entry.summary or "", body_text),
                )
                for alias in entry.aliases:
                    conn.execute(
                        "INSERT OR IGNORE INTO aliases (alias, page_id) VALUES (?, ?)",
                        (alias.lower(), page_id),
                    )
                page_count += 1

            for edge in backlinks.edges:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO backlinks (
                        source_page, target_page, context, confidence, linked_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        edge.source,
                        edge.target,
                        edge.context,
                        edge.confidence,
                        edge.linked_at,
                    ),
                )

        return page_count

    def sync_from_files(
        self, project_root: Path, changed_files: list[Path] | None = None
    ) -> None:
        """Refresh the cache after a write-side commit.

        v1 delegates to ``full_rebuild`` — correct and cheap at current
        scale. ``changed_files`` is accepted so callers don't need to
        change when an incremental path lands later.
        """
        del changed_files  # unused in v1
        self.full_rebuild(project_root)

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
