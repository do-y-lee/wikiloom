"""Tests for Component 12: SQLite query cache."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import git
import pytest

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.cache import SQLiteCache, init_cache
from wikiloom.frontmatter import Frontmatter, write_page
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Real init_project output with a git identity configured."""
    project_dir = init_project(name="testproj", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    repo.index.add(
        [
            "wikiloom.toml",
            ".gitignore",
            "wiki/index.md",
            "_registry/manifest.json",
            "_registry/backlinks.json",
        ]
    )
    repo.index.commit("initial scaffold")
    return project_dir


def _add_page(
    project_root: Path,
    page_id: str,
    title: str,
    *,
    body: str | None = None,
    aliases: list[str] | None = None,
    summary: str | None = None,
    status: str = "active",
    human_edited: bool = False,
    modified: str = "2026-04-14T12:00:00Z",
    type_: str | None = None,
    chunk_ids: list[str] | None = None,
) -> Path:
    """Write a page file + manifest entry the way the ingest pipeline would.

    Derives ``type`` from the page_id's top-level category if not given
    (e.g. ``concepts/foo`` → ``concept``). ``chunk_ids`` populates the
    ``sources`` frontmatter so cache sync can project chunk_pages edges.
    Returns the page file path.
    """
    category = page_id.split("/", 1)[0]
    inferred_type = type_ or category.rstrip("s")
    aliases = aliases or []
    summary_text = summary if summary is not None else f"Summary of {title}."
    body_text = body if body is not None else f"# {title}\n\nBody text.\n"
    sources = [{"chunk_ids": list(chunk_ids)}] if chunk_ids else []

    fm = Frontmatter(
        title=title,
        type=inferred_type,
        status=status,
        created="2026-04-01T00:00:00Z",
        modified=modified,
        summary=summary_text,
        aliases=aliases,
        human_edited=human_edited,
        sources=sources,
    )
    page_path = project_root / "wiki" / f"{page_id}.md"
    write_page(page_path, fm, body_text)

    registry = Registry(project_root / "_registry")
    entry = PageEntry(
        title=title,
        type=inferred_type,
        status=status,
        aliases=aliases,
        created=fm.created,
        modified=fm.modified,
        summary=summary_text,
        human_edited=human_edited,
    )
    registry.register_page(page_id, entry)
    # Preserve the modified we asked for (register_page overwrites it).
    registry.pages[page_id].modified = modified
    registry.pages[page_id].created = fm.created
    registry.save()
    return page_path


# ----------------------------------------------------------------------
# Schema / init
# ----------------------------------------------------------------------


def test_init_project_creates_schema(project: Path) -> None:
    """`wikiloom init` should leave a wiki.db with every expected table."""
    db_path = project / "_registry" / "wiki.db"
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table')"
            )
        }
    finally:
        conn.close()
    assert {"pages", "aliases", "backlinks", "events", "pages_fts"} <= tables


def test_init_cache_is_idempotent(tmp_path: Path) -> None:
    """Calling init_cache twice should not error."""
    db_path = tmp_path / "wiki.db"
    init_cache(db_path)
    init_cache(db_path)  # no CREATE IF NOT EXISTS crash


# ----------------------------------------------------------------------
# Full rebuild
# ----------------------------------------------------------------------


def test_full_rebuild_empty_project(project: Path) -> None:
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    count = cache.full_rebuild(project)
    assert count == 0
    stats = cache.get_stats()
    assert stats["total_pages"] == 0
    assert stats["backlinks"] == 0
    assert stats["aliases"] == 0


def test_full_rebuild_loads_pages_and_aliases(project: Path) -> None:
    _add_page(
        project,
        "concepts/transformer",
        "Transformer",
        aliases=["transformers", "attention model"],
    )
    _add_page(project, "concepts/attention", "Attention")
    _add_page(
        project,
        "entities/openai",
        "OpenAI",
        type_="entity",
        aliases=["oai"],
    )

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    count = cache.full_rebuild(project)
    assert count == 3

    stats = cache.get_stats()
    assert stats["total_pages"] == 3
    assert stats["by_type"] == {"concept": 2, "entity": 1}
    assert stats["by_status"] == {"active": 3}
    assert stats["aliases"] == 3  # transformers, attention model, oai

    row = cache.get_page("concepts/transformer")
    assert row is not None
    assert row["title"] == "Transformer"
    assert row["type"] == "concept"
    assert row["human_edited"] == 0


def test_full_rebuild_wipes_stale_rows(project: Path) -> None:
    """A rebuild after a page is removed from the manifest should drop it."""
    _add_page(project, "concepts/foo", "Foo")
    _add_page(project, "concepts/bar", "Bar")

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    assert cache.get_stats()["total_pages"] == 2

    # Drop one page from the manifest and rebuild.
    registry = Registry(project / "_registry")
    del registry.pages["concepts/bar"]
    registry.save()

    cache.full_rebuild(project)
    assert cache.get_stats()["total_pages"] == 1
    assert cache.get_page("concepts/bar") is None


def test_full_rebuild_is_idempotent(project: Path) -> None:
    _add_page(project, "concepts/foo", "Foo", aliases=["foobar"])
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    cache.full_rebuild(project)
    stats = cache.get_stats()
    assert stats["total_pages"] == 1
    assert stats["aliases"] == 1


def test_full_rebuild_loads_backlinks(project: Path) -> None:
    """After the backlink registry is rebuilt, cache sync pulls the edges."""
    _add_page(project, "concepts/transformer", "Transformer")
    _add_page(
        project,
        "concepts/attention",
        "Attention",
        body=(
            "# Attention\n\n"
            "Used in the [[concepts/transformer|Transformer]] paper.\n"
        ),
    )

    backlinks = BacklinkRegistry(project / "_registry")
    backlinks.rebuild()
    backlinks.save()

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)

    stats = cache.get_stats()
    assert stats["backlinks"] == 1

    # The edge should be queryable directly.
    conn = sqlite3.connect(str(project / "_registry" / "wiki.db"))
    try:
        row = conn.execute(
            "SELECT source_page, target_page, confidence FROM backlinks"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("concepts/attention", "concepts/transformer", "high")


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


def test_search_by_title(project: Path) -> None:
    _add_page(project, "concepts/transformer", "Transformer")
    _add_page(project, "concepts/attention", "Attention")
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)

    results = cache.search("Transformer")
    assert any(r["page_id"] == "concepts/transformer" for r in results)
    assert not any(r["page_id"] == "concepts/attention" for r in results)


def test_search_by_summary(project: Path) -> None:
    _add_page(
        project,
        "concepts/flash-attention",
        "Flash Attention",
        summary="An efficient attention algorithm for GPUs.",
    )
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)

    results = cache.search("efficient algorithm GPUs")
    assert len(results) == 1
    assert results[0]["page_id"] == "concepts/flash-attention"


def test_search_empty_query_returns_empty(project: Path) -> None:
    _add_page(project, "concepts/foo", "Foo")
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    assert cache.search("") == []
    assert cache.search("   ") == []


def test_search_respects_limit(project: Path) -> None:
    for i in range(5):
        _add_page(project, f"concepts/topic-{i}", f"Topic {i}")
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)

    results = cache.search("Topic", limit=3)
    assert len(results) == 3


# ----------------------------------------------------------------------
# get_stale
# ----------------------------------------------------------------------


def test_get_stale_returns_old_pages(project: Path) -> None:
    _add_page(
        project,
        "concepts/old",
        "Old",
        modified="2020-01-01T00:00:00Z",
    )
    _add_page(
        project,
        "concepts/fresh",
        "Fresh",
        modified="2026-04-14T00:00:00Z",
    )
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)

    stale = cache.get_stale(window_days=90)
    stale_ids = {row["page_id"] for row in stale}
    assert "concepts/old" in stale_ids
    assert "concepts/fresh" not in stale_ids


def test_get_stale_ignores_deprecated(project: Path) -> None:
    _add_page(
        project,
        "concepts/dead",
        "Dead",
        modified="2020-01-01T00:00:00Z",
        status="deprecated",
    )
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    assert cache.get_stale() == []


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------


def test_get_stats_counts_human_edited(project: Path) -> None:
    _add_page(project, "concepts/a", "A", human_edited=True)
    _add_page(project, "concepts/b", "B", human_edited=False)
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)

    stats = cache.get_stats()
    assert stats["human_edited"] == 1


# ----------------------------------------------------------------------
# Sync hook (sync_from_files currently delegates to full_rebuild)
# ----------------------------------------------------------------------


def test_sync_from_files_refreshes_after_manifest_change(project: Path) -> None:
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    assert cache.get_stats()["total_pages"] == 0

    _add_page(project, "concepts/new", "New")
    cache.sync_from_files(project, [project / "wiki" / "concepts" / "new.md"])
    assert cache.get_stats()["total_pages"] == 1


class _SpyEmbedder:
    """Records which texts it embedded and returns a deterministic vector."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t))] * 4 for t in texts]


def test_incremental_sync_only_embeds_changed_pages(project: Path) -> None:
    """Incremental sync must not re-embed the whole wiki on a single-page edit."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    # Seed three pages with a full rebuild so they all start with embeddings.
    for i in range(3):
        _add_page(project, f"concepts/page-{i}", f"Page {i}")
    seed_embedder = _SpyEmbedder()
    cache.full_rebuild(project, embedder=seed_embedder)
    assert cache.get_stats()["total_pages"] == 3
    assert len(seed_embedder.calls[-1]) == 3  # all three embedded on rebuild

    # Now change just one page's body and sync incrementally.
    changed_path = _add_page(
        project,
        "concepts/page-1",
        "Page 1 updated",
        body="# Page 1\n\nFreshly edited body.\n",
    )
    embedder = _SpyEmbedder()
    cache.sync_from_files(
        project, changed_files=[changed_path], embedder=embedder
    )

    # Only the edited page should have been embedded, not all three.
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 1

    # The row exists with the new title.
    row = cache.get_page("concepts/page-1")
    assert row is not None
    assert row["title"] == "Page 1 updated"


def test_incremental_sync_leaves_unchanged_rows_embedding_intact(
    project: Path,
) -> None:
    """Embeddings for pages not in changed_files must be preserved."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    _add_page(project, "concepts/alpha", "Alpha")
    _add_page(project, "concepts/beta", "Beta")
    cache.full_rebuild(project, embedder=_SpyEmbedder())

    # Capture the alpha row's embedding before the incremental sync.
    conn = sqlite3.connect(str(project / "_registry" / "wiki.db"))
    before = conn.execute(
        "SELECT embedding FROM pages WHERE page_id = ?", ("concepts/alpha",)
    ).fetchone()[0]
    conn.close()

    # Incremental sync that only touches beta.
    beta_path = project / "wiki" / "concepts" / "beta.md"
    cache.sync_from_files(
        project, changed_files=[beta_path], embedder=_SpyEmbedder()
    )

    conn = sqlite3.connect(str(project / "_registry" / "wiki.db"))
    after = conn.execute(
        "SELECT embedding FROM pages WHERE page_id = ?", ("concepts/alpha",)
    ).fetchone()[0]
    conn.close()
    assert before == after  # alpha's embedding untouched


def test_incremental_sync_drops_row_for_missing_file(project: Path) -> None:
    """A changed_files entry whose file no longer exists drops its row."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    page_path = _add_page(project, "concepts/goner", "Goner")
    cache.full_rebuild(project)
    assert cache.get_page("concepts/goner") is not None

    # Delete the file AND the manifest entry the way a purge would.
    page_path.unlink()
    registry = Registry(project / "_registry")
    del registry._pages["concepts/goner"]  # noqa: SLF001
    registry.save()

    cache.sync_from_files(project, changed_files=[page_path])
    assert cache.get_page("concepts/goner") is None


def test_incremental_sync_keeps_row_when_file_archived(project: Path) -> None:
    """Deprecate/merge moves the file to ``archive/`` but keeps the
    manifest entry. The cache row must stay (refreshed from the registry,
    body empty) so ``wikiloom status`` reports the deprecated count
    accurately and the incremental path agrees with ``full_rebuild``.
    """
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    page_path = _add_page(project, "concepts/loser", "Loser")
    cache.full_rebuild(project)
    assert cache.get_page("concepts/loser") is not None

    # Simulate the deprecate/merge path: file gone from canonical
    # location, manifest entry still present with status=deprecated.
    page_path.unlink()
    registry = Registry(project / "_registry")
    registry.pages["concepts/loser"].status = "deprecated"
    registry.save()

    cache.sync_from_files(project, changed_files=[page_path])
    row = cache.get_page("concepts/loser")
    assert row is not None
    assert row["status"] == "deprecated"


def test_incremental_sync_empty_list_is_noop(project: Path) -> None:
    """An empty changed_files must not touch the cache."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    _add_page(project, "concepts/foo", "Foo")
    cache.full_rebuild(project, embedder=_SpyEmbedder())

    embedder = _SpyEmbedder()
    cache.sync_from_files(project, changed_files=[], embedder=embedder)
    # No embedding calls, no changes.
    assert embedder.calls == []
    assert cache.get_stats()["total_pages"] == 1


def test_incremental_sync_dedupes_duplicate_paths(project: Path) -> None:
    """Duplicate paths in ``changed_files`` must not trip the
    executemany INSERT's UNIQUE constraint on ``pages.page_id``.

    Regression: ``wikiloom duplicates --review`` builds the
    ``touched_paths`` list by appending ``(winner.md, loser.md)``
    per merged pair. When a page is winner of one pair AND loser
    of another (or winner across multiple pairs), the same path
    appears twice — and the page row landed in ``page_rows`` twice,
    crashing the executemany INSERT with
    ``UNIQUE constraint failed: pages.page_id``. Surfaced in 0.1.6
    smoke testing.
    """
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    page_path = _add_page(project, "concepts/foo", "Foo")
    cache.full_rebuild(project)

    # The same path listed twice — the call must succeed and the
    # page must end up represented exactly once in the cache.
    cache.sync_from_files(project, changed_files=[page_path, page_path])

    conn = sqlite3.connect(str(project / "_registry" / "wiki.db"))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM pages WHERE page_id = ?",
            ("concepts/foo",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_incremental_sync_ignores_non_page_paths(project: Path) -> None:
    """Non-wiki paths in changed_files don't produce spurious deletes."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    _add_page(project, "concepts/foo", "Foo")
    cache.full_rebuild(project, embedder=_SpyEmbedder())

    embedder = _SpyEmbedder()
    # Pass paths outside wiki/ and non-.md files — both should be ignored.
    cache.sync_from_files(
        project,
        changed_files=[
            project / "_registry" / "manifest.json",
            project / "wiki" / "log.md",
        ],
        embedder=embedder,
    )
    assert embedder.calls == []
    assert cache.get_page("concepts/foo") is not None


def test_none_changed_files_still_full_rebuilds(project: Path) -> None:
    """Calling with changed_files=None must preserve the existing
    full-rebuild fallback used by rebuild-cache and broad writers."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    _add_page(project, "concepts/foo", "Foo")
    _add_page(project, "concepts/bar", "Bar")
    embedder = _SpyEmbedder()
    cache.sync_from_files(project, changed_files=None, embedder=embedder)
    # Full rebuild embeds every page.
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 2


def test_incremental_sync_batches_embed_calls(project: Path) -> None:
    """A single _incremental_sync over many changed files must call
    embed_texts in fixed-size batches (32) rather than one giant call.

    Regression guard: a previous implementation passed every text in
    one call, which caused the whole batch to fail silently when a
    single over-length page body exceeded the backend's max sequence
    length."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    # Seed 70 pages — should produce 3 batches (32 + 32 + 6).
    paths: list[Path] = []
    for i in range(70):
        p = _add_page(project, f"concepts/p{i:03d}", f"Page {i}")
        paths.append(p)

    embedder = _SpyEmbedder()
    cache.sync_from_files(project, changed_files=paths, embedder=embedder)

    assert len(embedder.calls) == 3
    assert [len(c) for c in embedder.calls] == [32, 32, 6]


class _FailingEmbedder:
    """Always raises on embed_texts — simulates model-input overflow."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("synthetic failure: input too long")


def test_incremental_sync_surfaces_embedder_failure(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When embed_texts raises, the previous behavior silently produced
    NULL embeddings with no feedback. Now we write a warning to stderr
    so users can diagnose without digging into the cache."""
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    path = _add_page(project, "concepts/foo", "Foo")

    cache.sync_from_files(
        project, changed_files=[path], embedder=_FailingEmbedder()
    )

    captured = capsys.readouterr()
    assert "embed_texts failed" in captured.err
    assert "NULL embeddings" in captured.err
    # Page row still upserted; just with NULL embedding.
    assert cache.get_page("concepts/foo") is not None


# ----------------------------------------------------------------------
# Edge queries (get_inbound_edges / get_outbound_edges)
# ----------------------------------------------------------------------


def _seed_edges(cache: SQLiteCache, edges: list[tuple[str, str]]) -> None:
    """Insert (source, target) edges directly into the backlinks table."""
    with cache._connect() as conn:
        for src, tgt in edges:
            conn.execute(
                "INSERT INTO backlinks (source_page, target_page, linked_at) "
                "VALUES (?, ?, '2026-01-01')",
                (src, tgt),
            )


def test_get_inbound_edges_returns_sources_for_target(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "wiki.db")
    _seed_edges(cache, [
        ("concepts/a", "concepts/b"),
        ("concepts/c", "concepts/b"),
        ("concepts/a", "concepts/d"),
    ])
    edges = cache.get_inbound_edges(["concepts/b"])
    sources = {e["source_page"] for e in edges}
    assert sources == {"concepts/a", "concepts/c"}
    assert all(e["target_page"] == "concepts/b" for e in edges)


def test_get_inbound_edges_empty_input(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "wiki.db")
    assert cache.get_inbound_edges([]) == []


def test_get_inbound_edges_no_inbound_links(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "wiki.db")
    _seed_edges(cache, [("concepts/a", "concepts/b")])
    assert cache.get_inbound_edges(["concepts/c"]) == []


def test_get_inbound_edges_batches_multiple_targets(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "wiki.db")
    _seed_edges(cache, [
        ("concepts/a", "concepts/b"),
        ("concepts/a", "concepts/d"),
        ("concepts/c", "concepts/d"),
    ])
    edges = cache.get_inbound_edges(["concepts/b", "concepts/d"])
    assert len(edges) == 3
    assert {e["target_page"] for e in edges} == {"concepts/b", "concepts/d"}


def test_get_inbound_edges_does_not_return_outbound_direction(tmp_path: Path) -> None:
    """Asking for X's inbound must not return rows where X is the source."""
    cache = SQLiteCache(tmp_path / "wiki.db")
    _seed_edges(cache, [("concepts/a", "concepts/b")])
    # ``a`` is the source of an edge but has zero inbound — should be empty.
    assert cache.get_inbound_edges(["concepts/a"]) == []


def test_get_inbound_edges_respects_custom_limit(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "wiki.db")
    edges = [(f"concepts/citer-{i}", "concepts/popular") for i in range(12)]
    _seed_edges(cache, edges)
    result = cache.get_inbound_edges(["concepts/popular"], limit=5)
    assert len(result) == 5


def test_get_inbound_edges_limit_larger_than_results(tmp_path: Path) -> None:
    """``limit`` is an upper bound, not a requirement — fewer rows is fine."""
    cache = SQLiteCache(tmp_path / "wiki.db")
    edges = [(f"concepts/citer-{i}", "concepts/popular") for i in range(3)]
    _seed_edges(cache, edges)
    result = cache.get_inbound_edges(["concepts/popular"], limit=100)
    assert len(result) == 3


# ----------------------------------------------------------------------
# chunk_pages projection (frontmatter sources.chunk_ids -> junction table)
# ----------------------------------------------------------------------


def _chunk_page_edges(cache: SQLiteCache) -> set[tuple[str, str]]:
    with cache._connect() as conn:
        return {
            (r[0], r[1])
            for r in conn.execute(
                "SELECT chunk_id, page_id FROM chunk_pages"
            ).fetchall()
        }


def test_full_rebuild_projects_chunk_pages_from_frontmatter(project: Path) -> None:
    """full_rebuild mirrors each page's sources.chunk_ids, including the
    one-chunk-to-many-pages edge the scalar chunks.page_id can't hold."""
    # c1 feeds BOTH pages (many-to-many); c2 feeds only auth.
    _add_page(project, "concepts/auth", "Auth", chunk_ids=["c1", "c2"])
    _add_page(project, "concepts/billing", "Billing", chunk_ids=["c1"])
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    assert _chunk_page_edges(cache) == {
        ("c1", "concepts/auth"),
        ("c1", "concepts/billing"),
        ("c2", "concepts/auth"),
    }


def test_incremental_sync_reprojects_chunk_pages_for_changed_page(
    project: Path,
) -> None:
    """Editing a page's sources re-derives its edges: dropped chunks are
    pruned, new ones added."""
    _add_page(project, "concepts/auth", "Auth", chunk_ids=["c1", "c2"])
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    # Re-synthesize: c2 dropped, c3 added.
    _add_page(project, "concepts/auth", "Auth", chunk_ids=["c1", "c3"])
    cache.sync_from_files(
        project, [project / "wiki" / "concepts" / "auth.md"]
    )
    assert _chunk_page_edges(cache) == {
        ("c1", "concepts/auth"),
        ("c3", "concepts/auth"),
    }


def test_incremental_sync_prunes_chunk_pages_on_page_removal(
    project: Path,
) -> None:
    """Removing a page drops its edges; a chunk shared with another page
    keeps that page's edge."""
    _add_page(project, "concepts/auth", "Auth", chunk_ids=["c1"])
    _add_page(project, "concepts/billing", "Billing", chunk_ids=["c1"])
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    # Retire billing from the manifest.
    registry = Registry(project / "_registry")
    del registry.pages["concepts/billing"]
    registry.save()
    cache.sync_from_files(
        project, [project / "wiki" / "concepts" / "billing.md"]
    )
    assert _chunk_page_edges(cache) == {("c1", "concepts/auth")}
