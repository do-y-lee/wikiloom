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
) -> Path:
    """Write a page file + manifest entry the way the ingest pipeline would.

    Derives ``type`` from the page_id's top-level category if not given
    (e.g. ``concepts/foo`` → ``concept``). Returns the page file path.
    """
    category = page_id.split("/", 1)[0]
    inferred_type = type_ or category.rstrip("s")
    aliases = aliases or []
    summary_text = summary if summary is not None else f"Summary of {title}."
    body_text = body if body is not None else f"# {title}\n\nBody text.\n"

    fm = Frontmatter(
        title=title,
        type=inferred_type,
        status=status,
        created="2026-04-01T00:00:00Z",
        modified=modified,
        summary=summary_text,
        aliases=aliases,
        human_edited=human_edited,
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
        summary="An I/O-aware attention algorithm for GPUs.",
    )
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)

    results = cache.search("I/O-aware")
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
