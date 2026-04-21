"""Tests for observability commands: status, log, cost.

These test the underlying data functions (parse_log, cache stats)
rather than Click's CLI runner, keeping the tests fast and focused
on correctness of the parsed output.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from wikiloom.cache import SQLiteCache
from wikiloom.chunk_store import ChunkStore
from wikiloom.events import (
    EventType,
    append_event,
    create_event,
    parse_log,
)
from wikiloom.frontmatter import Frontmatter, write_page
from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = init_project(name="obs-test", path=tmp_path, domain="test")
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


_DIR_TO_TYPE = {
    "entities": "entity",
    "concepts": "concept",
    "sources": "source",
    "syntheses": "synthesis",
    "decisions": "decision",
}


def _add_page(project: Path, page_id: str, title: str) -> None:
    category = page_id.split("/")[0]
    fm = Frontmatter(
        title=title,
        type=_DIR_TO_TYPE.get(category, category),
        created="2026-01-01T00:00:00Z",
        modified="2026-01-01T00:00:00Z",
        summary=f"Summary of {title}.",
    )
    write_page(project / "wiki" / f"{page_id}.md", fm, f"# {title}\n\nBody.\n")
    registry = Registry(project / "_registry")
    registry.register_page(
        page_id,
        PageEntry(title=title, type=fm.type, summary=fm.summary),
    )
    registry.save()


def _emit_event(
    project: Path,
    event_type: EventType = EventType.INGEST,
    description: str = "test.md",
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    pages_created: list[str] | None = None,
) -> None:
    event = create_event(
        event_type,
        description=description,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        pages_created=pages_created or [],
        git_commit_hash="abc12345",
    )
    append_event(project / "wiki" / "log.md", event)


# ----------------------------------------------------------------------
# parse_log
# ----------------------------------------------------------------------


def test_parse_log_empty_project(project: Path) -> None:
    entries = parse_log(project / "wiki" / "log.md")
    assert entries == []


def test_parse_log_single_event(project: Path) -> None:
    _emit_event(project, tokens_used=500, cost_usd=0.01)
    entries = parse_log(project / "wiki" / "log.md")
    assert len(entries) == 1
    assert entries[0]["event_type"] == "ingest"
    assert entries[0]["description"] == "test.md"
    assert entries[0]["tokens_used"] == 500
    assert entries[0]["cost_usd"] == 0.01


def test_parse_log_multiple_events_newest_first(project: Path) -> None:
    _emit_event(project, description="first.md", tokens_used=100)
    _emit_event(project, description="second.md", tokens_used=200)
    _emit_event(project, description="third.md", tokens_used=300)

    entries = parse_log(project / "wiki" / "log.md")
    assert len(entries) == 3
    assert entries[0]["description"] == "third.md"
    assert entries[2]["description"] == "first.md"


def test_parse_log_different_event_types(project: Path) -> None:
    _emit_event(project, EventType.INGEST, "paper.pdf", tokens_used=800)
    _emit_event(project, EventType.LINT, "full scan")
    _emit_event(project, EventType.DEPRECATE, "concepts/old")

    entries = parse_log(project / "wiki" / "log.md")
    types = {e["event_type"] for e in entries}
    assert types == {"ingest", "lint", "deprecate"}


def test_parse_log_handles_comma_formatted_tokens(project: Path) -> None:
    _emit_event(project, tokens_used=1500, cost_usd=0.05)
    entries = parse_log(project / "wiki" / "log.md")
    assert entries[0]["tokens_used"] == 1500


def test_parse_log_missing_file(tmp_path: Path) -> None:
    entries = parse_log(tmp_path / "nonexistent" / "log.md")
    assert entries == []


# ----------------------------------------------------------------------
# SQLiteCache.get_stats (used by status command)
# ----------------------------------------------------------------------


def test_status_stats_empty_project(project: Path) -> None:
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    stats = cache.get_stats()
    assert stats["total_pages"] == 0
    assert stats["backlinks"] == 0


def test_status_stats_with_pages(project: Path) -> None:
    _add_page(project, "concepts/alpha", "Alpha")
    _add_page(project, "concepts/beta", "Beta")
    _add_page(project, "entities/org-x", "Org X")

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    stats = cache.get_stats()
    assert stats["total_pages"] == 3
    assert stats["by_type"] == {"concept": 2, "entity": 1}


# ----------------------------------------------------------------------
# ChunkStore.count (used by status command)
# ----------------------------------------------------------------------


def test_status_chunk_count(project: Path) -> None:
    store = ChunkStore(project / "_registry" / "wiki.db")
    assert store.count() == 0

    chunks = [
        ExtractedContent(
            text="chunk text",
            metadata={"chunk_index": 0, "chunk_total": 1},
            source_path=None,
            content_type="markdown",
            extraction_method="test",
            token_estimate=50,
        )
    ]
    store.persist_chunks("hash-abc", chunks)
    assert store.count() == 1


# ----------------------------------------------------------------------
# Cost accumulation (used by cost command)
# ----------------------------------------------------------------------


def test_cost_accumulates_across_events(project: Path) -> None:
    _emit_event(project, EventType.INGEST, "paper1.pdf", tokens_used=500, cost_usd=0.01)
    _emit_event(project, EventType.INGEST, "paper2.pdf", tokens_used=800, cost_usd=0.02)
    _emit_event(project, EventType.QUERY, "query1", tokens_used=200, cost_usd=0.05)

    events = parse_log(project / "wiki" / "log.md")
    total_tokens = sum(int(e.get("tokens_used", 0)) for e in events)
    total_cost = sum(float(e.get("cost_usd", 0.0)) for e in events)

    assert total_tokens == 1500
    assert total_cost == pytest.approx(0.08)


def test_cost_groups_by_event_type(project: Path) -> None:
    # Use values that survive 2-decimal-place formatting in the log
    _emit_event(project, EventType.INGEST, "p1.pdf", tokens_used=500, cost_usd=0.05)
    _emit_event(project, EventType.INGEST, "p2.pdf", tokens_used=300, cost_usd=0.03)
    _emit_event(project, EventType.QUERY, "q1", tokens_used=100, cost_usd=0.01)

    events = parse_log(project / "wiki" / "log.md")
    by_type: dict[str, float] = {}
    for e in events:
        t = str(e.get("event_type", "other"))
        by_type[t] = by_type.get(t, 0.0) + float(e.get("cost_usd", 0.0))

    assert by_type["ingest"] == pytest.approx(0.08)
    assert by_type["query"] == pytest.approx(0.01)


def test_cost_handles_zero_cost_events(project: Path) -> None:
    _emit_event(project, EventType.LINT, "full scan")  # no tokens/cost
    _emit_event(project, EventType.INGEST, "p.pdf", tokens_used=100, cost_usd=0.05)

    events = parse_log(project / "wiki" / "log.md")
    total_cost = sum(float(e.get("cost_usd", 0.0)) for e in events)
    assert total_cost == pytest.approx(0.05)


def test_post_flight_budget_warning_reads_dict_events(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: parse_log returns dicts, not objects. The post-flight
    warning once used attribute access (``e.cost_usd``) and crashed
    right after a successful ingest.
    """
    import tomli_w

    from wikiloom.cli import _post_flight_budget_warning

    # Set an unreachably low budget so the warning fires.
    toml_path = project / "wikiloom.toml"
    import tomllib

    with toml_path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("llm", {})["monthly_budget_usd"] = 0.01
    toml_path.write_bytes(tomli_w.dumps(cfg).encode())

    _emit_event(project, EventType.INGEST, "p.pdf", tokens_used=100, cost_usd=0.50)

    _post_flight_budget_warning(project)  # must not raise

    captured = capsys.readouterr()
    assert "Budget warning" in captured.err
    assert "0.50" in captured.err
