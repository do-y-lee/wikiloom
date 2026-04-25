"""Tests for wikiloom.events."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikiloom.events import EventType, WikiEvent, append_event, create_event


# ----------------------------------------------------------------------
# EventType
# ----------------------------------------------------------------------


def test_event_type_covers_every_logged_kind() -> None:
    expected = {
        "ingest",
        "query",
        "lint",
        "merge",
        "deprecate",
        "purge",
        "reindex",
        "relink",
        "related",
        "rebuild-cache",
        "human-edit",
        "schema-migration",
        "stub-created",
    }
    actual = {e.value for e in EventType}
    assert actual == expected


# ----------------------------------------------------------------------
# create_event
# ----------------------------------------------------------------------


def test_create_event_sets_timestamp_and_type() -> None:
    event = create_event(EventType.INGEST, description="paper.pdf")
    assert event.event_type == EventType.INGEST
    assert event.description == "paper.pdf"
    assert event.timestamp.endswith("Z")
    assert event.pages_created == []
    assert event.pages_updated == []


def test_create_event_passes_through_kwargs() -> None:
    event = create_event(
        EventType.INGEST,
        description="paper.pdf",
        pages_created=["concepts/x"],
        tokens_used=1234,
        cost_usd=0.05,
    )
    assert event.pages_created == ["concepts/x"]
    assert event.tokens_used == 1234
    assert event.cost_usd == 0.05


# ----------------------------------------------------------------------
# WikiEvent.to_log_entry
# ----------------------------------------------------------------------


def test_to_log_entry_has_header_line() -> None:
    event = WikiEvent(
        timestamp="2026-04-12T00:00:00Z",
        event_type=EventType.INGEST,
        description="paper.pdf",
    )
    entry = event.to_log_entry()
    assert entry.startswith("## [2026-04-12T00:00:00Z] ingest | paper.pdf")


def test_to_log_entry_includes_populated_fields_only() -> None:
    event = WikiEvent(
        timestamp="2026-04-12T00:00:00Z",
        event_type=EventType.INGEST,
        description="paper.pdf",
        pages_created=["concepts/x", "entities/y"],
        links_inserted=5,
        tokens_used=4200,
        cost_usd=0.03,
    )
    entry = event.to_log_entry()
    assert "**Created**: concepts/x, entities/y" in entry
    assert "**Links inserted**: 5" in entry
    assert "**Tokens used**: 4,200" in entry
    assert "**Cost**: $0.03" in entry
    # Empty fields are omitted
    assert "**Updated**" not in entry
    assert "**Deprecated**" not in entry


def test_to_log_entry_renders_deprecate_event() -> None:
    event = WikiEvent(
        timestamp="2026-04-12T00:00:00Z",
        event_type=EventType.DEPRECATE,
        description="entities/old → entities/new",
        pages_deprecated=["entities/old"],
    )
    entry = event.to_log_entry()
    assert "deprecate" in entry
    assert "**Deprecated**: entities/old" in entry


def test_to_log_entry_summarizes_contradictions_count() -> None:
    event = WikiEvent(
        timestamp="2026-04-12T00:00:00Z",
        event_type=EventType.INGEST,
        description="paper.pdf",
        contradictions=[
            {"existing": "a", "new": "b", "source": "p1"},
            {"existing": "c", "new": "d", "source": "p2"},
        ],
    )
    entry = event.to_log_entry()
    assert "**Contradictions**: 2" in entry


def test_to_log_entry_truncates_commit_hash() -> None:
    event = WikiEvent(
        timestamp="2026-04-12T00:00:00Z",
        event_type=EventType.INGEST,
        description="paper.pdf",
        git_commit_hash="abcdef1234567890",
    )
    entry = event.to_log_entry()
    assert "**Commit**: abcdef12" in entry


# ----------------------------------------------------------------------
# append_event
# ----------------------------------------------------------------------


def test_append_event_creates_file_with_header(tmp_path: Path) -> None:
    log_path = tmp_path / "log.md"
    event = create_event(EventType.INGEST, description="paper.pdf")
    append_event(log_path, event)

    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("# WikiLoom Event Log\n\n")
    assert "ingest | paper.pdf" in content


def test_append_event_appends_to_existing_file(tmp_path: Path) -> None:
    log_path = tmp_path / "log.md"
    log_path.write_text("# WikiLoom Event Log\n\nold content\n", encoding="utf-8")

    event = create_event(EventType.INGEST, description="paper.pdf")
    append_event(log_path, event)

    content = log_path.read_text(encoding="utf-8")
    assert "old content" in content
    assert "ingest | paper.pdf" in content
    # Order: existing first, new entry after
    assert content.index("old content") < content.index("ingest | paper.pdf")


def test_append_event_creates_parent_dir_if_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "wiki" / "log.md"
    event = create_event(EventType.INGEST, description="paper.pdf")
    append_event(log_path, event)
    assert log_path.exists()


def test_append_multiple_events(tmp_path: Path) -> None:
    log_path = tmp_path / "log.md"
    append_event(log_path, create_event(EventType.INGEST, description="a.md"))
    append_event(log_path, create_event(EventType.INGEST, description="b.md"))
    append_event(log_path, create_event(EventType.DEPRECATE, description="entities/old"))

    content = log_path.read_text(encoding="utf-8")
    assert content.count("## [") == 3
    assert "a.md" in content
    assert "b.md" in content
    assert "deprecate | entities/old" in content
