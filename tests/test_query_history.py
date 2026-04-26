"""Tests for wikiloom.query_history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wikiloom.query_history import (
    HISTORY_FILENAME,
    LEGACY_LAST_QUERY_FILENAME,
    QueryHistory,
    QueryHistoryEntry,
    derive_query_id,
)


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    d = tmp_path / "_registry"
    d.mkdir()
    return d


def _entry(question: str = "What is X?", **overrides) -> QueryHistoryEntry:
    base = dict(
        query_id=derive_query_id(question, "2026-04-26T12:00:00Z"),
        timestamp="2026-04-26T12:00:00Z",
        question=question,
        answer="X is a thing.",
        confidence="high",
        relevance="high",
    )
    base.update(overrides)
    return QueryHistoryEntry(**base)


def test_load_returns_empty_when_file_missing(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    assert h.entries == []
    assert h.path == registry_dir / HISTORY_FILENAME


def test_append_prepends_newest_first(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    h.append(_entry("first"), max_entries=10)
    h.append(_entry("second"), max_entries=10)
    assert [e.question for e in h.entries] == ["second", "first"]


def test_append_trims_to_max_entries(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    for i in range(5):
        h.append(_entry(f"q{i}"), max_entries=3)
    assert len(h.entries) == 3
    # Newest 3 retained: q4, q3, q2
    assert [e.question for e in h.entries] == ["q4", "q3", "q2"]


def test_save_and_reload_roundtrip(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    h.append(
        _entry(
            "round-trip",
            sources=[{"page_path": "concepts/foo", "relevance": "high"}],
            tokens_in=100,
            cost_usd=0.0015,
        ),
        max_entries=10,
    )
    h.save()

    h2 = QueryHistory.load(registry_dir)
    assert len(h2.entries) == 1
    e = h2.entries[0]
    assert e.question == "round-trip"
    assert e.sources == [{"page_path": "concepts/foo", "relevance": "high"}]
    assert e.tokens_in == 100
    assert e.cost_usd == 0.0015


def test_save_writes_schema_version(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    h.append(_entry(), max_entries=10)
    h.save()
    payload = json.loads(
        (registry_dir / HISTORY_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1
    assert isinstance(payload["entries"], list)


def test_save_is_atomic_no_partial_files(registry_dir: Path) -> None:
    """No leftover .tmp file should remain after a successful save."""
    h = QueryHistory.load(registry_dir)
    h.append(_entry(), max_entries=10)
    h.save()
    leftovers = list(registry_dir.glob("*.tmp"))
    assert leftovers == []


def test_get_by_query_id(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    h.append(_entry("first"), max_entries=10)
    target = h.entries[0].query_id
    assert h.get(target) is not None
    assert h.get(target).question == "first"


def test_get_by_query_id_prefix(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    h.append(_entry("first"), max_entries=10)
    full_id = h.entries[0].query_id
    # First 4 chars should be unique enough for a single entry
    assert h.get(full_id[:4]) is not None


def test_get_by_index_string(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    h.append(_entry("first"), max_entries=10)
    h.append(_entry("second"), max_entries=10)
    # 1 = most recent
    assert h.get("1").question == "second"
    assert h.get("2").question == "first"
    assert h.get("3") is None


def test_get_returns_none_for_unknown(registry_dir: Path) -> None:
    h = QueryHistory.load(registry_dir)
    h.append(_entry(), max_entries=10)
    assert h.get("nonexistent") is None


def test_migrate_legacy_seeds_history(registry_dir: Path) -> None:
    legacy = registry_dir / LEGACY_LAST_QUERY_FILENAME
    legacy.write_text(
        json.dumps(
            {
                "question": "old question",
                "answer": "old answer",
                "confidence": "medium",
                "sources_consulted": [{"page_path": "concepts/x"}],
                "tokens_in": 50,
                "cost_usd": 0.001,
                "timestamp": "2026-04-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    h = QueryHistory.load(registry_dir)
    migrated = h.migrate_legacy(registry_dir)

    assert migrated is True
    assert len(h.entries) == 1
    assert h.entries[0].question == "old question"
    assert h.entries[0].answer == "old answer"
    assert not legacy.exists(), "legacy file should be removed after migration"


def test_migrate_legacy_no_op_when_history_already_populated(
    registry_dir: Path,
) -> None:
    legacy = registry_dir / LEGACY_LAST_QUERY_FILENAME
    legacy.write_text('{"question": "old"}', encoding="utf-8")

    h = QueryHistory.load(registry_dir)
    h.append(_entry("new"), max_entries=10)
    migrated = h.migrate_legacy(registry_dir)

    assert migrated is False
    assert h.entries[0].question == "new"
    # Legacy file untouched when history was already populated.
    assert legacy.exists()


def test_migrate_legacy_handles_corrupt_file(registry_dir: Path) -> None:
    legacy = registry_dir / LEGACY_LAST_QUERY_FILENAME
    legacy.write_text("not json {{{", encoding="utf-8")

    h = QueryHistory.load(registry_dir)
    migrated = h.migrate_legacy(registry_dir)

    assert migrated is False
    assert h.entries == []
    # Corrupt legacy file should be cleaned up so we don't keep retrying.
    assert not legacy.exists()


def test_load_tolerates_corrupt_history_file(registry_dir: Path) -> None:
    (registry_dir / HISTORY_FILENAME).write_text(
        "not json", encoding="utf-8"
    )
    h = QueryHistory.load(registry_dir)
    assert h.entries == []


def test_derive_query_id_is_deterministic() -> None:
    a = derive_query_id("hello", "2026-04-26T12:00:00Z")
    b = derive_query_id("hello", "2026-04-26T12:00:00Z")
    assert a == b
    assert len(a) == 10


def test_derive_query_id_varies_by_input() -> None:
    a = derive_query_id("hello", "2026-04-26T12:00:00Z")
    b = derive_query_id("world", "2026-04-26T12:00:00Z")
    assert a != b
