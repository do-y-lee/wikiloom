"""Tests for wikiloom.registry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from wikiloom.registry import PageEntry, Registry


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a minimal project layout with _registry/ and wiki/."""
    (tmp_path / "_registry").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "entities").mkdir()
    (tmp_path / "wiki" / "concepts").mkdir()
    return tmp_path


@pytest.fixture
def registry(project: Path) -> Registry:
    return Registry(project / "_registry")


def _entry(title: str, type_: str = "entity", **kwargs) -> PageEntry:
    return PageEntry(title=title, type=type_, **kwargs)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_save_and_reload_roundtrip(project: Path) -> None:
    reg = Registry(project / "_registry")
    reg.register_page(
        "entities/google-brain",
        _entry("Google Brain", aliases=["google brain", "google ai"], summary="AI lab"),
    )
    reg.save()

    reg2 = Registry(project / "_registry")
    page = reg2.get_page("entities/google-brain")
    assert page is not None
    assert page.title == "Google Brain"
    assert page.aliases == ["google brain", "google ai"]
    assert page.summary == "AI lab"
    assert page.page_id == "entities/google-brain"


def test_register_sets_timestamps(registry: Registry) -> None:
    entry = _entry("Flash Attention", type_="concept")
    registry.register_page("concepts/flash-attention", entry)
    assert entry.created
    assert entry.modified
    assert entry.page_id == "concepts/flash-attention"


# ----------------------------------------------------------------------
# Read API
# ----------------------------------------------------------------------


def test_get_page_returns_none_for_unknown(registry: Registry) -> None:
    assert registry.get_page("entities/nope") is None


def test_get_page_list_excludes_deprecated(registry: Registry) -> None:
    registry.register_page("entities/active", _entry("Active"))
    registry.register_page("entities/dead", _entry("Dead", status="deprecated"))
    items = registry.get_page_list()
    assert len(items) == 1
    assert items[0].page_id == "entities/active"


def test_find_by_alias_matches_title_case_insensitively(registry: Registry) -> None:
    registry.register_page("entities/openai", _entry("OpenAI"))
    page = registry.find_by_alias("openai")
    assert page is not None
    assert page.title == "OpenAI"


def test_find_by_alias_matches_alternate_names(registry: Registry) -> None:
    registry.register_page(
        "entities/google-brain",
        _entry("Google Brain", aliases=["google ai research"]),
    )
    page = registry.find_by_alias("Google AI Research")
    assert page is not None
    assert page.title == "Google Brain"


def test_get_all_aliases_includes_titles_and_aliases(registry: Registry) -> None:
    registry.register_page(
        "concepts/attention",
        _entry("Attention", type_="concept", aliases=["attention mechanism"]),
    )
    aliases = registry.get_all_aliases()
    assert aliases["attention"] == "concepts/attention"
    assert aliases["attention mechanism"] == "concepts/attention"


def test_get_all_aliases_excludes_deprecated(registry: Registry) -> None:
    registry.register_page("entities/old", _entry("Old", status="deprecated"))
    assert "old" not in registry.get_all_aliases()


# ----------------------------------------------------------------------
# Stale pages
# ----------------------------------------------------------------------


def test_get_stale_pages_returns_only_old_pages(registry: Registry) -> None:
    fresh_modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    old_modified = (
        datetime.now(timezone.utc) - timedelta(days=400)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    registry.register_page("entities/fresh", _entry("Fresh"))
    # Override the modified timestamp set by register_page
    registry._pages["entities/fresh"].modified = fresh_modified
    registry._pages["entities/fresh"].dormant_window_days = 90

    registry.register_page("entities/stale", _entry("Stale"))
    registry._pages["entities/stale"].modified = old_modified
    registry._pages["entities/stale"].dormant_window_days = 90

    stale = registry.get_stale_pages()
    assert len(stale) == 1
    assert stale[0].title == "Stale"
    # Return type is list[PageEntry], not list[tuple]
    assert isinstance(stale[0], PageEntry)


# ----------------------------------------------------------------------
# Update bulk method
# ----------------------------------------------------------------------


def test_update_accepts_tuples(registry: Registry) -> None:
    registry.update([
        ("entities/a", _entry("A")),
        ("entities/b", _entry("B")),
    ])
    assert registry.get_page("entities/a") is not None
    assert registry.get_page("entities/b") is not None


def test_update_accepts_entries_with_page_id(registry: Registry) -> None:
    a = _entry("A")
    a.page_id = "entities/a"
    registry.update([a])
    assert registry.get_page("entities/a") is not None


def test_update_rejects_entry_without_page_id(registry: Registry) -> None:
    with pytest.raises(ValueError):
        registry.update([_entry("nameless")])


# ----------------------------------------------------------------------
# Deprecation
# ----------------------------------------------------------------------


def test_deprecate_page_sets_status_and_supersedes(registry: Registry) -> None:
    registry.register_page("entities/old", _entry("Old"))
    registry.register_page("entities/new", _entry("New"))
    registry.deprecate_page("entities/old", superseded_by="entities/new", move_to_archive=False)

    page = registry.get_page("entities/old")
    assert page is not None
    assert page.status == "deprecated"
    assert page.superseded_by == "entities/new"


def test_deprecate_page_moves_file_to_archive(project: Path) -> None:
    reg = Registry(project / "_registry")
    page_path = project / "wiki" / "entities" / "old.md"
    page_path.write_text("# Old\n\nbody", encoding="utf-8")

    reg.register_page("entities/old", _entry("Old"))
    archive_path = reg.deprecate_page("entities/old")

    assert archive_path is not None
    assert archive_path.exists()
    assert archive_path.parent == project / "wiki" / "archive"
    # Original location is gone
    assert not page_path.exists()
    # Archive name preserves the original category
    assert archive_path.name == "entities__old.md"


def test_deprecate_page_handles_missing_file(project: Path) -> None:
    reg = Registry(project / "_registry")
    reg.register_page("entities/nonexistent", _entry("Nope"))
    # No .md file on disk — should still update status, no exception
    result = reg.deprecate_page("entities/nonexistent")
    assert result is None
    assert reg.get_page("entities/nonexistent").status == "deprecated"


def test_deprecate_unknown_page_is_noop(registry: Registry) -> None:
    assert registry.deprecate_page("entities/ghost") is None


# ----------------------------------------------------------------------
# Add alias
# ----------------------------------------------------------------------


def test_add_alias_appends_new_alias(registry: Registry) -> None:
    registry.register_page("entities/openai", _entry("OpenAI"))
    registry.add_alias("entities/openai", "Open AI")
    page = registry.get_page("entities/openai")
    assert "open ai" in page.aliases


def test_add_alias_dedupes(registry: Registry) -> None:
    registry.register_page("entities/openai", _entry("OpenAI", aliases=["openai"]))
    registry.add_alias("entities/openai", "OpenAI")
    page = registry.get_page("entities/openai")
    # Should not have duplicate "openai"
    assert page.aliases.count("openai") == 1


def test_add_alias_unknown_page_raises(registry: Registry) -> None:
    with pytest.raises(KeyError):
        registry.add_alias("entities/ghost", "alias")
