"""Tests for wikiloom.search.IndexUpdater."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikiloom.frontmatter import Frontmatter, render_frontmatter
from wikiloom.registry import PageEntry, Registry
from wikiloom.search import IndexUpdater


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "_registry").mkdir()
    wiki = tmp_path / "wiki"
    for sub in ("concepts", "entities", "sources", "syntheses", "decisions", "archive"):
        sub_dir = wiki / sub
        sub_dir.mkdir(parents=True)
        fm = Frontmatter(
            title=f"{sub.title()} Index",
            type="index",
            status="active",
            created="2026-04-01T00:00:00Z",
            modified="2026-04-01T00:00:00Z",
            summary=f"Index of {sub}",
        )
        (sub_dir / "index.md").write_text(
            render_frontmatter(fm) + "\n# placeholder\n", encoding="utf-8"
        )
    return tmp_path


def _write_page(
    project: Path,
    rel: str,
    *,
    title: str,
    summary: str,
    modified: str,
    source_count: int = 0,
) -> Path:
    path = project / "wiki" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = Frontmatter(
        title=title,
        type=path.parent.name.rstrip("s") or "concept",
        status="active",
        created="2026-04-01T00:00:00Z",
        modified=modified,
        summary=summary,
        source_count=source_count,
    )
    path.write_text(render_frontmatter(fm) + "\nbody\n", encoding="utf-8")
    return path


def _registry_with(project: Path, pages: dict[str, PageEntry]) -> Registry:
    reg = Registry(project / "_registry", project / "wiki")
    for page_id, entry in pages.items():
        reg.register_page(page_id, entry)
    reg.save()
    return reg


# ----------------------------------------------------------------------
# rebuild_sub_index
# ----------------------------------------------------------------------


def test_rebuild_sub_index_generates_table(project: Path) -> None:
    _write_page(
        project,
        "concepts/transformer.md",
        title="Transformer",
        summary="Encoder-decoder architecture",
        modified="2026-04-05T00:00:00Z",
        source_count=7,
    )
    reg = _registry_with(
        project,
        {
            "concepts/transformer": PageEntry(
                title="Transformer", type="concept", inbound_link_count=23
            )
        },
    )

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_sub_index(project / "wiki" / "concepts")

    text = (project / "wiki" / "concepts" / "index.md").read_text()
    assert "# Concepts Index (1 pages)" in text
    assert "| [Transformer](transformer.md) |" in text
    assert "Encoder-decoder architecture" in text
    assert "| 2026-04-05 |" in text
    assert "| 7 |" in text  # source count
    assert "| 23 |" in text  # inbound


def test_rebuild_sub_index_sorts_newest_first(project: Path) -> None:
    _write_page(
        project,
        "concepts/old.md",
        title="Old",
        summary="s",
        modified="2026-03-01T00:00:00Z",
    )
    _write_page(
        project,
        "concepts/new.md",
        title="New",
        summary="s",
        modified="2026-04-05T00:00:00Z",
    )
    _write_page(
        project,
        "concepts/middle.md",
        title="Middle",
        summary="s",
        modified="2026-04-01T00:00:00Z",
    )
    reg = _registry_with(
        project,
        {
            "concepts/old": PageEntry(title="Old", type="concept"),
            "concepts/new": PageEntry(title="New", type="concept"),
            "concepts/middle": PageEntry(title="Middle", type="concept"),
        },
    )

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_sub_index(project / "wiki" / "concepts")

    text = (project / "wiki" / "concepts" / "index.md").read_text()
    new_idx = text.index("| [New]")
    middle_idx = text.index("| [Middle]")
    old_idx = text.index("| [Old]")
    assert new_idx < middle_idx < old_idx


def test_rebuild_sub_index_preserves_frontmatter(project: Path) -> None:
    _write_page(
        project,
        "concepts/a.md",
        title="A",
        summary="s",
        modified="2026-04-05T00:00:00Z",
    )
    reg = _registry_with(
        project, {"concepts/a": PageEntry(title="A", type="concept")}
    )

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_sub_index(project / "wiki" / "concepts")

    text = (project / "wiki" / "concepts" / "index.md").read_text()
    assert text.startswith("---\n")
    assert "title: Concepts Index" in text
    assert "type: index" in text


def test_rebuild_sub_index_handles_empty_directory(project: Path) -> None:
    reg = _registry_with(project, {})
    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_sub_index(project / "wiki" / "concepts")

    text = (project / "wiki" / "concepts" / "index.md").read_text()
    assert "# Concepts Index (0 pages)" in text
    assert "*No pages yet.*" in text


def test_rebuild_sub_index_missing_modified_sorts_last(project: Path) -> None:
    _write_page(
        project,
        "concepts/new.md",
        title="New",
        summary="s",
        modified="2026-04-05T00:00:00Z",
    )
    _write_page(
        project,
        "concepts/blank.md",
        title="Blank",
        summary="s",
        modified="",
    )
    reg = _registry_with(
        project,
        {
            "concepts/new": PageEntry(title="New", type="concept"),
            "concepts/blank": PageEntry(title="Blank", type="concept"),
        },
    )

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_sub_index(project / "wiki" / "concepts")

    text = (project / "wiki" / "concepts" / "index.md").read_text()
    assert text.index("| [New]") < text.index("| [Blank]")


def test_rebuild_sub_index_escapes_pipes_in_summary(project: Path) -> None:
    _write_page(
        project,
        "concepts/a.md",
        title="A",
        summary="foo | bar",
        modified="2026-04-05T00:00:00Z",
    )
    reg = _registry_with(
        project, {"concepts/a": PageEntry(title="A", type="concept")}
    )

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_sub_index(project / "wiki" / "concepts")

    text = (project / "wiki" / "concepts" / "index.md").read_text()
    assert r"foo \| bar" in text


def test_rebuild_sub_index_is_deterministic(project: Path) -> None:
    _write_page(
        project,
        "concepts/a.md",
        title="A",
        summary="s",
        modified="2026-04-05T00:00:00Z",
    )
    _write_page(
        project,
        "concepts/b.md",
        title="B",
        summary="s",
        modified="2026-04-05T00:00:00Z",
    )
    reg = _registry_with(
        project,
        {
            "concepts/a": PageEntry(title="A", type="concept"),
            "concepts/b": PageEntry(title="B", type="concept"),
        },
    )

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_sub_index(project / "wiki" / "concepts")
    first = (project / "wiki" / "concepts" / "index.md").read_text()

    updater.rebuild_sub_index(project / "wiki" / "concepts")
    second = (project / "wiki" / "concepts" / "index.md").read_text()

    assert first == second


# ----------------------------------------------------------------------
# rebuild_root_index
# ----------------------------------------------------------------------


def test_rebuild_root_index_lists_all_categories(project: Path) -> None:
    _write_page(
        project, "concepts/a.md", title="A", summary="s", modified="2026-04-05T00:00:00Z"
    )
    _write_page(
        project, "concepts/b.md", title="B", summary="s", modified="2026-04-05T00:00:00Z"
    )
    _write_page(
        project, "entities/x.md", title="X", summary="s", modified="2026-04-05T00:00:00Z"
    )
    reg = _registry_with(project, {})

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_root_index()

    text = (project / "wiki" / "index.md").read_text()
    assert "# Wiki Index" in text
    assert "## Concepts (2 pages)" in text
    assert "## Entities (1 pages)" in text
    assert "## Sources (0 pages)" in text
    assert "→ See [concepts/index.md](concepts/index.md)" in text


def test_rebuild_root_index_excludes_archive(project: Path) -> None:
    (project / "wiki" / "archive" / "old.md").write_text("body", encoding="utf-8")
    reg = _registry_with(project, {})

    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_root_index()

    text = (project / "wiki" / "index.md").read_text()
    assert "archive" not in text.lower()


def test_rebuild_root_index_includes_descriptions(project: Path) -> None:
    reg = _registry_with(project, {})
    updater = IndexUpdater(project / "wiki", registry=reg)
    updater.rebuild_root_index()

    text = (project / "wiki" / "index.md").read_text()
    assert "People, organizations, tools, projects." in text
    assert "Ideas, methods, patterns, principles." in text


# ----------------------------------------------------------------------
# rebuild_all
# ----------------------------------------------------------------------


def test_rebuild_all_writes_root_and_sub_indexes(project: Path) -> None:
    _write_page(
        project, "concepts/a.md", title="A", summary="s", modified="2026-04-05T00:00:00Z"
    )
    _write_page(
        project, "entities/x.md", title="X", summary="s", modified="2026-04-05T00:00:00Z"
    )
    reg = _registry_with(project, {})

    updater = IndexUpdater(project / "wiki", registry=reg)
    written = updater.rebuild_all()

    names = {p.name for p in written}
    parents = {p.parent.name for p in written}
    assert names == {"index.md"}
    assert "wiki" in parents
    assert "concepts" in parents
    assert "entities" in parents


def test_rebuild_all_skips_archive(project: Path) -> None:
    (project / "wiki" / "archive" / "old.md").write_text("body", encoding="utf-8")
    reg = _registry_with(project, {})

    updater = IndexUpdater(project / "wiki", registry=reg)
    written = updater.rebuild_all()

    assert not any(p.parent.name == "archive" for p in written)
