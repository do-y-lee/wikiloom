"""Tests for wikiloom deprecate + purge CLI commands."""

from __future__ import annotations

from pathlib import Path

import git
import pytest
from click.testing import CliRunner

from wikiloom.cli import main
from wikiloom.frontmatter import Frontmatter, write_page
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Init a real project, configure git identity, and commit the scaffold."""
    project_dir = init_project(name="testproj", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    return project_dir


def _add_active_page(project_root: Path, page_id: str, title: str = "Foo") -> None:
    """Write a page file + manifest entry so commands have something to act on."""
    wiki_dir = project_root / "wiki"
    registry = Registry(project_root / "_registry", wiki_dir=wiki_dir)
    entry = PageEntry(
        title=title,
        type=page_id.split("/", 1)[0].rstrip("s"),
        page_id=page_id,
        created="2026-04-19T10:00:00Z",
        modified="2026-04-19T10:00:00Z",
    )
    registry.register_page(page_id, entry)
    registry.save()
    page_path = wiki_dir / f"{page_id}.md"
    write_page(
        page_path,
        Frontmatter(
            title=title,
            type=entry.type,
            created=entry.created,
            modified=entry.modified,
            summary=f"Summary of {title}",
        ),
        f"# {title}\n\nBody.\n",
    )
    # Commit the seeded state so the dirty-tree preflight passes.
    repo = git.Repo(project_root)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("ingest: seed")


# ----------------------------------------------------------------------
# deprecate
# ----------------------------------------------------------------------


def test_deprecate_moves_active_page_to_archive(project: Path) -> None:
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["deprecate", "concepts/foo", "--yes", "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert not (project / "wiki" / "concepts" / "foo.md").exists()
    assert (project / "wiki" / "archive" / "concepts__foo.md").exists()

    registry = Registry(project / "_registry")
    entry = registry.get_page("concepts/foo")
    assert entry is not None
    assert entry.status == "deprecated"
    assert entry.superseded_by is None


def test_deprecate_records_superseded_by(project: Path) -> None:
    _add_active_page(project, "concepts/old")
    _add_active_page(project, "concepts/new")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "deprecate", "concepts/old",
            "--superseded-by", "concepts/new",
            "--yes",
            "--project", str(project),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = Registry(project / "_registry").get_page("concepts/old")
    assert entry is not None
    assert entry.superseded_by == "concepts/new"


def test_deprecate_refuses_unknown_superseded_by(project: Path) -> None:
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "deprecate", "concepts/foo",
            "--superseded-by", "concepts/does-not-exist",
            "--yes",
            "--project", str(project),
        ],
    )

    assert result.exit_code != 0
    assert "does-not-exist" in result.output


def test_deprecate_refuses_already_deprecated(project: Path) -> None:
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()
    runner.invoke(
        main,
        ["deprecate", "concepts/foo", "--yes", "--project", str(project)],
    )

    result = runner.invoke(
        main,
        ["deprecate", "concepts/foo", "--yes", "--project", str(project)],
    )

    assert result.exit_code != 0
    assert "already deprecated" in result.output


# ----------------------------------------------------------------------
# purge
# ----------------------------------------------------------------------


def test_purge_refuses_active_page(project: Path) -> None:
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["purge", "concepts/foo", "--yes", "--project", str(project)],
    )

    assert result.exit_code != 0
    assert "not deprecated" in result.output
    # Page is still on disk and in the manifest.
    assert (project / "wiki" / "concepts" / "foo.md").exists()


def test_purge_removes_deprecated_page(project: Path) -> None:
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()
    # Deprecate first so the page is in archive.
    runner.invoke(
        main,
        ["deprecate", "concepts/foo", "--yes", "--project", str(project)],
    )
    archive_file = project / "wiki" / "archive" / "concepts__foo.md"
    assert archive_file.exists()

    result = runner.invoke(
        main,
        ["purge", "concepts/foo", "--yes", "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert not archive_file.exists()
    registry = Registry(project / "_registry")
    assert registry.get_page("concepts/foo") is None


def test_purge_refuses_unknown_page(project: Path) -> None:
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["purge", "concepts/ghost", "--yes", "--project", str(project)],
    )

    assert result.exit_code != 0
    assert "not found" in result.output
