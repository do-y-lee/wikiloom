"""Regression tests for the post-command clean-tree invariant.

State-changing CLI commands (ingest, lint --fix, reindex, relink, purge,
related, rebuild-cache, merge) append a log event to ``wiki/log.md``
*after* the primary commit so the event can carry that commit's hash.
``_commit_log_tail`` (cli.py) lands the resulting log.md change in a
small follow-up commit so the working tree stays clean.

These tests pin that invariant for the cheap, embedder-free commands.
The ingest path is covered by
``test_ingest_commit_includes_backlinks_and_indexes`` in
test_ingest_pipeline.py; the merge path is covered indirectly by the
existing merge tests.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest
from click.testing import CliRunner

from wikiloom.cli import _commit_log_tail, main
from wikiloom.frontmatter import Frontmatter, write_page
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = init_project(name="testproj", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    return project_dir


def _add_active_page(project_root: Path, page_id: str) -> None:
    wiki_dir = project_root / "wiki"
    registry = Registry(project_root / "_registry", wiki_dir=wiki_dir)
    entry = PageEntry(
        title=page_id.split("/")[-1].title(),
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
            title=entry.title,
            type=entry.type,
            created=entry.created,
            modified=entry.modified,
            summary=f"Summary of {entry.title}",
        ),
        f"# {entry.title}\n\nBody.\n",
    )
    repo = git.Repo(project_root)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed: add page")


def test_purge_leaves_clean_tree(project: Path) -> None:
    """`wikiloom purge` commits its PURGE log event in a follow-up so
    log.md is not left dirty."""
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()
    runner.invoke(
        main,
        ["deprecate", "concepts/foo", "--yes", "--project", str(project)],
    )

    result = runner.invoke(
        main,
        ["purge", "concepts/foo", "--yes", "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    repo = git.Repo(project)
    assert repo.is_dirty(untracked_files=True) is False, (
        f"working tree dirty after purge:\n{repo.git.status()}"
    )


def test_reindex_leaves_clean_tree(project: Path) -> None:
    """`wikiloom reindex` commits its REINDEX log event in a follow-up
    so log.md is not left dirty."""
    _add_active_page(project, "concepts/foo")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["reindex", "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    repo = git.Repo(project)
    assert repo.is_dirty(untracked_files=True) is False, (
        f"working tree dirty after reindex:\n{repo.git.status()}"
    )


def test_commit_log_tail_commits_log_md(project: Path) -> None:
    """The helper stages and commits log.md when dirty, leaving the
    rest of the working tree alone."""
    _add_active_page(project, "concepts/foo")
    log_path = project / "wiki" / "log.md"
    log_path.write_text("# log\n\n## entry\n", encoding="utf-8")
    repo = git.Repo(project)
    assert repo.is_dirty(untracked_files=True)

    _commit_log_tail(project, "test: commit log tail")

    assert repo.is_dirty(untracked_files=True) is False
    assert repo.head.commit.message.startswith("test: commit log tail")


def test_commit_log_tail_noops_when_log_missing(tmp_path: Path) -> None:
    """No log.md → no commit attempt, no exception."""
    repo_path = tmp_path / "empty"
    repo_path.mkdir()
    repo = git.Repo.init(repo_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    # Seed the repo so HEAD exists; otherwise GitOps init may differ.
    (repo_path / "README.md").write_text("seed\n", encoding="utf-8")
    repo.git.add("README.md")
    repo.index.commit("seed")

    _commit_log_tail(repo_path, "test: noop")

    assert repo.head.commit.message.strip() == "seed"
