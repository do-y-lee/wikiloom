"""End-to-end smoke tests for ``wikiloom.ingest.processor.ingest``.

These tests verify the write-side pipeline composes correctly:
extraction → raw copy → backlinks rebuild → manifest sync → index
regeneration → single atomic git commit. They deliberately do NOT
cover the LLM-dependent stages (synthesis, linking invocation, page
write) since those land with Component 20. When those stages are
wired in, this file is where regressions will be caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import git
import pytest

from wikiloom.ingest.processor import ingest
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
    # Baseline commit so HEAD is valid before any ingest.
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


@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text(
        "# Sample\n\nA short document mentioning nothing in particular.\n",
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# Smoke tests
# ----------------------------------------------------------------------


def test_ingest_writes_backlinks_file(project: Path, sample_markdown: Path) -> None:
    ingest(sample_markdown, project_root=project)
    backlinks = project / "_registry" / "backlinks.json"
    assert backlinks.exists()
    data = json.loads(backlinks.read_text())
    assert data["version"] == 1
    assert "links" in data


def test_ingest_copies_source_to_raw(project: Path, sample_markdown: Path) -> None:
    result = ingest(sample_markdown, project_root=project)
    assert result.raw_path is not None
    assert result.raw_path.exists()
    # Ends up under raw/articles/ for markdown (per RAW_DEST_BY_CONTENT_TYPE)
    assert "articles" in result.raw_path.parts


def test_ingest_regenerates_index_files(project: Path, sample_markdown: Path) -> None:
    root_index = project / "wiki" / "index.md"
    concepts_index = project / "wiki" / "concepts" / "index.md"
    before_root = root_index.read_text()

    ingest(sample_markdown, project_root=project)

    # Root index was rewritten in the IndexUpdater format
    after_root = root_index.read_text()
    assert "# Wiki Index" in after_root
    assert "## Concepts" in after_root
    # Sub-indexes now follow the table format (or empty placeholder)
    assert "Concepts Index" in concepts_index.read_text()


def test_ingest_produces_single_ingest_commit(
    project: Path, sample_markdown: Path
) -> None:
    repo = git.Repo(project)
    commits_before = len(list(repo.iter_commits()))

    ingest(sample_markdown, project_root=project)

    commits_after = len(list(repo.iter_commits()))
    assert commits_after == commits_before + 1

    head = repo.head.commit
    assert head.message.startswith("ingest: sample.md")


def test_ingest_commit_includes_backlinks_and_indexes(
    project: Path, sample_markdown: Path
) -> None:
    """The ingest commit should be atomic: raw copy + backlinks + indexes
    all land in one commit so history reflects a coherent state."""
    repo = git.Repo(project)
    ingest(sample_markdown, project_root=project)

    head = repo.head.commit
    # Diff the ingest commit against its parent — every touched file in
    # this run should appear in the delta.
    changed = {item.a_path or item.b_path for item in head.diff(head.parents[0])}
    assert any("raw/articles/sample.md" in p for p in changed)
    assert any("_registry/backlinks.json" in p for p in changed)
    assert any("wiki/index.md" == p for p in changed)


def test_ingest_is_reentrant_on_identical_source(
    project: Path, sample_markdown: Path
) -> None:
    """Re-ingesting the same file should not pollute git history with
    empty commits. Second run's commit hash should equal HEAD from
    before (no-op) since the pipeline produces identical derived state."""
    repo = git.Repo(project)
    ingest(sample_markdown, project_root=project)
    head_after_first = repo.head.commit.hexsha

    ingest(sample_markdown, project_root=project)
    head_after_second = repo.head.commit.hexsha

    assert head_after_first == head_after_second
