"""Tests for wikiloom dormant --review CLI signals."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import git
import pytest
from click.testing import CliRunner

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.cli import main
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


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _seed(
    project_root: Path,
    page_id: str,
    title: str,
    modified: str,
    body: str,
) -> None:
    wiki_dir = project_root / "wiki"
    registry = Registry(project_root / "_registry", wiki_dir=wiki_dir)
    entry = PageEntry(
        title=title,
        type=page_id.split("/", 1)[0].rstrip("s"),
        page_id=page_id,
        created=modified,
        modified=modified,
    )
    registry.register_page(page_id, entry)
    # register_page overwrites created/modified with now_iso(); restore
    # the caller-supplied timestamps so dormant candidate checks see the
    # seeded age.
    registry.pages[page_id].created = modified
    registry.pages[page_id].modified = modified
    registry.save()
    write_page(
        wiki_dir / f"{page_id}.md",
        Frontmatter(
            title=title,
            type=entry.type,
            created=modified,
            modified=modified,
            summary=f"Summary of {title}",
        ),
        body,
    )


def test_dormant_review_shows_inbound_signals(project: Path) -> None:
    """--review displays inbound count and top linked-from titles."""
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(days=200))
    fresh = _iso(now)

    _seed(project, "concepts/old-target", "Old Target", old, "# Old Target\n\nBody.\n")
    _seed(
        project,
        "concepts/linking-source",
        "Linking Source",
        fresh,
        "# Linking Source\n\nSee [[concepts/old-target]].\n",
    )

    bl = BacklinkRegistry(project / "_registry", wiki_dir=project / "wiki")
    bl.rebuild()
    bl.save()

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed")

    result = CliRunner().invoke(
        main,
        ["dormant", "--review", "--project", str(project)],
        input="q\n",
    )

    assert result.exit_code == 0, result.output
    assert re.search(r"inbound:\s+1\b", result.output), result.output
    assert re.search(r"linked from:\s+Linking Source", result.output), result.output


def test_dormant_review_shows_zero_inbound_without_linked_from(project: Path) -> None:
    """Candidates with no inbound links show 'inbound: 0' and no linked-from line."""
    old = _iso(datetime.now(timezone.utc) - timedelta(days=200))
    _seed(project, "concepts/lonely", "Lonely", old, "# Lonely\n\nBody.\n")

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed")

    result = CliRunner().invoke(
        main,
        ["dormant", "--review", "--project", str(project)],
        input="q\n",
    )

    assert result.exit_code == 0, result.output
    assert re.search(r"inbound:\s+0\b", result.output), result.output
    assert "linked from:" not in result.output
