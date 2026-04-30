"""Tests for wikiloom dormant signals + pipeable output on listing commands."""

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


# ----------------------------------------------------------------------
# Piped output — CliRunner stdout isn't a TTY, so `is_piped()` returns
# True and the listing commands emit the one-line-per-item format.
# ----------------------------------------------------------------------


def test_dormant_candidates_piped_is_tsv(project: Path) -> None:
    """Piped `wikiloom dormant` emits tab-separated one line per candidate."""
    old = _iso(datetime.now(timezone.utc) - timedelta(days=200))
    _seed(project, "concepts/old-one", "Old One", old, "# Old One\n\nBody.\n")
    _seed(project, "concepts/old-two", "Old Two", old, "# Old Two\n\nBody.\n")

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed")

    result = CliRunner().invoke(main, ["dormant", "--project", str(project)])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert len(lines) == 2, result.output
    for ln in lines:
        cols = ln.split("\t")
        assert len(cols) == 4, f"expected 4 TSV columns, got {cols!r}"
        assert cols[0].startswith("concepts/")
        assert cols[1] == "concept"
        assert cols[2].isdigit() and cols[3].isdigit()
    assert "Dormant candidates" not in result.output
    assert "Tip:" not in result.output


def test_dormant_list_marked_piped_is_tsv(project: Path) -> None:
    """Piped `wikiloom dormant --list-marked` emits tab-separated lines."""
    old = _iso(datetime.now(timezone.utc) - timedelta(days=200))
    _seed(project, "concepts/old-one", "Old One", old, "# Old One\n\nBody.\n")

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed")

    CliRunner().invoke(
        main, ["dormant", "concepts/old-one", "--project", str(project)]
    )

    result = CliRunner().invoke(
        main, ["dormant", "--list-marked", "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert len(lines) == 1, result.output
    cols = lines[0].split("\t")
    assert cols[0] == "concepts/old-one"
    assert cols[1] == "concept"
    assert cols[3] == "Old One"
    assert "Marked dormant" not in result.output


def test_orphans_piped_is_tsv(project: Path) -> None:
    """Piped `wikiloom orphans` emits tab-separated one line per orphan."""
    fresh = _iso(datetime.now(timezone.utc))
    _seed(project, "concepts/island", "Island With Spaces", fresh, "# Island\n\nBody.\n")

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed")

    result = CliRunner().invoke(main, ["orphans", "--project", str(project)])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert len(lines) == 1, result.output
    cols = lines[0].split("\t")
    assert cols == ["concepts/island", "concept", "Island With Spaces"]
    assert "Orphan pages" not in result.output


def test_log_piped_is_tsv_with_full_fields(project: Path) -> None:
    """Piped `wikiloom log` emits TSV including tokens, cost, commit columns."""
    log_path = project / "wiki" / "log.md"
    log_path.write_text(
        "# Event Log\n\n"
        "## [2026-04-01T10:00:00Z] ingest | ingested sample.md\n\n"
        "- **Tokens Used**: 1,234\n"
        "- **Cost**: $0.0567\n"
        "- **Commit**: abcdef1234567890\n\n"
        "## [2026-04-02T11:00:00Z] merge | concepts/a → concepts/b\n\n"
        "- **Commit**: 1234567890abcdef\n",
        encoding="utf-8",
    )
    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki")
    repo.index.commit("seed")

    result = CliRunner().invoke(main, ["log", "--project", str(project)])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert len(lines) == 2, result.output
    # Newest-first: merge event comes before ingest.
    merge_cols = lines[0].split("\t")
    ingest_cols = lines[1].split("\t")
    assert len(merge_cols) == 6 and len(ingest_cols) == 6
    # Description with spaces/arrow stays in column 3 thanks to tabs.
    assert merge_cols[1] == "merge"
    assert merge_cols[2] == "concepts/a → concepts/b"
    assert merge_cols[3] == "-"  # no tokens
    assert merge_cols[4] == "-"  # no cost
    assert merge_cols[5] == "12345678"
    assert ingest_cols[3] == "1234"
    assert ingest_cols[4] == "0.0567"
    assert ingest_cols[5] == "abcdef12"


def test_log_backfills_commit_hash_for_query_events(project: Path) -> None:
    """A query event written without a commit hash should pick up the
    matching commit's hash at display time.

    Query events land in ``log.md`` *before* their commit, so the
    write-side can't carry the hash. ``wikiloom log`` matches the
    commit subject (``query: <description>``) against recent git
    history and backfills the hash for display. Regression guard
    for the offset-naive vs. offset-aware datetime mismatch that
    previously crashed this path on real git output.
    """
    # Use a current timestamp so the event sits inside the lookup's
    # 5-minute window relative to the commit that lands a moment later.
    now_ts = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    log_path = project / "wiki" / "log.md"
    log_path.write_text(
        "# Event Log\n\n"
        f"## [{now_ts}] query | what is a chargeback?\n\n"
        "- **Tokens used**: 3,541\n"
        "- **Cost**: $0.0200\n",
        encoding="utf-8",
    )

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki")
    commit = repo.index.commit("query: what is a chargeback?")

    result = CliRunner().invoke(main, ["log", "--project", str(project)])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    # Pretty (non-piped) output renders the hash in parentheses.
    assert any(commit.hexsha[:8] in line for line in lines), result.output


def test_orphans_limit_caps_output(project: Path) -> None:
    """`wikiloom orphans -n 1` prints at most one orphan."""
    fresh = _iso(datetime.now(timezone.utc))
    _seed(project, "concepts/a", "A", fresh, "# A\n\nBody.\n")
    _seed(project, "concepts/b", "B", fresh, "# B\n\nBody.\n")
    _seed(project, "concepts/c", "C", fresh, "# C\n\nBody.\n")

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed")

    result = CliRunner().invoke(
        main, ["orphans", "-n", "1", "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert len(lines) == 1, result.output


def test_dormant_limit_caps_candidates(project: Path) -> None:
    """`wikiloom dormant -n 1` prints at most one candidate."""
    old = _iso(datetime.now(timezone.utc) - timedelta(days=200))
    _seed(project, "concepts/a", "A", old, "# A\n\nBody.\n")
    _seed(project, "concepts/b", "B", old, "# B\n\nBody.\n")

    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("seed")

    result = CliRunner().invoke(
        main, ["dormant", "-n", "1", "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert len(lines) == 1, result.output


def test_edits_piped_is_tsv(project: Path) -> None:
    """Piped `wikiloom edits` emits tab-separated one line per edit."""
    fresh = _iso(datetime.now(timezone.utc))
    _seed(project, "concepts/page", "Page", fresh, "# Page\n\nBody.\n")
    repo = git.Repo(project)
    repo.git.add("-A", "--", "wiki", "_registry")
    repo.index.commit("human-edit: seed with spaces in subject")

    result = CliRunner().invoke(main, ["edits", "--project", str(project)])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln]
    assert len(lines) == 1, result.output
    cols = lines[0].split("\t")
    assert len(cols) == 4
    assert cols[3] == "human-edit: seed with spaces in subject"
    assert "Recent human edits" not in result.output
