"""Tests for wikiloom.git_ops."""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from wikiloom.git_ops import CommitInfo, GitOps, _format_ingest_stats, _parse_commit_type


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Initialize an empty git repo with a committed README so HEAD is valid."""
    r = git.Repo.init(tmp_path)
    with r.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    readme = tmp_path / "README.md"
    readme.write_text("# test\n")
    r.index.add(["README.md"])
    r.index.commit("initial commit")
    return tmp_path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def test_parse_commit_type_extracts_prefix() -> None:
    assert _parse_commit_type("ingest: paper.pdf [+3 pages]") == "ingest"
    assert _parse_commit_type("human-edit: concepts/x.md [protected]") == "human-edit"
    assert _parse_commit_type("migration: v1 → v2") == "migration"


def test_parse_commit_type_returns_unknown_for_non_conforming() -> None:
    assert _parse_commit_type("just a message") == "unknown"
    assert _parse_commit_type("") == "unknown"


def test_format_ingest_stats_accepts_ints() -> None:
    stats = {"pages_created": 3, "pages_updated": 1, "links_inserted": 14}
    assert _format_ingest_stats(stats) == " [+3 pages, ~1 page, 14 links]"


def test_format_ingest_stats_accepts_lists() -> None:
    stats = {
        "pages_created": ["concepts/a", "concepts/b"],
        "pages_updated": ["entities/x"],
        "links_inserted": 5,
    }
    assert _format_ingest_stats(stats) == " [+2 pages, ~1 page, 5 links]"


def test_format_ingest_stats_empty_when_all_zero() -> None:
    assert _format_ingest_stats({}) == ""
    assert _format_ingest_stats({"pages_created": 0}) == ""


def test_format_ingest_stats_singular_plural() -> None:
    assert "+1 page," in _format_ingest_stats({"pages_created": 1, "pages_updated": 2})
    assert "~2 pages" in _format_ingest_stats({"pages_created": 1, "pages_updated": 2})


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_gitops_rejects_non_repo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a git repository"):
        GitOps(tmp_path)


def test_gitops_opens_existing_repo(repo: Path) -> None:
    ops = GitOps(repo)
    assert ops.repo_path == repo.resolve()


# ----------------------------------------------------------------------
# commit_ingest
# ----------------------------------------------------------------------


def test_commit_ingest_writes_conventional_subject(repo: Path) -> None:
    _write(repo / "wiki" / "concepts" / "attention.md", "body")
    ops = GitOps(repo)
    sha = ops.commit_ingest(
        "attention-is-all-you-need.pdf",
        [repo / "wiki" / "concepts" / "attention.md"],
        {"pages_created": 1, "links_inserted": 3},
    )
    commit = ops.repo.commit(sha)
    assert commit.message.startswith(
        "ingest: attention-is-all-you-need.pdf [+1 page, 3 links]"
    )


def test_commit_ingest_accepts_relative_paths(repo: Path) -> None:
    _write(repo / "wiki" / "sources" / "paper.md", "body")
    ops = GitOps(repo)
    sha = ops.commit_ingest(
        "paper.pdf",
        [Path("wiki/sources/paper.md")],
        {"pages_created": 1},
    )
    assert ops.repo.commit(sha).message.startswith("ingest: paper.pdf")


def test_commit_ingest_no_changes_returns_head_without_new_commit(repo: Path) -> None:
    ops = GitOps(repo)
    head_before = ops.repo.head.commit.hexsha
    sha = ops.commit_ingest("nothing.pdf", [], {})
    assert sha == head_before
    assert ops.repo.head.commit.hexsha == head_before


def test_commit_ingest_stages_multiple_files_in_one_commit(repo: Path) -> None:
    _write(repo / "wiki" / "concepts" / "a.md", "a")
    _write(repo / "wiki" / "concepts" / "b.md", "b")
    _write(repo / "_registry" / "manifest.json", "{}")
    ops = GitOps(repo)
    sha = ops.commit_ingest(
        "multi.pdf",
        [
            repo / "wiki" / "concepts" / "a.md",
            repo / "wiki" / "concepts" / "b.md",
            repo / "_registry" / "manifest.json",
        ],
        {"pages_created": 2},
    )
    commit = ops.repo.commit(sha)
    # All three files appear in the tree for this commit.
    tree_paths = {b.path for b in commit.tree.traverse() if b.type == "blob"}
    assert "wiki/concepts/a.md" in tree_paths
    assert "wiki/concepts/b.md" in tree_paths
    assert "_registry/manifest.json" in tree_paths


# ----------------------------------------------------------------------
# commit_human_edit
# ----------------------------------------------------------------------


def test_commit_human_edit_uses_human_edit_prefix(repo: Path) -> None:
    page = repo / "wiki" / "concepts" / "transformer.md"
    _write(page, "hand-written content")
    ops = GitOps(repo)
    sha = ops.commit_human_edit(page)
    commit = ops.repo.commit(sha)
    assert commit.message.startswith(
        "human-edit: wiki/concepts/transformer.md [protected]"
    )


# ----------------------------------------------------------------------
# History queries
# ----------------------------------------------------------------------


def test_get_file_history_returns_newest_first(repo: Path) -> None:
    page = repo / "wiki" / "concepts" / "x.md"
    ops = GitOps(repo)

    _write(page, "v1")
    ops.commit_ingest("src.pdf", [page], {"pages_created": 1})

    _write(page, "v2")
    ops.commit_human_edit(page)

    history = ops.get_file_history(page)
    assert len(history) == 2
    assert history[0].commit_type == "human-edit"
    assert history[1].commit_type == "ingest"


def test_get_file_history_returns_commit_info_fields(repo: Path) -> None:
    page = repo / "wiki" / "concepts" / "x.md"
    _write(page, "v1")
    ops = GitOps(repo)
    ops.commit_ingest("src.pdf", [page], {"pages_created": 1})

    history = ops.get_file_history(page)
    info = history[0]
    assert isinstance(info, CommitInfo)
    assert info.hash
    assert info.author == "Test"
    assert info.timestamp  # ISO string
    assert info.commit_type == "ingest"


def test_get_file_history_empty_for_untracked(repo: Path) -> None:
    ops = GitOps(repo)
    assert ops.get_file_history(repo / "wiki" / "nope.md") == []


# ----------------------------------------------------------------------
# is_human_edited
# ----------------------------------------------------------------------


def test_is_human_edited_false_for_ingest_only(repo: Path) -> None:
    page = repo / "wiki" / "concepts" / "x.md"
    _write(page, "v1")
    ops = GitOps(repo)
    ops.commit_ingest("src.pdf", [page], {"pages_created": 1})
    assert ops.is_human_edited(page) is False


def test_is_human_edited_true_after_human_commit(repo: Path) -> None:
    page = repo / "wiki" / "concepts" / "x.md"
    ops = GitOps(repo)

    _write(page, "v1")
    ops.commit_ingest("src.pdf", [page], {"pages_created": 1})

    _write(page, "hand-edited")
    ops.commit_human_edit(page)

    assert ops.is_human_edited(page) is True


def test_is_human_edited_false_when_llm_rewrites_after_human(repo: Path) -> None:
    """A later ingest over a human-edited page flips the flag back.

    The linter can then decide whether to respect the override or warn.
    """
    page = repo / "wiki" / "concepts" / "x.md"
    ops = GitOps(repo)

    _write(page, "v1")
    ops.commit_ingest("src.pdf", [page], {"pages_created": 1})

    _write(page, "human")
    ops.commit_human_edit(page)

    _write(page, "llm rewrite")
    ops.commit_ingest("src2.pdf", [page], {"pages_updated": 1})

    assert ops.is_human_edited(page) is False


def test_is_human_edited_false_for_untracked(repo: Path) -> None:
    ops = GitOps(repo)
    assert ops.is_human_edited(repo / "never-existed.md") is False


def test_is_human_edited_true_for_plain_commit_message(repo: Path) -> None:
    """Commits with a subject that isn't an AUTO prefix count as human.

    This covers the case where a user edits a file in their editor and
    runs ``git commit -m "fix typo"`` — no ``human-edit:`` prefix, but
    it's still a human edit.
    """
    page = repo / "wiki" / "concepts" / "x.md"
    _write(page, "body")
    r = git.Repo(repo)
    r.index.add(["wiki/concepts/x.md"])
    r.index.commit("fix typo")

    ops = GitOps(repo)
    assert ops.is_human_edited(page) is True


def test_latest_commit_type_returns_parsed_prefix(repo: Path) -> None:
    page = repo / "wiki" / "concepts" / "x.md"
    _write(page, "body")
    ops = GitOps(repo)
    ops.commit_ingest("src.pdf", [page], {"pages_created": 1})
    assert ops.latest_commit_type(page) == "ingest"


# ----------------------------------------------------------------------
# latest_commit_types_bulk
# ----------------------------------------------------------------------


def test_latest_commit_types_bulk_empty_input(repo: Path) -> None:
    ops = GitOps(repo)
    assert ops.latest_commit_types_bulk([]) == {}


def test_latest_commit_types_bulk_returns_none_for_untouched(repo: Path) -> None:
    ops = GitOps(repo)
    ghost = repo / "wiki" / "concepts" / "never-committed.md"
    result = ops.latest_commit_types_bulk([ghost])
    assert result == {ghost: None}


def test_latest_commit_types_bulk_matches_per_page(repo: Path) -> None:
    """The bulk walk must agree with N per-page calls. This is the
    contract the scan() rewrite relies on — same answer, fewer walks."""
    a = repo / "wiki" / "concepts" / "a.md"
    b = repo / "wiki" / "concepts" / "b.md"
    c = repo / "wiki" / "concepts" / "c.md"
    ops = GitOps(repo)

    _write(a, "a-v1")
    ops.commit_ingest("a.pdf", [a], {"pages_created": 1})
    _write(b, "b-v1")
    ops.commit_ingest("b.pdf", [b], {"pages_created": 1})
    _write(c, "c-v1")
    r = git.Repo(repo)
    r.index.add(["wiki/concepts/c.md"])
    r.index.commit("hand-written note")
    # Touch a again with a human edit so its newest commit-type flips.
    _write(a, "a-v2")
    ops.commit_human_edit(a)

    bulk = ops.latest_commit_types_bulk([a, b, c])
    expected = {p: ops.latest_commit_type(p) for p in (a, b, c)}
    assert bulk == expected
    # Sanity-check the actual values, not just the agreement with the
    # per-page version — protects against both implementations being
    # broken in the same way.
    assert bulk[a] == "human-edit"
    assert bulk[b] == "ingest"
    assert bulk[c] == "unknown"


def test_latest_commit_types_bulk_terminates_after_all_paths_resolved(
    repo: Path,
) -> None:
    """Once every requested path has an answer, the walk must stop —
    otherwise the perf win evaporates on deep histories. We can't
    observe iter_commits internals, but we can confirm the result is
    stable across an arbitrarily-long unrelated tail of commits."""
    page = repo / "wiki" / "concepts" / "x.md"
    _write(page, "v1")
    ops = GitOps(repo)
    ops.commit_ingest("src.pdf", [page], {"pages_created": 1})
    # Pile up unrelated commits AFTER the page's last touch. They
    # shouldn't affect the answer for `page` but they extend history.
    other = repo / "README.md"
    r = git.Repo(repo)
    for i in range(10):
        other.write_text(f"# test {i}\n", encoding="utf-8")
        r.index.add(["README.md"])
        r.index.commit(f"chore: bump {i}")

    result = ops.latest_commit_types_bulk([page])
    assert result == {page: "ingest"}


def test_latest_commit_types_bulk_handles_missing_head(tmp_path: Path) -> None:
    """A fresh repo with no commits has an invalid HEAD. The helper
    must return None for every path rather than crashing."""
    r = git.Repo.init(tmp_path)
    with r.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    ops = GitOps(tmp_path)
    page = tmp_path / "wiki" / "concepts" / "x.md"
    assert ops.latest_commit_types_bulk([page]) == {page: None}


# ----------------------------------------------------------------------
# get_changed_files_since
# ----------------------------------------------------------------------


def test_get_changed_files_since_lists_changes(repo: Path) -> None:
    ops = GitOps(repo)
    base = ops.repo.head.commit.hexsha

    _write(repo / "wiki" / "concepts" / "a.md", "a")
    _write(repo / "wiki" / "concepts" / "b.md", "b")
    ops.commit_ingest("p.pdf", [
        repo / "wiki" / "concepts" / "a.md",
        repo / "wiki" / "concepts" / "b.md",
    ], {"pages_created": 2})

    changed = ops.get_changed_files_since(base)
    names = {p.name for p in changed}
    assert names == {"a.md", "b.md"}
    # Paths are absolute
    assert all(p.is_absolute() for p in changed)


def test_get_changed_files_since_empty_when_no_changes(repo: Path) -> None:
    ops = GitOps(repo)
    base = ops.repo.head.commit.hexsha
    assert ops.get_changed_files_since(base) == []


def test_get_changed_files_since_raises_for_unknown_commit(repo: Path) -> None:
    ops = GitOps(repo)
    with pytest.raises(ValueError, match="Unknown commit"):
        ops.get_changed_files_since("deadbeef" * 5)
