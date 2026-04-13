"""Tests for wikiloom.protection."""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from wikiloom.frontmatter import Frontmatter, parse_frontmatter, render_frontmatter
from wikiloom.git_ops import GitOps
from wikiloom.protection import AUTO_MARKER, HumanEditProtection
from wikiloom.registry import PageEntry, Registry


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A real git-initialized project with one concepts subdir."""
    r = git.Repo.init(tmp_path)
    with r.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    (tmp_path / "_registry").mkdir()
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    (tmp_path / "wiki" / "log.md").write_text("# Event Log\n\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# test\n")
    r.index.add(["README.md"])
    r.index.commit("initial")
    return tmp_path


def _write_page(
    project: Path,
    rel: str,
    body: str,
    *,
    human_edited: bool = False,
) -> Path:
    path = project / "wiki" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = Frontmatter(
        title=path.stem.replace("-", " ").title(),
        type="concept",
        status="active",
        created="2026-04-01T00:00:00Z",
        modified="2026-04-01T00:00:00Z",
        summary="summary",
        human_edited=human_edited,
    )
    path.write_text(render_frontmatter(fm) + "\n" + body, encoding="utf-8")
    return path


def _register(project: Path, page_id: str, **kwargs) -> Registry:
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        page_id,
        PageEntry(
            title=page_id.split("/")[-1].replace("-", " ").title(),
            type="concept",
            **kwargs,
        ),
    )
    reg.save()
    return reg


# ----------------------------------------------------------------------
# split / merge
# ----------------------------------------------------------------------


def test_split_without_marker_returns_all_human() -> None:
    human, auto = HumanEditProtection.split("just a plain body\n")
    assert human == "just a plain body\n"
    assert auto == ""


def test_split_with_marker_returns_both_regions() -> None:
    body = f"human part\n\n{AUTO_MARKER}\n\nauto part\n"
    human, auto = HumanEditProtection.split(body)
    assert "human part" in human
    assert "auto part" in auto
    assert AUTO_MARKER not in human
    assert AUTO_MARKER not in auto


def test_merge_adds_marker_between_regions() -> None:
    merged = HumanEditProtection.merge("human\n", "auto\n")
    assert AUTO_MARKER in merged
    assert merged.index("human") < merged.index(AUTO_MARKER) < merged.index("auto")


def test_merge_omits_auto_section_if_empty() -> None:
    merged = HumanEditProtection.merge("just human\n", "")
    assert AUTO_MARKER in merged
    assert merged.endswith(AUTO_MARKER + "\n")


def test_split_merge_roundtrip() -> None:
    original_human = "human region\nmore human\n"
    original_auto = "auto region\nmore auto\n"
    merged = HumanEditProtection.merge(original_human, original_auto)
    human, auto = HumanEditProtection.split(merged)
    assert "human region" in human
    assert "auto region" in auto


# ----------------------------------------------------------------------
# preserve_human
# ----------------------------------------------------------------------


def test_preserve_human_passthrough_for_new_page(project: Path) -> None:
    pp = HumanEditProtection(project)
    out = pp.preserve_human(
        project / "wiki" / "concepts" / "new.md",
        "auto body\n",
    )
    assert out == "auto body\n"


def test_preserve_human_wraps_existing_unmarked_body(project: Path) -> None:
    page = _write_page(
        project, "concepts/a.md", "hand-written existing body\n"
    )
    _register(project, "concepts/a")

    pp = HumanEditProtection(project)
    out = pp.preserve_human(page, "new auto body\n")

    assert "hand-written existing body" in out
    assert "new auto body" in out
    assert AUTO_MARKER in out
    assert out.index("hand-written") < out.index(AUTO_MARKER)


def test_preserve_human_swaps_auto_region_only(project: Path) -> None:
    body = f"human region\n\n{AUTO_MARKER}\n\nold auto\n"
    page = _write_page(project, "concepts/a.md", body)
    _register(project, "concepts/a")

    pp = HumanEditProtection(project)
    out = pp.preserve_human(page, "brand new auto\n")

    assert "human region" in out
    assert "brand new auto" in out
    assert "old auto" not in out


# ----------------------------------------------------------------------
# is_protected (integration with git_ops)
# ----------------------------------------------------------------------


def test_is_protected_true_after_plain_commit(project: Path) -> None:
    """A page committed with a plain message (not an AUTO prefix) is
    treated as human-edited. This is the main C10 semantic widening."""
    page = _write_page(project, "concepts/a.md", "body\n")
    _register(project, "concepts/a")
    repo = git.Repo(project)
    repo.index.add([str(page.relative_to(project))])
    repo.index.commit("quick fix to a.md")  # no wikiloom prefix

    pp = HumanEditProtection(project)
    assert pp.is_protected("concepts/a") is True


def test_is_protected_false_after_ingest_commit(project: Path) -> None:
    page = _write_page(project, "concepts/a.md", "body\n")
    _register(project, "concepts/a")
    GitOps(project).commit_ingest("src.pdf", [page], {"pages_created": 1})

    pp = HumanEditProtection(project)
    assert pp.is_protected("concepts/a") is False


def test_is_protected_false_for_unknown_page(project: Path) -> None:
    pp = HumanEditProtection(project)
    assert pp.is_protected("concepts/missing") is False


# ----------------------------------------------------------------------
# scan / sync
# ----------------------------------------------------------------------


def test_scan_finds_drift_between_manifest_and_git(project: Path) -> None:
    page = _write_page(project, "concepts/a.md", "body\n", human_edited=False)
    reg = _register(project, "concepts/a", human_edited=False)
    repo = git.Repo(project)
    repo.index.add([str(page.relative_to(project))])
    repo.index.commit("hand-edited fix")

    pp = HumanEditProtection(project, registry=reg)
    drifted = pp.scan()
    assert len(drifted) == 1
    assert drifted[0].page_id == "concepts/a"
    assert drifted[0].git_says is True
    assert drifted[0].manifest_says is False


def test_scan_returns_empty_when_in_sync(project: Path) -> None:
    page = _write_page(project, "concepts/a.md", "body\n")
    reg = _register(project, "concepts/a")
    GitOps(project).commit_ingest("src.pdf", [page], {"pages_created": 1})

    pp = HumanEditProtection(project, registry=reg)
    assert pp.scan() == []


def test_sync_updates_manifest_flag(project: Path) -> None:
    page = _write_page(project, "concepts/a.md", "body\n")
    reg = _register(project, "concepts/a", human_edited=False)
    repo = git.Repo(project)
    repo.index.add([str(page.relative_to(project))])
    repo.index.commit("user fix")

    pp = HumanEditProtection(project, registry=reg)
    pp.sync()

    reloaded = Registry(project / "_registry", project / "wiki")
    assert reloaded.get_page("concepts/a").human_edited is True


def test_sync_updates_frontmatter_flag_and_timestamp(project: Path) -> None:
    page = _write_page(project, "concepts/a.md", "body\n")
    reg = _register(project, "concepts/a", human_edited=False)
    repo = git.Repo(project)
    repo.index.add([str(page.relative_to(project))])
    repo.index.commit("user fix")

    pp = HumanEditProtection(project, registry=reg)
    pp.sync()

    fm, _ = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert fm is not None
    assert fm.human_edited is True
    assert fm.human_edited_at  # ISO timestamp from the commit


def test_sync_emits_human_edit_event(project: Path) -> None:
    page = _write_page(project, "concepts/a.md", "body\n")
    reg = _register(project, "concepts/a", human_edited=False)
    repo = git.Repo(project)
    repo.index.add([str(page.relative_to(project))])
    repo.index.commit("fix")

    pp = HumanEditProtection(project, registry=reg)
    pp.sync()

    log_text = (project / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "human-edit" in log_text
    assert "concepts/a" in log_text


def test_sync_no_op_when_nothing_drifted(project: Path) -> None:
    page = _write_page(project, "concepts/a.md", "body\n")
    reg = _register(project, "concepts/a")
    GitOps(project).commit_ingest("src.pdf", [page], {"pages_created": 1})

    pp = HumanEditProtection(project, registry=reg)
    result = pp.sync()
    assert result == []
    # Log file should not have a HUMAN_EDIT entry
    log_text = (project / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "human-edit" not in log_text


def test_sync_flips_back_when_ingest_overwrites_human(project: Path) -> None:
    """If a human edit is later overwritten by an ingest commit, sync
    flips the flag back to False."""
    page = _write_page(project, "concepts/a.md", "body\n", human_edited=True)
    reg = _register(project, "concepts/a", human_edited=True)
    GitOps(project).commit_ingest("src.pdf", [page], {"pages_created": 1})

    pp = HumanEditProtection(project, registry=reg)
    pp.sync()

    reloaded = Registry(project / "_registry", project / "wiki")
    assert reloaded.get_page("concepts/a").human_edited is False
