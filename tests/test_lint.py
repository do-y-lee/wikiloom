"""Tests for wikiloom.lint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.config import StalenessConfig
from wikiloom.frontmatter import Frontmatter, render_frontmatter
from wikiloom.lint import (
    BrokenLink,
    DuplicateSet,
    LintReport,
    StalePage,
    WikiLinter,
)
from wikiloom.registry import PageEntry, Registry
from wikiloom.utils import now_iso


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
        # Empty sub-index matches what `wikiloom init` creates so the
        # index-consistency check sees aligned state by default.
        (sub_dir / "index.md").write_text("# " + sub.title() + "\n", encoding="utf-8")
    return tmp_path


def _write_page(
    project: Path,
    rel: str,
    body: str = "body",
    *,
    title: str | None = None,
    type_: str | None = None,
    status: str = "active",
    modified: str | None = None,
    contradictions: list[dict] | None = None,
) -> Path:
    path = project / "wiki" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = Frontmatter(
        title=title or path.stem.replace("-", " ").title(),
        type=type_ or path.parent.name.rstrip("s") or "concept",
        status=status,
        created=now_iso(),
        modified=modified or now_iso(),
        summary="summary",
        contradictions=contradictions or [],
    )
    path.write_text(render_frontmatter(fm) + "\n" + body, encoding="utf-8")
    return path


def _register(
    project: Path,
    page_id: str,
    *,
    type_: str = "concept",
    title: str | None = None,
    aliases: list[str] | None = None,
    status: str = "active",
    modified: str | None = None,
) -> Registry:
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        page_id,
        PageEntry(
            title=title or page_id.split("/")[-1].replace("-", " ").title(),
            type=type_,
            status=status,
            aliases=aliases or [],
            modified=modified or now_iso(),
            summary="summary",
        ),
    )
    reg.save()
    return reg


def _rebuild_backlinks(project: Path) -> None:
    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()
    reg.save()


def _iso_days_ago(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# check_broken_links
# ----------------------------------------------------------------------


def test_check_broken_links_flags_missing_target(project: Path) -> None:
    _write_page(project, "concepts/a.md", body="refers to [[concepts/ghost]]")
    _register(project, "concepts/a")
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    broken = linter.check_broken_links()
    assert len(broken) == 1
    assert broken[0].target == "concepts/ghost"
    assert broken[0].source == "concepts/a"


def test_check_broken_links_clean_when_all_targets_exist(project: Path) -> None:
    _write_page(project, "concepts/a.md", body="[[concepts/b]]")
    _write_page(project, "concepts/b.md", body="body")
    reg = _register(project, "concepts/a")
    reg.register_page("concepts/b", PageEntry(title="B", type="concept"))
    reg.save()
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    assert linter.check_broken_links() == []


# ----------------------------------------------------------------------
# check_orphans
# ----------------------------------------------------------------------


def test_check_orphans_flags_page_with_no_inbound(project: Path) -> None:
    _write_page(project, "concepts/a.md", body="[[concepts/b]]")
    _write_page(project, "concepts/b.md")
    reg = _register(project, "concepts/a")
    reg.register_page("concepts/b", PageEntry(title="B", type="concept"))
    reg.save()
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    orphans = linter.check_orphans()
    assert "concepts/a" in orphans
    assert "concepts/b" not in orphans


def test_check_orphans_includes_pages_absent_from_backlinks(project: Path) -> None:
    # Page exists in manifest but has no links in or out
    _write_page(project, "concepts/lonely.md")
    _register(project, "concepts/lonely")
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    assert "concepts/lonely" in linter.check_orphans()


def test_check_orphans_skips_source_pages(project: Path) -> None:
    _write_page(project, "sources/paper.md", type_="source")
    _register(project, "sources/paper", type_="source")
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    assert "sources/paper" not in linter.check_orphans()


# ----------------------------------------------------------------------
# check_staleness
# ----------------------------------------------------------------------


def test_check_staleness_flags_old_concept(project: Path) -> None:
    old = _iso_days_ago(200)
    _write_page(project, "concepts/old.md", modified=old)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "concepts/old",
        PageEntry(title="Old", type="concept", modified=old, staleness_window_days=0),
    )
    # Clobber modified which register_page auto-refreshes
    reg.pages["concepts/old"].modified = old
    reg.pages["concepts/old"].staleness_window_days = 120
    reg.save()

    linter = WikiLinter(project, staleness=StalenessConfig(concept_window_days=120))
    stale = linter.check_staleness()
    assert len(stale) == 1
    assert stale[0].page_id == "concepts/old"
    assert stale[0].age_days >= 120


def test_check_staleness_uses_per_type_window(project: Path) -> None:
    recent = _iso_days_ago(30)
    _write_page(project, "syntheses/recent.md", type_="synthesis", modified=recent)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "syntheses/recent",
        PageEntry(title="Recent", type="synthesis"),
    )
    reg.pages["syntheses/recent"].modified = recent
    reg.pages["syntheses/recent"].staleness_window_days = 0  # force fallback
    reg.save()

    # Synthesis window is 60 days by default — 30-day page is fresh
    linter = WikiLinter(project)
    assert linter.check_staleness() == []


def test_check_staleness_skips_deprecated(project: Path) -> None:
    old = _iso_days_ago(500)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "concepts/old",
        PageEntry(title="Old", type="concept", status="deprecated"),
    )
    reg.pages["concepts/old"].modified = old
    reg.save()

    linter = WikiLinter(project)
    assert linter.check_staleness() == []


# ----------------------------------------------------------------------
# check_duplicates
# ----------------------------------------------------------------------


def test_check_duplicates_flags_near_identical_titles(project: Path) -> None:
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "concepts/flash-attention",
        PageEntry(title="Flash Attention", type="concept"),
    )
    reg.register_page(
        "concepts/flash-attentions",
        PageEntry(title="Flash Attentions", type="concept"),
    )
    reg.save()

    linter = WikiLinter(project)
    dups = linter.check_duplicates()
    assert len(dups) == 1
    assert set(dups[0].pages) == {"concepts/flash-attention", "concepts/flash-attentions"}
    assert dups[0].reason == "title"


def test_check_duplicates_flags_alias_overlap(project: Path) -> None:
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "entities/openai",
        PageEntry(title="OpenAI", type="entity", aliases=["open ai"]),
    )
    reg.register_page(
        "entities/open-ai",
        PageEntry(title="Open-AI", type="entity", aliases=["open ai inc"]),
    )
    reg.save()

    linter = WikiLinter(project)
    dups = linter.check_duplicates()
    assert len(dups) == 1


def test_check_duplicates_ignores_cross_type(project: Path) -> None:
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page("entities/transformer", PageEntry(title="Transformer", type="entity"))
    reg.register_page("concepts/transformer", PageEntry(title="Transformer", type="concept"))
    reg.save()

    linter = WikiLinter(project)
    assert linter.check_duplicates() == []


# ----------------------------------------------------------------------
# check_frontmatter
# ----------------------------------------------------------------------


def test_check_frontmatter_flags_missing_file(project: Path) -> None:
    (project / "wiki" / "concepts" / "broken.md").write_text("no frontmatter\n")
    linter = WikiLinter(project)
    issues = linter.check_frontmatter()
    assert "concepts/broken" in issues


def test_check_frontmatter_flags_missing_required_field(project: Path) -> None:
    path = project / "wiki" / "concepts" / "partial.md"
    fm = Frontmatter(title="", type="concept", status="active", created="x", modified="x", summary="s")
    path.write_text(render_frontmatter(fm) + "\nbody", encoding="utf-8")

    linter = WikiLinter(project)
    assert "concepts/partial" in linter.check_frontmatter()


def test_check_frontmatter_clean_page_has_no_issues(project: Path) -> None:
    _write_page(project, "concepts/good.md")
    linter = WikiLinter(project)
    assert linter.check_frontmatter() == []


# ----------------------------------------------------------------------
# check_index_consistency
# ----------------------------------------------------------------------


def test_check_index_consistency_flags_drift(project: Path) -> None:
    _write_page(project, "concepts/a.md")
    _write_page(project, "concepts/b.md")
    # Sub-index table only lists "a", not "b"
    (project / "wiki" / "concepts" / "index.md").write_text(
        "# Concepts\n\n| Page | Summary |\n|------|---------|\n| [a](a.md) | s |\n"
    )
    linter = WikiLinter(project)
    assert "concepts" in linter.check_index_consistency()


def test_check_index_consistency_clean_when_aligned(project: Path) -> None:
    _write_page(project, "concepts/a.md")
    (project / "wiki" / "concepts" / "index.md").write_text(
        "# Concepts\n\n| Page | Summary |\n|------|---------|\n| [a](a.md) | s |\n"
    )
    linter = WikiLinter(project)
    assert "concepts" not in linter.check_index_consistency()


def test_check_index_consistency_flags_missing_index(project: Path) -> None:
    _write_page(project, "concepts/a.md")
    # No index.md
    linter = WikiLinter(project)
    assert "concepts" in linter.check_index_consistency()


# ----------------------------------------------------------------------
# check_contradictions
# ----------------------------------------------------------------------


def test_check_contradictions_empty_by_default(project: Path) -> None:
    _write_page(project, "concepts/a.md")
    linter = WikiLinter(project)
    assert linter.check_contradictions() == []


def test_check_contradictions_reads_frontmatter_field(project: Path) -> None:
    _write_page(
        project,
        "concepts/a.md",
        contradictions=[{"existing": "x", "new": "y", "source": "paper"}],
    )
    linter = WikiLinter(project)
    found = linter.check_contradictions()
    assert len(found) == 1
    assert found[0].existing == "x"
    assert found[0].new == "y"


# ----------------------------------------------------------------------
# check_stubs
# ----------------------------------------------------------------------


def test_check_stubs_returns_stub_pages(project: Path) -> None:
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page("concepts/stub1", PageEntry(title="Stub1", type="concept", status="stub"))
    reg.register_page("concepts/real", PageEntry(title="Real", type="concept"))
    reg.save()

    linter = WikiLinter(project)
    stubs = linter.check_stubs()
    assert stubs == ["concepts/stub1"]


# ----------------------------------------------------------------------
# run_all
# ----------------------------------------------------------------------


def test_run_all_aggregates_findings(project: Path) -> None:
    _write_page(project, "concepts/a.md", body="[[concepts/ghost]]")
    _register(project, "concepts/a")
    _rebuild_backlinks(project)

    report = WikiLinter(project).run_all()
    assert isinstance(report, LintReport)
    assert len(report.broken_links) == 1
    assert report.total_issues >= 1
    assert report.is_healthy is False


def test_run_all_healthy_on_clean_project(project: Path) -> None:
    # Nothing in the wiki at all
    _rebuild_backlinks(project)
    report = WikiLinter(project).run_all()
    assert report.is_healthy


# ----------------------------------------------------------------------
# fix_all
# ----------------------------------------------------------------------


def test_fix_all_strips_broken_wikilinks(project: Path) -> None:
    page = _write_page(
        project,
        "concepts/a.md",
        body="This refers to [[concepts/ghost|the ghost]] in the machine.",
    )
    _register(project, "concepts/a")
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    report = linter.run_all()
    fixes = linter.fix_all(report)

    assert fixes.broken_links_fixed == 1
    content = page.read_text(encoding="utf-8")
    assert "[[concepts/ghost" not in content
    assert "the ghost" in content


def test_fix_all_marks_stale_pages(project: Path) -> None:
    old = _iso_days_ago(500)
    page = _write_page(project, "concepts/old.md", modified=old)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page("concepts/old", PageEntry(title="Old", type="concept"))
    reg.pages["concepts/old"].modified = old
    reg.save()

    linter = WikiLinter(project)
    report = linter.run_all()
    fixes = linter.fix_all(report)

    assert fixes.stale_marked >= 1
    assert "status: stale" in page.read_text(encoding="utf-8")


def test_fix_all_repairs_missing_frontmatter(project: Path) -> None:
    path = project / "wiki" / "concepts" / "bare.md"
    path.write_text("just a body, no frontmatter\n", encoding="utf-8")

    linter = WikiLinter(project)
    report = linter.run_all()
    fixes = linter.fix_all(report)

    assert fixes.frontmatter_repaired == 1
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "title:" in text


def test_fix_all_skips_human_edited_pages(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _write_page(project, "concepts/a.md", body="[[concepts/ghost]]")
    _register(project, "concepts/a")
    _rebuild_backlinks(project)

    linter = WikiLinter(project)

    class FakeGit:
        def is_human_edited(self, p: Path) -> bool:
            return p == page

    linter.git = FakeGit()  # type: ignore[assignment]
    report = linter.run_all()
    fixes = linter.fix_all(report)

    assert fixes.broken_links_fixed == 0
    assert fixes.skipped_human_edited >= 1
    assert "[[concepts/ghost]]" in page.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# LintReport
# ----------------------------------------------------------------------


def test_lint_report_total_issues_sums_all_categories() -> None:
    report = LintReport(
        broken_links=[BrokenLink("a", "b", "")],
        orphans=["c"],
        stale=[StalePage("d", 100, 90)],
        duplicates=[DuplicateSet(pages=("e", "f"), reason="title", score=95)],
        frontmatter_issues=["g"],
        index_drift=["concepts"],
        contradictions=[],
        stubs=["h"],
    )
    assert report.total_issues == 7
    assert report.is_healthy is False


def test_lint_report_empty_is_healthy() -> None:
    assert LintReport().is_healthy
