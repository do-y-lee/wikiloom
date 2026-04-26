"""Tests for wikiloom.lint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.config import DormantConfig
from wikiloom.frontmatter import Frontmatter, parse_frontmatter, render_frontmatter
from wikiloom.lint import (
    BrokenLink,
    DormantPage,
    DuplicateSet,
    LintReport,
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


def test_check_broken_links_flags_deprecated_target(project: Path) -> None:
    _write_page(project, "concepts/a.md", body="[[concepts/old]]")
    _write_page(project, "concepts/old.md", status="deprecated")
    reg = _register(project, "concepts/a")
    reg.register_page(
        "concepts/old",
        PageEntry(title="Old", type="concept", status="deprecated"),
    )
    reg.save()
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    broken = linter.check_broken_links()
    assert len(broken) == 1
    assert broken[0].target == "concepts/old"
    assert broken[0].reason == "deprecated"


def test_check_broken_links_ignores_stub_targets(project: Path) -> None:
    """Stubs are placeholders — links to them should stay until the stub
    is filled in. check_stubs tracks them separately."""
    _write_page(project, "concepts/a.md", body="[[concepts/stubby]]")
    _write_page(project, "concepts/stubby.md", status="stub")
    reg = _register(project, "concepts/a")
    reg.register_page(
        "concepts/stubby",
        PageEntry(title="Stubby", type="concept", status="stub"),
    )
    reg.save()
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    assert linter.check_broken_links() == []


def test_check_broken_links_ignores_deprecated_source(project: Path) -> None:
    """Edges from a deprecated/archived source page are noise, not signal:
    archive content is a historical record and shouldn't be flagged for
    cosmetic rewrites. The dominant real-world case is archive→archive
    after curation via deprecation."""
    _write_page(
        project, "concepts/old-source.md",
        body="[[concepts/old-target]]",
        status="deprecated",
    )
    _write_page(project, "concepts/old-target.md", status="deprecated")
    reg = _register(project, "concepts/old-source", status="deprecated")
    reg.register_page(
        "concepts/old-target",
        PageEntry(title="Old Target", type="concept", status="deprecated"),
    )
    reg.save()
    _rebuild_backlinks(project)

    linter = WikiLinter(project)
    assert linter.check_broken_links() == []


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


def test_check_orphans_excludes_deprecated_source_and_index(
    project: Path,
) -> None:
    """Regression guard: the old ``check_orphans`` leaked source pages
    through because the type filter was only applied in one of two
    code branches. Now the shared ``find_orphan_page_ids`` helper
    filters consistently — deprecated, source, and index pages never
    appear in the orphan list regardless of backlink state."""
    # A source page with zero inbound — previously leaked.
    _write_page(project, "sources/paper.md", type_="source")
    _register(project, "sources/paper", type_="source")
    # A deprecated page — should not appear.
    _write_page(project, "concepts/old.md", status="deprecated")
    _register(project, "concepts/old", status="deprecated")
    # A genuine orphan to prove the function still finds real cases.
    _write_page(project, "concepts/lonely.md")
    _register(project, "concepts/lonely")
    _rebuild_backlinks(project)

    orphans = WikiLinter(project).check_orphans()
    assert orphans == ["concepts/lonely"]


def test_lint_and_cli_orphans_agree_on_same_project(project: Path) -> None:
    """``wikiloom orphans`` and ``wikiloom lint`` must report the same
    set of page_ids. Previous versions had two different definitions
    (zero-edges vs zero-inbound) that disagreed on the same wiki."""
    from click.testing import CliRunner

    from wikiloom.cli import main
    from wikiloom.lint import find_orphan_page_ids

    # Seed a realistic mix: linked, orphan-by-inbound, deprecated,
    # source — so any filter bug would show up as a divergence.
    _write_page(project, "concepts/hub.md", body="[[concepts/leaf]]")
    _write_page(project, "concepts/leaf.md")
    _write_page(project, "concepts/island.md")  # orphan
    _write_page(project, "sources/paper.md", type_="source")  # excluded
    _register(project, "concepts/hub")
    _register(project, "concepts/leaf")
    _register(project, "concepts/island")
    _register(project, "sources/paper", type_="source")
    _rebuild_backlinks(project)

    # Shared helper — the single source of truth for both commands.
    expected = set(
        find_orphan_page_ids(
            Registry(project / "_registry"),
            BacklinkRegistry(project / "_registry"),
        )
    )

    # lint.check_orphans and cli.orphans must both match.
    lint_result = set(WikiLinter(project).check_orphans())
    assert lint_result == expected

    runner = CliRunner()
    cli_result = runner.invoke(
        main, ["orphans", "--project", str(project)]
    )
    assert cli_result.exit_code == 0
    for page_id in expected:
        assert page_id in cli_result.output, (
            f"{page_id} missing from cli orphans: {cli_result.output}"
        )


# ----------------------------------------------------------------------
# check_dormant
# ----------------------------------------------------------------------


def test_check_dormant_flags_old_concept(project: Path) -> None:
    old = _iso_days_ago(200)
    _write_page(project, "concepts/old.md", modified=old)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "concepts/old",
        PageEntry(title="Old", type="concept", modified=old, dormant_window_days=0),
    )
    # Clobber modified which register_page auto-refreshes
    reg.pages["concepts/old"].modified = old
    reg.pages["concepts/old"].dormant_window_days = 120
    reg.save()

    linter = WikiLinter(project, dormant=DormantConfig(concept_window_days=120))
    dormant = linter.check_dormant()
    assert len(dormant) == 1
    assert dormant[0].page_id == "concepts/old"
    assert dormant[0].age_days >= 120


def test_check_dormant_uses_per_type_window(project: Path) -> None:
    recent = _iso_days_ago(30)
    _write_page(project, "syntheses/recent.md", type_="synthesis", modified=recent)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "syntheses/recent",
        PageEntry(title="Recent", type="synthesis"),
    )
    reg.pages["syntheses/recent"].modified = recent
    reg.pages["syntheses/recent"].dormant_window_days = 0  # force fallback
    reg.save()

    # Synthesis window is 60 days by default — 30-day page is fresh
    linter = WikiLinter(project)
    assert linter.check_dormant() == []


def test_check_dormant_skips_deprecated(project: Path) -> None:
    old = _iso_days_ago(500)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page(
        "concepts/old",
        PageEntry(title="Old", type="concept", status="deprecated"),
    )
    reg.pages["concepts/old"].modified = old
    reg.save()

    linter = WikiLinter(project)
    assert linter.check_dormant() == []


# ----------------------------------------------------------------------
# check_duplicates
# ----------------------------------------------------------------------


def test_check_duplicates_flags_near_identical_titles(project: Path) -> None:
    """Near-identical slugs get flagged via find_duplicates's slug match."""
    from wikiloom.cache import SQLiteCache

    _write_page(project, "concepts/flash-attention.md", title="Flash Attention")
    _write_page(project, "concepts/flash-attentions.md", title="Flash Attentions")
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
    # lint.check_duplicates now delegates to find_duplicates, which
    # reads from the cache. Sync it to match real-world usage (lint
    # typically runs after ingest, which syncs automatically).
    SQLiteCache(project / "_registry" / "wiki.db").full_rebuild(project)

    linter = WikiLinter(project)
    dups = linter.check_duplicates()
    assert len(dups) == 1
    assert set(dups[0].pages) == {
        "concepts/flash-attention",
        "concepts/flash-attentions",
    }
    # Both signals populated on the shared DuplicateSet shape.
    assert dups[0].slug_score >= 80
    assert dups[0].embedding_score >= -1.0  # may be n/a if no cache


def test_check_duplicates_flags_near_identical_slugs(project: Path) -> None:
    """Two pages with slug variants (openai / open-ai) are flagged."""
    from wikiloom.cache import SQLiteCache

    _write_page(project, "entities/openai.md", title="OpenAI", type_="entity")
    _write_page(project, "entities/open-ai.md", title="Open-AI", type_="entity")
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
    SQLiteCache(project / "_registry" / "wiki.db").full_rebuild(project)

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


def test_check_index_consistency_compares_slugs_not_titles(project: Path) -> None:
    """Regression guard: the index rows use ``[Human Title](slug.md)``,
    where the link text is a human-readable title and the URL is the
    slug. The check must compare slugs (from the URL) against file
    stems on disk — not the title text — or every real-world index
    falsely reports drift because ``"Human Title" != "slug"``."""
    _write_page(project, "concepts/consent-withdrawal.md")
    (project / "wiki" / "concepts" / "index.md").write_text(
        "# Concepts\n\n"
        "| Page | Summary |\n"
        "|------|---------|\n"
        "| [Consent Withdrawal](consent-withdrawal.md) | s |\n"
    )
    linter = WikiLinter(project)
    assert "concepts" not in linter.check_index_consistency()


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


def test_check_promoted_from_update_flags_only_promoted_pages(
    project: Path,
) -> None:
    """The check lists page_ids whose frontmatter has
    ``promoted_from_update: true``, ignoring everything else."""
    # Regular page — flag is False by default.
    _write_page(project, "concepts/normal.md")
    _register(project, "concepts/normal")

    # Promoted page — write manually with the flag set.
    path = project / "wiki" / "concepts" / "from-hallucination.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = Frontmatter(
        title="From Hallucination",
        type="concept",
        status="active",
        created=now_iso(),
        modified=now_iso(),
        summary="summary",
        promoted_from_update=True,
    )
    path.write_text(render_frontmatter(fm) + "\nbody\n", encoding="utf-8")
    _register(project, "concepts/from-hallucination")

    # Deprecated pages are excluded — they're out of normal flow.
    _write_page(project, "concepts/old.md", status="deprecated")
    _register(project, "concepts/old", status="deprecated")
    deprecated_path = project / "wiki" / "concepts" / "old.md"
    fm_dep, body = parse_frontmatter(
        deprecated_path.read_text(encoding="utf-8")
    )
    assert fm_dep is not None
    fm_dep.promoted_from_update = True
    deprecated_path.write_text(
        render_frontmatter(fm_dep) + "\n" + body,
        encoding="utf-8",
    )

    linter = WikiLinter(project)
    promoted = linter.check_promoted_from_update()
    assert promoted == ["concepts/from-hallucination"]


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
    assert report.total_warnings >= 1
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


def test_fix_all_does_not_auto_mark_dormant(project: Path) -> None:
    """`lint --fix` reports dormant candidates but never marks them.

    Marking is a user decision via `wikiloom dormant <page>` — age
    alone is not a verdict on usefulness.
    """
    old = _iso_days_ago(500)
    page = _write_page(project, "concepts/old.md", modified=old)
    reg = Registry(project / "_registry", project / "wiki")
    reg.register_page("concepts/old", PageEntry(title="Old", type="concept"))
    reg.pages["concepts/old"].modified = old
    reg.save()

    linter = WikiLinter(project)
    report = linter.run_all()
    fixes = linter.fix_all(report)

    # The candidate should appear in the report but the page must NOT
    # have been auto-marked dormant.
    assert any(d.page_id == "concepts/old" for d in report.dormant)
    assert "status: dormant" not in page.read_text(encoding="utf-8")
    # FixReport no longer carries a stale_marked / dormant_marked counter.
    assert not hasattr(fixes, "stale_marked")
    assert not hasattr(fixes, "dormant_marked")


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


def test_lint_report_total_warnings_sums_all_categories() -> None:
    """Warnings = content-integrity findings only. Orphans, dormant,
    stubs, and promoted-from-update are tracking signals, not warnings,
    and don't contribute to the count or affect ``is_healthy``."""
    report = LintReport(
        broken_links=[BrokenLink("a", "b", "")],     # warning
        orphans=["c"],                                # tracking
        dormant=[DormantPage("d", 100, 90)],          # tracking
        duplicates=[                                  # warning
            DuplicateSet(pages=("e", "f"), slug_score=95, embedding_score=-1.0)
        ],
        frontmatter_issues=["g"],                     # warning
        index_drift=["concepts"],                     # warning
        contradictions=[],                            # warning (empty)
        stubs=["h"],                                  # tracking
    )
    assert report.total_warnings == 4
    assert report.is_healthy is False


def test_lint_report_total_tracking_counts_informational() -> None:
    """Tracking aggregates orphans, dormant, stubs, and promoted-from-update."""
    report = LintReport(
        orphans=["a", "b"],
        dormant=[DormantPage("c", 100, 90)],
        stubs=["d"],
        promoted_from_update=["e", "f"],
    )
    assert report.total_tracking == 6
    assert report.total_warnings == 0
    assert report.is_healthy is True  # tracking-only is still healthy


def test_lint_report_empty_is_healthy() -> None:
    assert LintReport().is_healthy
