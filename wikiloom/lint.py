"""Lint & Health System.

Batch health checks and auto-fix over a WikiLoom project. Skips
human-edited pages to avoid clobbering hand-written content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.config import DormantConfig
from wikiloom.frontmatter import (
    Frontmatter,
    parse_frontmatter,
    render_frontmatter,
)
from wikiloom.git_ops import GitOps
from wikiloom.registry import Registry
from wikiloom.utils import now_iso, page_id_from_path, parse_iso


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BrokenLink:
    source: str          # page_id containing the link
    target: str          # missing-or-inactive page_id the link points at
    context: str         # snippet stored in backlinks.json
    reason: str = "missing"  # "missing" | "deprecated" | "stub" | "archived"


@dataclass(frozen=True)
class DormantPage:
    page_id: str
    age_days: int
    window_days: int


@dataclass(frozen=True)
class DuplicateSet:
    pages: tuple[str, ...]
    reason: str          # "title" | "alias"
    score: int           # rapidfuzz similarity 0-100


@dataclass(frozen=True)
class Contradiction:
    page_id: str
    existing: str
    new: str
    source: str


@dataclass
class LintReport:
    broken_links: list[BrokenLink] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    dormant: list[DormantPage] = field(default_factory=list)
    duplicates: list[DuplicateSet] = field(default_factory=list)
    frontmatter_issues: list[str] = field(default_factory=list)
    index_drift: list[str] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    stubs: list[str] = field(default_factory=list)
    promoted_from_update: list[str] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        # ``dormant`` and ``promoted_from_update`` are informational,
        # not "issues" — they're review prompts, not health problems.
        # Excluded from the total so a healthy wiki with promoted
        # pages under review still reads as healthy.
        return (
            len(self.broken_links)
            + len(self.orphans)
            + len(self.duplicates)
            + len(self.frontmatter_issues)
            + len(self.index_drift)
            + len(self.contradictions)
            + len(self.stubs)
        )

    @property
    def is_healthy(self) -> bool:
        return self.total_issues == 0


@dataclass
class FixReport:
    broken_links_fixed: int = 0
    frontmatter_repaired: int = 0
    indexes_rebuilt: int = 0
    skipped_human_edited: int = 0

    @property
    def total_fixed(self) -> int:
        return (
            self.broken_links_fixed
            + self.frontmatter_repaired
            + self.indexes_rebuilt
        )


# ----------------------------------------------------------------------
# Linter
# ----------------------------------------------------------------------


# Required frontmatter fields for a non-index page to be considered valid.
REQUIRED_FRONTMATTER_FIELDS: tuple[str, ...] = (
    "title",
    "type",
    "status",
    "created",
    "modified",
    "summary",
)

# Fuzzy duplicate threshold — pages whose titles/aliases exceed this
# rapidfuzz ratio are flagged. Conservative: exact matches always
# trigger, near-identical slugs trigger, unrelated titles do not.
_DUPLICATE_THRESHOLD = 92

_WIKILINK_RE = re.compile(r"\[\[([^\]|\n]+)(?:\|([^\]\n]+))?\]\]")

_INDEX_TABLE_ROW_RE = re.compile(r"^\|\s*\[([^\]]+)\]\(([^)]+)\)", re.MULTILINE)


class WikiLinter:
    """Runs the full lint pass and (optionally) applies auto-fixes."""

    def __init__(
        self,
        project_root: Path,
        dormant: DormantConfig | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.wiki_dir = self.project_root / "wiki"
        self.registry_dir = self.project_root / "_registry"
        self.dormant = dormant or DormantConfig()

        self.registry = Registry(self.registry_dir, self.wiki_dir)
        self.backlinks = BacklinkRegistry(self.registry_dir, self.wiki_dir)
        try:
            self.git = GitOps(self.project_root)
        except ValueError:
            self.git = None  # lint can run outside a git repo for tests

    # ------------------------------------------------------------------
    # Top-level entry points
    # ------------------------------------------------------------------

    def run_all(self) -> LintReport:
        """Run every health check and return an aggregate report."""
        return LintReport(
            broken_links=self.check_broken_links(),
            orphans=self.check_orphans(),
            dormant=self.check_dormant(),
            duplicates=self.check_duplicates(),
            frontmatter_issues=self.check_frontmatter(),
            index_drift=self.check_index_consistency(),
            contradictions=self.check_contradictions(),
            stubs=self.check_stubs(),
            promoted_from_update=self.check_promoted_from_update(),
        )

    def fix_all(self, report: LintReport) -> FixReport:
        """Apply mechanical fixes from a report.

        Skips any page whose most recent commit is a ``human-edit:`` —
        that's the enforcement point Component 10 relies on.
        """
        fixes = FixReport()

        for broken in report.broken_links:
            page_path = self._page_path(broken.source)
            if page_path is None:
                continue
            if self._is_protected(page_path):
                fixes.skipped_human_edited += 1
                continue
            if self._strip_broken_wikilink(page_path, broken.target):
                fixes.broken_links_fixed += 1

        # Dormant pages are reported but never auto-marked. Marking is
        # a user decision via `wikiloom dormant <page>` — age alone is
        # not a verdict on usefulness.

        for page_id in report.frontmatter_issues:
            page_path = self._page_path(page_id)
            if page_path is None:
                continue
            if self._is_protected(page_path):
                fixes.skipped_human_edited += 1
                continue
            if self._repair_frontmatter(page_path):
                fixes.frontmatter_repaired += 1

        # Index drift: regenerate the drifted sub-indexes (and the root
        # index, since counts may have shifted). Indexes are derived
        # state — human-edit protection doesn't apply here.
        if report.index_drift:
            from wikiloom.search import IndexUpdater

            updater = IndexUpdater(self.wiki_dir, registry=self.registry)
            for name in report.index_drift:
                subdir = self.wiki_dir / name
                if subdir.is_dir():
                    updater.rebuild_sub_index(subdir)
                    fixes.indexes_rebuilt += 1
            updater.rebuild_root_index()

        return fixes

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_broken_links(self) -> list[BrokenLink]:
        """Edges whose target is missing or retired.

        Reads from ``backlinks.json`` (already rebuilt by the ingest
        pipeline) rather than re-parsing bodies — cheaper and keeps the
        wikilink regex as a single source of truth in ``backlinks.py``.

        Classification:

        - **missing**: target absent from the manifest → ``--fix`` strips
          the wikilink wrapper.
        - **deprecated** / **archived**: target was explicitly retired →
          ``--fix`` strips the wrapper (callers may want to redirect to
          ``superseded_by`` in a future pass).
        - **stub**: deliberately *not* flagged. Stubs are placeholder
          pages the linker created for unresolved entities; links to
          them are expected to stay until the stub gets filled in, and
          ``check_stubs`` tracks them separately.
        """
        broken: list[BrokenLink] = []
        for edge in self.backlinks._edges:
            entry = self.registry.get_page(edge.target)
            if entry is None:
                reason = "missing"
            elif entry.status in ("active", "stub", "dormant"):
                continue  # dormant pages are valid link targets
            else:
                reason = entry.status  # "deprecated" | "archived" | ...
            broken.append(
                BrokenLink(
                    source=edge.source,
                    target=edge.target,
                    context=edge.context,
                    reason=reason,
                )
            )
        return broken

    def check_orphans(self) -> list[str]:
        """Active manifest pages with zero inbound links.

        Combines the backlink view (pages with edges but zero inbound)
        with manifest pages that never showed up in ``backlinks.json``
        at all — both count as orphans the linter should surface.
        Excludes index pages and sources, which aren't expected to be
        linked *to* from wiki prose.
        """
        backlink_orphans = set(self.backlinks.get_orphans())
        seen_in_backlinks: set[str] = set()
        for edge in self.backlinks._edges:
            seen_in_backlinks.add(edge.source)
            seen_in_backlinks.add(edge.target)

        orphans: set[str] = set(backlink_orphans)
        for page_id, entry in self.registry.pages.items():
            if entry.status != "active":
                continue
            if entry.type in ("source", "index"):
                continue
            if page_id not in seen_in_backlinks:
                orphans.add(page_id)
        return sorted(orphans)

    def check_dormant(self) -> list[DormantPage]:
        """Active pages whose ``modified`` date exceeds the dormant window.

        Per-page ``dormant_window_days`` in the manifest takes
        precedence; otherwise falls back to per-type windows from
        ``DormantConfig``. Dormant pages whose status is already
        ``dormant`` (user-marked) are not reported again — they're
        already known.
        """
        dormant_candidates: list[DormantPage] = []
        now = parse_iso(now_iso())
        for page_id, entry in self.registry.pages.items():
            if entry.status != "active" or not entry.modified:
                continue
            try:
                modified = parse_iso(entry.modified)
            except ValueError:
                continue
            age_days = (now - modified).days
            window = entry.dormant_window_days or self._window_for_type(entry.type)
            if age_days > window:
                dormant_candidates.append(
                    DormantPage(page_id=page_id, age_days=age_days, window_days=window)
                )
        return dormant_candidates

    def check_duplicates(self) -> list[DuplicateSet]:
        """Fuzzy-match titles, aliases, slugs, and embeddings for near-duplicates.

        Title/alias signal catches LLM output where the page name is
        similar. Slug/embedding signal catches LLM output where the
        slug got a disambiguating suffix (``pending-transactions`` vs
        ``pending-transactions-banking``) or where titles diverged but
        the bodies describe the same concept. Only runs on active pages
        and only within the same type.
        """
        from wikiloom.duplicates import find_duplicates

        pages = [
            (pid, entry)
            for pid, entry in self.registry.pages.items()
            if entry.status != "deprecated"
        ]
        duplicates: list[DuplicateSet] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, (pid_a, entry_a) in enumerate(pages):
            for pid_b, entry_b in pages[i + 1 :]:
                if entry_a.type != entry_b.type:
                    continue  # only flag within a category
                pair = tuple(sorted((pid_a, pid_b)))
                if pair in seen_pairs:
                    continue
                score = int(fuzz.ratio(entry_a.title.lower(), entry_b.title.lower()))
                reason = "title"
                if score < _DUPLICATE_THRESHOLD:
                    alias_score = self._best_alias_score(entry_a.aliases, entry_b.aliases)
                    if alias_score >= _DUPLICATE_THRESHOLD:
                        score = alias_score
                        reason = "alias"
                    else:
                        continue
                seen_pairs.add(pair)
                duplicates.append(
                    DuplicateSet(pages=pair, reason=reason, score=score)
                )

        # Slug + embedding signal — catches duplicates the title check
        # misses (different titles, same concept; slug-suffix variants).
        # Cache may not exist yet on a fresh project; find_duplicates
        # returns [] in that case.
        try:
            slug_pairs = find_duplicates(
                self.project_root,
                slug_threshold=85.0,
                embedding_threshold=0.88,
                same_type_only=True,
            )
        except Exception:
            slug_pairs = []
        for sp in slug_pairs:
            pair = tuple(sorted((sp.page_a, sp.page_b)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if sp.embedding_score >= 0.88:
                reason = "embedding"
                score = int(sp.embedding_score * 100)
            else:
                reason = "slug"
                score = int(sp.slug_score)
            duplicates.append(
                DuplicateSet(pages=pair, reason=reason, score=score)
            )
        return duplicates

    def check_frontmatter(self) -> list[str]:
        """Pages missing required frontmatter fields or with no frontmatter."""
        issues: list[str] = []
        for md_path in self._iter_content_pages():
            page_id = page_id_from_path(self.wiki_dir, md_path)
            fm, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
            if fm is None:
                issues.append(page_id)
                continue
            data = fm.to_dict()
            for required in REQUIRED_FRONTMATTER_FIELDS:
                if not data.get(required):
                    issues.append(page_id)
                    break
        return sorted(set(issues))

    def check_index_consistency(self) -> list[str]:
        """Sub-indexes whose table doesn't match on-disk page list.

        Detection only. The fixer lives in Component 9's ``IndexUpdater``;
        ``fix_all`` intentionally does not touch this finding today.
        Returns the list of sub-directories with drift.
        """
        drifted: list[str] = []
        if not self.wiki_dir.exists():
            return drifted

        for subdir in sorted(self.wiki_dir.iterdir()):
            if not subdir.is_dir() or subdir.name == "archive":
                continue
            index_path = subdir / "index.md"
            if not index_path.exists():
                drifted.append(subdir.name)
                continue
            on_disk = {
                p.stem for p in subdir.glob("*.md") if p.name != "index.md"
            }
            listed = {
                match.group(1)
                for match in _INDEX_TABLE_ROW_RE.finditer(
                    index_path.read_text(encoding="utf-8")
                )
            }
            if on_disk != listed:
                drifted.append(subdir.name)
        return drifted

    def check_contradictions(self) -> list[Contradiction]:
        """Pages with non-empty ``contradictions`` in frontmatter.

        Plumbing is live today; Component 13 populates the field during
        synthesis. Until then this returns empty and that's fine.
        """
        found: list[Contradiction] = []
        for md_path in self._iter_content_pages():
            fm, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
            if fm is None or not fm.contradictions:
                continue
            page_id = page_id_from_path(self.wiki_dir, md_path)
            for item in fm.contradictions:
                found.append(
                    Contradiction(
                        page_id=page_id,
                        existing=str(item.get("existing", "")),
                        new=str(item.get("new", "")),
                        source=str(item.get("source", "")),
                    )
                )
        return found

    def check_stubs(self) -> list[str]:
        """Pages whose manifest status is ``stub``."""
        return sorted(
            pid for pid, entry in self.registry.pages.items() if entry.status == "stub"
        )

    def check_promoted_from_update(self) -> list[str]:
        """Pages created via the 'unresolved update → promoted create' path.

        Reads ``promoted_from_update`` from each active page's
        frontmatter (not cached in the manifest, so this walks the
        files). Reviewer decides whether to keep, deprecate, or
        split; the flag is cleared by editing + ``wikiloom save``.
        """
        promoted: list[str] = []
        for page_id, entry in self.registry.pages.items():
            if entry.status == "deprecated":
                continue
            page_path = self._page_path(page_id)
            if page_path is None or not page_path.exists():
                continue
            fm, _ = parse_frontmatter(
                page_path.read_text(encoding="utf-8")
            )
            if fm is not None and fm.promoted_from_update:
                promoted.append(page_id)
        return sorted(promoted)

    # ------------------------------------------------------------------
    # Fix helpers
    # ------------------------------------------------------------------

    def _strip_broken_wikilink(self, page_path: Path, target: str) -> bool:
        """Replace ``[[target|display]]`` with its display text (or target)."""
        text = page_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)

        def repl(match: re.Match[str]) -> str:
            matched_target = match.group(1).strip()
            if matched_target != target:
                return match.group(0)
            display = match.group(2)
            return display.strip() if display else matched_target

        new_body, count = _WIKILINK_RE.subn(repl, body)
        if count == 0:
            return False

        if fm is not None:
            page_path.write_text(
                render_frontmatter(fm) + "\n" + new_body, encoding="utf-8"
            )
        else:
            page_path.write_text(new_body, encoding="utf-8")
        return True

    def _repair_frontmatter(self, page_path: Path) -> bool:
        """Fill in missing required fields with sensible defaults."""
        text = page_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if fm is None:
            fm = Frontmatter(
                title=page_path.stem.replace("-", " ").title(),
                type=self._infer_type(page_path),
                created=now_iso(),
                modified=now_iso(),
                summary="",
            )
            page_path.write_text(
                render_frontmatter(fm) + "\n" + text, encoding="utf-8"
            )
            return True

        changed = False
        if not fm.title:
            fm.title = page_path.stem.replace("-", " ").title()
            changed = True
        if not fm.type:
            fm.type = self._infer_type(page_path)
            changed = True
        if not fm.status:
            fm.status = "active"
            changed = True
        if not fm.created:
            fm.created = now_iso()
            changed = True
        if not fm.modified:
            fm.modified = now_iso()
            changed = True

        if changed:
            page_path.write_text(
                render_frontmatter(fm) + "\n" + body, encoding="utf-8"
            )
        return changed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_content_pages(self):
        """Yield every non-index markdown page under ``wiki/``."""
        if not self.wiki_dir.exists():
            return
        for md_path in self.wiki_dir.rglob("*.md"):
            if md_path.name in ("index.md", "log.md"):
                continue
            yield md_path

    def _page_path(self, page_id: str) -> Path | None:
        candidate = self.wiki_dir / f"{page_id}.md"
        return candidate if candidate.exists() else None

    def _is_protected(self, page_path: Path) -> bool:
        if self.git is None:
            return False
        try:
            return self.git.is_human_edited(page_path)
        except ValueError:
            return False

    def _window_for_type(self, page_type: str) -> int:
        if page_type == "entity":
            return self.dormant.entity_window_days
        if page_type == "concept":
            return self.dormant.concept_window_days
        if page_type == "synthesis":
            return self.dormant.synthesis_window_days
        return self.dormant.default_window_days

    def _infer_type(self, page_path: Path) -> str:
        parts = page_path.resolve().relative_to(self.wiki_dir.resolve()).parts
        if not parts:
            return "concept"
        category = parts[0]
        mapping = {
            "entities": "entity",
            "concepts": "concept",
            "sources": "source",
            "syntheses": "synthesis",
            "decisions": "decision",
        }
        return mapping.get(category, "concept")

    @staticmethod
    def _best_alias_score(a: list[str], b: list[str]) -> int:
        if not a or not b:
            return 0
        best = 0
        for x in a:
            for y in b:
                score = int(fuzz.ratio(x.lower(), y.lower()))
                if score > best:
                    best = score
        return best
