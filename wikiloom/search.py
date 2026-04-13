"""Component 9: Tiered Navigation — IndexUpdater.

Regenerates ``wiki/index.md`` and ``wiki/<category>/index.md`` from the
current manifest + on-disk frontmatter. Index files are the LLM's
scalable entry point into the wiki: root → sub-index → specific pages,
so the LLM never has to load the whole corpus into context.

Design notes
------------
- **Frontmatter is preserved.** Each index.md keeps its existing YAML
  frontmatter block on regeneration; only the markdown body is
  rewritten. This protects any human annotations in the header.
- **Deterministic output.** Rows sort by ``modified`` descending, then
  page name ascending, so reindexing an unchanged wiki produces a
  byte-identical file and doesn't create cosmetic git churn.
- **Archive is excluded** from both the root index's category list
  and ``rebuild_all``. Deprecated pages live there and the deprecate
  flow maintains their index separately.
- **Category descriptions are hard-coded.** They're conventional and
  rarely change; pulling them into config would add surface area for
  no user benefit.
- **Query flow is deferred.** The ``wikiloom query`` CLI and the full
  LLM-driven navigation loop land after Component 20 (LLM Provider
  Abstraction). This module only handles the deterministic write path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wikiloom.frontmatter import parse_frontmatter, render_frontmatter
from wikiloom.registry import Registry


# Categories shown in the root index, in display order. ``archive`` is
# deliberately absent — it's not part of the navigable surface.
_ROOT_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("entities", "Entities", "People, organizations, tools, projects."),
    ("concepts", "Concepts", "Ideas, methods, patterns, principles."),
    ("sources", "Sources", "Summaries of ingested raw documents."),
    ("syntheses", "Syntheses", "Cross-cutting analyses and comparisons."),
    ("decisions", "Decisions", "Decision records with rationale."),
)

_SUB_INDEX_HEADER = "| Page | Summary | Modified | Sources | Links |"
_SUB_INDEX_DIVIDER = "|------|---------|----------|---------|-------|"


@dataclass(frozen=True)
class _PageRow:
    """One row in a sub-index table."""

    name: str
    title: str
    summary: str
    modified: str
    source_count: int
    inbound_links: int

    def render(self) -> str:
        modified = self.modified.split("T", 1)[0] if self.modified else ""
        summary = _escape_pipes(self.summary or "")
        title = _escape_pipes(self.title or self.name)
        return (
            f"| [{title}]({self.name}.md) | {summary} | {modified} | "
            f"{self.source_count} | {self.inbound_links} |"
        )


def _escape_pipes(text: str) -> str:
    """Escape ``|`` so it doesn't break markdown table rendering."""
    return text.replace("|", "\\|").replace("\n", " ")


class IndexUpdater:
    """Regenerates wiki index files from the manifest and page frontmatter."""

    def __init__(self, wiki_dir: Path, registry: Registry | None = None):
        self.wiki_dir = Path(wiki_dir)
        self.registry = registry or Registry(
            self.wiki_dir.parent / "_registry", self.wiki_dir
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebuild_sub_index(self, directory: Path) -> Path:
        """Regenerate ``directory/index.md``.

        Rows are sorted by ``modified`` descending (most recent first),
        with page name ascending as a deterministic tiebreaker.
        Returns the path to the rewritten index.
        """
        directory = Path(directory)
        index_path = directory / "index.md"
        rows = self._collect_rows(directory)
        body = self._render_sub_index_body(directory.name, rows)
        self._write_with_frontmatter(index_path, body)
        return index_path

    def rebuild_root_index(self) -> Path:
        """Regenerate ``wiki/index.md`` with live category counts."""
        index_path = self.wiki_dir / "index.md"
        body = self._render_root_index_body()
        self._write_with_frontmatter(index_path, body)
        return index_path

    def rebuild_all(self) -> list[Path]:
        """Rebuild the root index plus every non-archive sub-index.

        Returns the list of index files that were rewritten.
        """
        written: list[Path] = []
        if not self.wiki_dir.exists():
            return written

        for subdir in sorted(self.wiki_dir.iterdir()):
            if not subdir.is_dir() or subdir.name == "archive":
                continue
            written.append(self.rebuild_sub_index(subdir))

        written.append(self.rebuild_root_index())
        return written

    # ------------------------------------------------------------------
    # Sub-index rendering
    # ------------------------------------------------------------------

    def _collect_rows(self, directory: Path) -> list[_PageRow]:
        rows: list[_PageRow] = []
        category = directory.name
        for md_path in directory.glob("*.md"):
            if md_path.name == "index.md":
                continue
            fm, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
            name = md_path.stem
            page_id = f"{category}/{name}"
            entry = self.registry.get_page(page_id)

            if fm is not None:
                title = fm.title or name
                summary = fm.summary or ""
                modified = fm.modified or ""
                source_count = fm.source_count
            else:
                title = entry.title if entry else name
                summary = entry.summary if entry else ""
                modified = entry.modified if entry else ""
                source_count = entry.source_count if entry else 0

            inbound = entry.inbound_link_count if entry else 0
            rows.append(
                _PageRow(
                    name=name,
                    title=title,
                    summary=summary,
                    modified=modified,
                    source_count=source_count,
                    inbound_links=inbound,
                )
            )

        # Deterministic: newest first, name ascending as tiebreaker.
        # Pages with empty `modified` sort last (treated as epoch).
        rows.sort(key=lambda r: (r.modified or "", r.name), reverse=False)
        rows.sort(key=lambda r: r.modified or "", reverse=True)
        return rows

    def _render_sub_index_body(self, category: str, rows: list[_PageRow]) -> str:
        title = category.title()
        count = len(rows)
        lines = [f"# {title} Index ({count} pages)", ""]
        if not rows:
            lines.append("*No pages yet.*")
            lines.append("")
            return "\n".join(lines)

        lines.append(_SUB_INDEX_HEADER)
        lines.append(_SUB_INDEX_DIVIDER)
        lines.extend(row.render() for row in rows)
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Root index rendering
    # ------------------------------------------------------------------

    def _render_root_index_body(self) -> str:
        lines = ["# Wiki Index", ""]
        for slug, title, description in _ROOT_CATEGORIES:
            subdir = self.wiki_dir / slug
            count = self._count_pages(subdir)
            lines.append(f"## {title} ({count} pages)")
            lines.append(description)
            lines.append(f"→ See [{slug}/index.md]({slug}/index.md)")
            lines.append("")
        return "\n".join(lines)

    def _count_pages(self, directory: Path) -> int:
        if not directory.exists():
            return 0
        return sum(
            1 for p in directory.glob("*.md") if p.name != "index.md"
        )

    # ------------------------------------------------------------------
    # Frontmatter-preserving write
    # ------------------------------------------------------------------

    def _write_with_frontmatter(self, index_path: Path, body: str) -> None:
        """Rewrite ``index_path`` while preserving its frontmatter block.

        If the existing file has no frontmatter, the body is written as-is.
        """
        if index_path.exists():
            existing = index_path.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(existing)
        else:
            fm = None

        index_path.parent.mkdir(parents=True, exist_ok=True)
        if fm is not None:
            index_path.write_text(
                render_frontmatter(fm) + "\n" + body, encoding="utf-8"
            )
        else:
            index_path.write_text(body, encoding="utf-8")
