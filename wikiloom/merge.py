"""Page merge — combine two pages into one.

Used by ``wikiloom merge`` to resolve near-duplicate pages produced
by the LLM synthesis loop. Combines bodies, redirects inbound
wikilinks, and deprecates the loser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.events import EventType, append_event, create_event
from wikiloom.frontmatter import read_page, write_page
from wikiloom.registry import Registry
from wikiloom.utils import now_iso


@dataclass
class MergeResult:
    winner_page_id: str
    loser_page_id: str
    rewrote_links_in: list[str] = field(default_factory=list)
    archive_path: Path | None = None


def merge_pages(
    project_root: Path,
    winner_page_id: str,
    loser_page_id: str,
) -> MergeResult:
    """Merge ``loser`` into ``winner``.

    Steps:
    1. Append loser's body under a "Merged content from ..." section in
       winner's body so a human can reconcile the two later.
    2. Union aliases, sources, chunk_ids; add loser's title as alias.
    3. Rewrite ``[[loser]]`` wikilinks in every other page to ``[[winner]]``.
    4. Deprecate loser (move to ``wiki/archive/``, set ``superseded_by``).
    5. Emit a MERGE event to ``wiki/log.md``.

    Raises ``ValueError`` if the pages don't exist, are the same page,
    or the loser is already deprecated.
    """
    if winner_page_id == loser_page_id:
        raise ValueError("Cannot merge a page with itself.")

    wiki_dir = project_root / "wiki"
    registry = Registry(project_root / "_registry", wiki_dir=wiki_dir)

    winner_entry = registry.get_page(winner_page_id)
    loser_entry = registry.get_page(loser_page_id)

    if winner_entry is None:
        raise ValueError(f"Winner page not found in manifest: {winner_page_id}")
    if loser_entry is None:
        raise ValueError(f"Loser page not found in manifest: {loser_page_id}")
    if loser_entry.status != "active":
        raise ValueError(
            f"Loser page is not active (status={loser_entry.status}): {loser_page_id}"
        )

    winner_path = wiki_dir / f"{winner_page_id}.md"
    loser_path = wiki_dir / f"{loser_page_id}.md"

    if not winner_path.exists():
        raise ValueError(f"Winner page file missing: {winner_path}")
    if not loser_path.exists():
        raise ValueError(f"Loser page file missing: {loser_path}")

    winner_fm, winner_body = read_page(winner_path)
    loser_fm, loser_body = read_page(loser_path)

    new_body = _combine_bodies(winner_body, loser_body, loser_entry.title)

    if winner_fm is not None:
        winner_fm.aliases = _union_strings(
            winner_fm.aliases or [],
            (loser_fm.aliases if loser_fm else []) + [loser_entry.title],
        )
        winner_fm.sources = _union_sources(
            winner_fm.sources or [],
            loser_fm.sources if loser_fm else [],
        )
        winner_fm.chunk_ids = _union_strings(
            winner_fm.chunk_ids or [],
            loser_fm.chunk_ids if loser_fm else [],
        )
        winner_fm.source_count = max(
            len(winner_fm.sources), winner_fm.source_count or 0
        )
        winner_fm.modified = now_iso()
        write_page(winner_path, winner_fm, new_body)
    else:
        winner_path.write_text(new_body, encoding="utf-8")

    # Mirror the union into the manifest.
    winner_entry.aliases = _union_strings(
        winner_entry.aliases or [],
        (loser_entry.aliases or []) + [loser_entry.title],
    )
    winner_entry.source_count = (winner_entry.source_count or 0) + (
        loser_entry.source_count or 0
    )
    winner_entry.modified = now_iso()

    backlinks = BacklinkRegistry(project_root / "_registry", wiki_dir=wiki_dir)
    rewrote = _rewrite_inbound_links(
        wiki_dir, backlinks, loser_page_id, winner_page_id
    )

    archive_path = registry.deprecate_page(
        loser_page_id,
        superseded_by=winner_page_id,
        move_to_archive=True,
        emit_event=False,  # we emit MERGE instead
    )

    registry.save()

    backlinks.rebuild()
    backlinks.save()

    log_path = wiki_dir / "log.md"
    if log_path.parent.exists():
        event = create_event(
            EventType.MERGE,
            description=f"{loser_page_id} → {winner_page_id}",
            pages_updated=[winner_page_id] + rewrote,
            pages_deprecated=[loser_page_id],
        )
        append_event(log_path, event)

    return MergeResult(
        winner_page_id=winner_page_id,
        loser_page_id=loser_page_id,
        rewrote_links_in=rewrote,
        archive_path=archive_path,
    )


def _combine_bodies(winner_body: str, loser_body: str, loser_title: str) -> str:
    """Append loser's body under a "Merged content from ..." header.

    Doesn't try to be clever about deduplicating overlapping prose —
    that's a human-judgment task. Surfacing both side-by-side is the
    safer default.
    """
    winner_body = winner_body.rstrip()
    loser_body = loser_body.strip()
    if not loser_body:
        return winner_body + "\n"
    return (
        f"{winner_body}\n\n"
        f"<!-- merged from page: {loser_title} -->\n"
        f'## Merged content from "{loser_title}"\n\n'
        f"{loser_body}\n"
    )


def _union_strings(a: list[str], b: list[str]) -> list[str]:
    """Union two string lists preserving order, case-insensitive dedup."""
    seen: set[str] = set()
    out: list[str] = []
    for item in list(a) + list(b):
        key = item.lower() if isinstance(item, str) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _union_sources(a: list[dict], b: list[dict]) -> list[dict]:
    """Union two source-dict lists, deduping by ``name``."""
    seen: set[str] = set()
    out: list[dict] = []
    for item in list(a) + list(b):
        if not isinstance(item, dict):
            out.append(item)
            continue
        key = (item.get("name") or "").lower() or str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _rewrite_inbound_links(
    wiki_dir: Path,
    backlinks: BacklinkRegistry,
    loser_page_id: str,
    winner_page_id: str,
) -> list[str]:
    """Rewrite ``[[loser]]`` → ``[[winner]]`` in every page that linked to loser.

    Preserves any ``|display text`` segment. Returns the list of
    page_ids whose bodies were modified.
    """
    inbound_sources: list[str] = []
    for edge in backlinks.edges:
        if edge.target == loser_page_id and edge.source not in inbound_sources:
            inbound_sources.append(edge.source)

    pattern = re.compile(rf"\[\[{re.escape(loser_page_id)}(\|[^\]]*)?\]\]")
    modified: list[str] = []
    for source_id in inbound_sources:
        page_path = wiki_dir / f"{source_id}.md"
        if not page_path.exists():
            continue
        text = page_path.read_text(encoding="utf-8")
        new_text = pattern.sub(
            lambda m: f"[[{winner_page_id}{m.group(1) or ''}]]",
            text,
        )
        if new_text != text:
            page_path.write_text(new_text, encoding="utf-8")
            modified.append(source_id)

    return modified
