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
        # Merge sources, including their nested chunk_ids — when both
        # sides reference the same source, their chunk_ids unions.
        winner_fm.sources = _union_sources(
            winner_fm.sources or [],
            loser_fm.sources if loser_fm else [],
        )
        winner_fm.source_count = len(winner_fm.sources)
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
    rewrote = rewrite_inbound_links(
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

    # Event emission moved to callers so the ``git_commit_hash``
    # field can be populated (the commit happens *after* this returns).
    # Callers should use ``emit_merge_event`` below.

    return MergeResult(
        winner_page_id=winner_page_id,
        loser_page_id=loser_page_id,
        rewrote_links_in=rewrote,
        archive_path=archive_path,
    )


def emit_merge_event(
    project_root: Path,
    result: MergeResult,
    commit_hash: str | None,
) -> None:
    """Append a MERGE event to log.md with the commit hash attached.

    Called by every merge path (single cli.merge, the batched
    review/auto-merge paths, and the post-ingest merge) after the
    commit has landed so ``git_commit_hash`` is populated on the
    event — matching how INGEST events work. Previously the event
    was emitted inside ``merge_pages`` before the commit existed, so
    the hash was always empty in the log.
    """
    log_path = project_root / "wiki" / "log.md"
    if not log_path.parent.exists():
        return
    event = create_event(
        EventType.MERGE,
        description=f"{result.loser_page_id} → {result.winner_page_id}",
        pages_updated=[result.winner_page_id] + result.rewrote_links_in,
        pages_deprecated=[result.loser_page_id],
        git_commit_hash=commit_hash,
    )
    append_event(log_path, event)


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
    """Union two source-dict lists, deduping by ``hash`` (then ``name``).

    When the same source appears on both sides, the resulting entry's
    ``chunk_ids`` is the union of both sides' chunk_ids (order-preserved,
    deduped). Non-dict entries pass through unchanged.
    """
    by_key: dict[str, dict] = {}
    order: list[str] = []
    out: list[dict] = []

    def _key(item: dict) -> str:
        return item.get("hash") or item.get("name", "").lower() or str(id(item))

    for item in list(a) + list(b):
        if not isinstance(item, dict):
            out.append(item)
            continue
        k = _key(item)
        if k not in by_key:
            by_key[k] = {
                **item,
                "chunk_ids": list(item.get("chunk_ids") or []),
            }
            order.append(k)
        else:
            existing_ids = by_key[k]["chunk_ids"]
            seen = set(existing_ids)
            for cid in item.get("chunk_ids") or []:
                if cid and cid not in seen:
                    seen.add(cid)
                    existing_ids.append(cid)

    out.extend(by_key[k] for k in order)
    return out


def rewrite_inbound_links(
    wiki_dir: Path,
    backlinks: BacklinkRegistry,
    loser_page_id: str,
    winner_page_id: str,
) -> list[str]:
    """Rewrite ``[[loser]]`` → ``[[winner]]`` in every page that linked to loser.

    Preserves any ``|display text`` segment. Returns the list of
    page_ids whose bodies were modified.

    Used by both ``wikiloom merge`` (loser → winner) and
    ``wikiloom deprecate --superseded-by`` (deprecated → replacement).
    Archived source pages are skipped automatically because the
    backlinks rebuilder excludes ``wiki/archive/`` from its walk —
    historical content stays as-is.
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
