"""Page writer: synthesis output → wiki pages on disk.

Handles three write paths: fresh creates, collision with --force
(preserves human edits above the wikiloom:auto marker), and
append-updates from new sources. Also writes source summary pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wikiloom.frontmatter import Frontmatter, read_page, write_page
from wikiloom.protection import AUTO_MARKER, HumanEditProtection
from wikiloom.registry import PageEntry, Registry
from wikiloom.source_catalog import SourceEntry
from wikiloom.synthesis import PageProposal, SourceSummary, SynthesisResult
from wikiloom.utils import now_iso, slugify

# wiki/ subdirectories are plural forms of the type enum
_TYPE_TO_DIR = {
    "entity": "entities",
    "concept": "concepts",
    "synthesis": "syntheses",
    "decision": "decisions",
}
_SOURCE_DIR = "sources"


@dataclass
class PageWriteResult:
    """Summary of what the writer put on disk.

    The processor uses this to build the git commit's file list and
    populate ``pages_created`` / ``pages_updated`` on the ingest
    result.
    """

    created_paths: list[Path] = field(default_factory=list)
    updated_paths: list[Path] = field(default_factory=list)
    skipped_collisions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def created_page_ids(self) -> list[str]:
        return [_path_to_page_id(p, _wiki_root_of(p)) for p in self.created_paths]

    @property
    def updated_page_ids(self) -> list[str]:
        return [_path_to_page_id(p, _wiki_root_of(p)) for p in self.updated_paths]


def _wiki_root_of(page_path: Path) -> Path:
    """Walk upward to find the ``wiki/`` directory that contains a page path."""
    for parent in page_path.parents:
        if parent.name == "wiki":
            return parent
    return page_path.parent


def _path_to_page_id(path: Path, wiki_root: Path) -> str:
    rel = path.resolve().relative_to(wiki_root.resolve())
    return rel.with_suffix("").as_posix()


class PageWriter:
    """Writes synthesized pages to disk and registers them in the manifest."""

    def __init__(
        self,
        project_root: Path,
        registry: Registry,
        *,
        force: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.wiki_dir = self.project_root / "wiki"
        self.registry = registry
        self.force = force
        self.protection = HumanEditProtection(
            self.project_root, registry=registry
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def write(
        self,
        synthesis: SynthesisResult,
        source_entry: SourceEntry | None,
    ) -> PageWriteResult:
        """Write every page proposal in a ``SynthesisResult`` to disk.

        The order is:
        1. source page (if source_summary is present)
        2. pages_to_create (fresh or --force collision handling)
        3. pages_to_update (append-to-auto-region handling)

        Returns a ``PageWriteResult`` the processor can consume for
        commit staging and event logging.
        """
        result = PageWriteResult()

        # 1. Source page
        if synthesis.source_summary is not None and source_entry is not None:
            all_chunk_ids = _all_chunk_ids_from_synthesis(synthesis)
            src_path = self._write_source_page(
                synthesis.source_summary, source_entry, all_chunk_ids
            )
            if src_path is not None:
                # Source pages are always fresh-or-updated, not "created"
                # in the spec sense. Treat as created on first write,
                # updated otherwise — the registry tells us which.
                page_id = _path_to_page_id(src_path, self.wiki_dir)
                if self.registry.get_page(page_id) is None:
                    result.created_paths.append(src_path)
                else:
                    result.updated_paths.append(src_path)
                self._register(
                    src_path,
                    title=synthesis.source_summary.title,
                    type_="source",
                    summary=synthesis.source_summary.one_line,
                    source_hash=source_entry.content_hash,
                )

        # 2. pages_to_create
        for proposal in synthesis.pages_to_create:
            outcome = self._write_create(proposal, source_entry)
            if outcome.kind == "created":
                result.created_paths.append(outcome.path)
            elif outcome.kind == "updated":
                result.updated_paths.append(outcome.path)
            elif outcome.kind == "skipped":
                result.skipped_collisions.append(outcome.page_id)
                result.notes.append(
                    f"skipped {outcome.page_id} (page exists; use --force to replace)"
                )

        # 3. pages_to_update
        for proposal in synthesis.pages_to_update:
            outcome = self._write_update(proposal, source_entry)
            if outcome.kind == "updated":
                result.updated_paths.append(outcome.path)
            elif outcome.kind == "created":
                # Update target didn't exist — the writer promoted
                # the update into a create using the LLM's proposed
                # page_id. Count it under created pages.
                result.created_paths.append(outcome.path)
                result.notes.append(
                    f"update target not found: {outcome.page_id} "
                    f"— wrote as new page instead"
                )
            elif outcome.kind == "skipped":
                result.notes.append(
                    f"update target not found: {outcome.page_id} "
                    f"(LLM proposed update for nonexistent page)"
                )

        return result

    # ------------------------------------------------------------------
    # Source page
    # ------------------------------------------------------------------

    def _write_source_page(
        self,
        summary: SourceSummary,
        source_entry: SourceEntry,
        all_chunk_ids: list[str],
    ) -> Path | None:
        """Write the summary page for a just-ingested source."""
        target = self._resolve_source_path(source_entry)
        existing_fm, _ = (None, "") if not target.exists() else read_page(target)

        body = _ensure_auto_marker(summary.content_markdown)
        if target.exists():
            body = self.protection.preserve_human(target, summary.content_markdown)
            body = _ensure_auto_marker(body)
        fm = Frontmatter(
            title=summary.title or source_entry.name,
            type="source",
            status="active",
            created=(existing_fm.created if existing_fm else now_iso()),
            modified=now_iso(),
            summary=summary.one_line,
            sources=[
                {
                    "hash": source_entry.content_hash,
                    "name": source_entry.name,
                    "raw_path": source_entry.raw_path or "",
                    "chunk_ids": all_chunk_ids,
                }
            ],
            source_count=1,
            confidence="high",
        )
        write_page(target, fm, body)
        return target

    def _resolve_source_path(self, source_entry: SourceEntry) -> Path:
        """Pick a target path for a source page, handling slug collisions.

        Strategy: ``slugify(filename without extension)``. On collision,
        check the existing page's frontmatter — if it already references
        the same content_hash, overwrite. Otherwise, append a counter
        suffix so two different sources with the same filename stem
        don't stomp each other.
        """
        base = source_entry.name.rsplit(".", 1)[0] if "." in source_entry.name else source_entry.name
        base_slug = slugify(base) or "source"
        sources_dir = self.wiki_dir / _SOURCE_DIR
        sources_dir.mkdir(parents=True, exist_ok=True)

        candidate = sources_dir / f"{base_slug}.md"
        if not candidate.exists():
            return candidate

        existing_fm, _ = read_page(candidate)
        if existing_fm is not None:
            for src in existing_fm.sources or []:
                if isinstance(src, dict) and src.get("hash") == source_entry.content_hash:
                    return candidate  # same source, overwrite

        counter = 2
        while True:
            next_candidate = sources_dir / f"{base_slug}-{counter}.md"
            if not next_candidate.exists():
                return next_candidate
            existing_fm, _ = read_page(next_candidate)
            if existing_fm is not None:
                for src in existing_fm.sources or []:
                    if isinstance(src, dict) and src.get("hash") == source_entry.content_hash:
                        return next_candidate  # same source, overwrite
            counter += 1

    # ------------------------------------------------------------------
    # pages_to_create
    # ------------------------------------------------------------------

    def _write_create(
        self,
        proposal: PageProposal,
        source_entry: SourceEntry | None,
        *,
        promoted: bool = False,
    ) -> "_WriteOutcome":
        target = self._resolve_create_path(proposal)
        page_id = _path_to_page_id(target, self.wiki_dir)

        if target.exists():
            if not self.force:
                return _WriteOutcome(kind="skipped", path=target, page_id=page_id)
            # --force collision: preserve human region, replace auto
            body = self.protection.preserve_human(target, proposal.content_markdown)
            body = _ensure_auto_marker(body)
            existing_fm, _ = read_page(target)
            fm = self._make_create_frontmatter(
                proposal=proposal,
                existing_fm=existing_fm,
                source_entry=source_entry,
                promoted=promoted,
            )
            write_page(target, fm, body)
            self._register(
                target,
                title=proposal.title,
                type_=proposal.type,
                summary=_derive_summary(proposal.content_markdown),
                confidence=proposal.confidence,
            )
            return _WriteOutcome(kind="updated", path=target, page_id=page_id)

        # Fresh create
        body = _ensure_auto_marker(proposal.content_markdown)
        fm = self._make_create_frontmatter(
            proposal=proposal,
            existing_fm=None,
            source_entry=source_entry,
            promoted=promoted,
        )
        write_page(target, fm, body)
        self._register(
            target,
            title=proposal.title,
            type_=proposal.type,
            summary=_derive_summary(proposal.content_markdown),
            confidence=proposal.confidence,
        )
        return _WriteOutcome(kind="created", path=target, page_id=page_id)

    def _resolve_create_path(self, proposal: PageProposal) -> Path:
        subdir = _TYPE_TO_DIR.get(proposal.type)
        if subdir is None:
            raise ValueError(
                f"unknown page type {proposal.type!r} in create proposal"
            )
        slug = slugify(proposal.suggested_slug) or slugify(proposal.title) or "untitled"
        return self.wiki_dir / subdir / f"{slug}.md"

    def _make_create_frontmatter(
        self,
        proposal: PageProposal,
        existing_fm: Frontmatter | None,
        source_entry: SourceEntry | None,
        *,
        promoted: bool = False,
    ) -> Frontmatter:
        existing_sources = (existing_fm.sources if existing_fm else [])
        sources = _append_source(
            existing_sources, source_entry, [proposal.chunk_id]
        )
        return Frontmatter(
            title=proposal.title,
            type=proposal.type,
            status="active",
            created=(existing_fm.created if existing_fm else now_iso()),
            modified=now_iso(),
            summary=_derive_summary(proposal.content_markdown),
            aliases=(existing_fm.aliases if existing_fm else []),
            sources=sources,
            source_count=len(sources) or 1,
            confidence=proposal.confidence or "medium",
            human_edited=False,
            contradictions=(existing_fm.contradictions if existing_fm else []),
            tags=(existing_fm.tags if existing_fm else []),
            promoted_from_update=promoted,
        )

    # ------------------------------------------------------------------
    # pages_to_update
    # ------------------------------------------------------------------

    def _write_update(
        self,
        proposal: PageProposal,
        source_entry: SourceEntry | None,
    ) -> "_WriteOutcome":
        page_id = _normalize_existing_path(proposal.existing_path or "")
        if not page_id:
            return _WriteOutcome(kind="skipped", path=Path(), page_id="<empty>")

        target = self.wiki_dir / f"{page_id}.md"
        if not target.exists():
            # The LLM referenced a page that doesn't exist yet (a
            # common hallucination: it names a plausible-sounding
            # sibling page from training rather than from the
            # manifest context). Instead of dropping the content on
            # the floor, synthesize a create proposal from the
            # update's additions_markdown and write it. Preserves
            # the LLM's work; any resulting near-duplicate gets
            # caught by the slug-collision guard or post-ingest
            # merge on the next writer pass. The ``promoted=True``
            # flag flows into the page's frontmatter so
            # ``wikiloom lint`` can surface it for review.
            return self._write_create(
                _update_to_create_proposal(proposal, page_id),
                source_entry,
                promoted=True,
            )

        existing_fm, existing_body = read_page(target)
        if existing_fm is None:
            return _WriteOutcome(kind="skipped", path=target, page_id=page_id)

        human, auto = HumanEditProtection.split(existing_body)
        if auto:
            merged_auto = auto.rstrip() + "\n\n" + proposal.additions_markdown.lstrip()
        else:
            merged_auto = proposal.additions_markdown
        new_body = HumanEditProtection.merge(human, merged_auto)

        merged_contradictions = list(existing_fm.contradictions or [])
        for c in proposal.contradictions or []:
            if c not in merged_contradictions:
                merged_contradictions.append(c)

        # Auto-freshen: re-ingest update on a dormant page flips it
        # back to active. The user's "this is dormant" label was only
        # accurate at the time it was set; the page just got new content.
        new_status = "active" if existing_fm.status == "dormant" else existing_fm.status
        # Append the contributing source (or extend its chunk_ids if
        # already present). Provenance lives entirely under sources now.
        merged_sources = _append_source(
            existing_fm.sources, source_entry, [proposal.chunk_id]
        )
        fm = Frontmatter(
            title=existing_fm.title,
            type=existing_fm.type,
            status=new_status,
            created=existing_fm.created or now_iso(),
            modified=now_iso(),
            summary=existing_fm.summary,
            aliases=existing_fm.aliases,
            sources=merged_sources,
            source_count=len(merged_sources),
            confidence=existing_fm.confidence,
            dormant_window_days=existing_fm.dormant_window_days,
            human_edited=existing_fm.human_edited,
            human_edited_at=existing_fm.human_edited_at,
            superseded_by=existing_fm.superseded_by,
            contradictions=merged_contradictions,
            tags=existing_fm.tags,
        )
        write_page(target, fm, new_body)
        self._register(
            target,
            title=existing_fm.title,
            type_=existing_fm.type,
            summary=existing_fm.summary,
            confidence=existing_fm.confidence,
            source_hash=None,
        )
        return _WriteOutcome(kind="updated", path=target, page_id=page_id)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def _register(
        self,
        path: Path,
        *,
        title: str,
        type_: str,
        summary: str = "",
        confidence: str = "medium",
        source_hash: str | None = None,
    ) -> None:
        page_id = _path_to_page_id(path, self.wiki_dir)
        existing = self.registry.get_page(page_id)
        entry = PageEntry(
            title=title,
            type=type_,
            status="active",
            aliases=(existing.aliases if existing else []),
            created=(existing.created if existing else ""),
            modified="",  # register_page sets this
            summary=summary,
            source_count=((existing.source_count if existing else 0) + (1 if source_hash else 0)),
            confidence=confidence or "medium",
        )
        self.registry.register_page(page_id, entry)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@dataclass
class _WriteOutcome:
    kind: str  # "created" | "updated" | "skipped"
    path: Path
    page_id: str


def _ensure_auto_marker(body: str) -> str:
    """Prepend the auto marker if it isn't already in the body.

    The ``HumanEditProtection.preserve_human`` helper returns an
    unmarked body for brand-new pages (case 1 in its docstring).
    That's fine for its internal semantics but wrong for our
    on-disk format — a fresh page needs the marker so future
    re-ingests treat the whole body as auto. This helper is the
    single place that enforces the invariant.
    """
    if AUTO_MARKER in body:
        return body
    return f"{AUTO_MARKER}\n\n{body.lstrip()}"


def _append_source(
    existing: list[dict],
    source_entry: SourceEntry | None,
    chunk_ids: list[str] | None = None,
) -> list[dict]:
    """Append ``source_entry`` to ``existing`` and union ``chunk_ids``.

    If a source with the same ``hash`` (or ``name`` as fallback)
    already exists in ``existing``, its ``chunk_ids`` list is extended
    with any new ids from ``chunk_ids`` (deduped, order preserved).
    Otherwise a new entry is appended. Returns a new list — never
    mutates the input.
    """
    out: list[dict] = []
    for s in existing:
        if not isinstance(s, dict):
            continue
        copy = dict(s)
        copy["chunk_ids"] = list(s.get("chunk_ids") or [])
        out.append(copy)

    if source_entry is None:
        return out

    new_chunk_ids = list(chunk_ids or [])

    # Find an existing matching source and extend its chunk_ids.
    for s in out:
        same_hash = s.get("hash") and s["hash"] == source_entry.content_hash
        same_name = (
            not s.get("hash") and s.get("name") == source_entry.name
        )
        if same_hash or same_name:
            seen = set(s["chunk_ids"])
            for cid in new_chunk_ids:
                if cid and cid not in seen:
                    seen.add(cid)
                    s["chunk_ids"].append(cid)
            return out

    out.append(
        {
            "hash": source_entry.content_hash,
            "name": source_entry.name,
            "raw_path": source_entry.raw_path or "",
            "chunk_ids": new_chunk_ids,
        }
    )
    return out


_DIR_TO_TYPE = {v: k for k, v in _TYPE_TO_DIR.items()}
_DIR_TO_TYPE[_SOURCE_DIR] = "source"


def _update_to_create_proposal(
    proposal: PageProposal, page_id: str
) -> PageProposal:
    """Synthesize a create proposal from an unresolved update.

    Derives ``type`` and ``suggested_slug`` from the update's
    ``existing_path`` (which the LLM already chose), infers a title
    from the slug, and uses the update's ``additions_markdown`` as
    the new page body. Confidence defaults to ``"medium"`` since
    the LLM didn't explicitly set one for this proposal.
    """
    if "/" in page_id:
        type_dir, slug = page_id.split("/", 1)
    else:
        type_dir, slug = "", page_id
    inferred_type = _DIR_TO_TYPE.get(type_dir, "concept")
    title = slug.replace("-", " ").title() if slug else page_id
    return PageProposal(
        intent="create",
        chunk_id=proposal.chunk_id,
        title=title,
        type=inferred_type,
        suggested_slug=slug,
        content_markdown=proposal.additions_markdown,
        confidence="medium",
        claims=[],
    )


def _normalize_existing_path(raw: str) -> str:
    """Turn a variety of LLM-produced path formats into a page_id.

    Accepts ``concepts/foo``, ``concepts/foo.md``, ``wiki/concepts/foo``,
    ``wiki/concepts/foo.md``, with or without leading slashes. Returns
    the canonical ``category/slug`` form used elsewhere.
    """
    path = raw.strip().lstrip("/")
    if path.startswith("wiki/"):
        path = path[len("wiki/"):]
    if path.endswith(".md"):
        path = path[: -len(".md")]
    return path


def _derive_summary(body: str, max_chars: int = 160) -> str:
    """Extract a summary from the body if none was provided by the LLM.

    Takes the first non-heading paragraph, collapses whitespace,
    truncates to ``max_chars``. Used for manifest + search snippets;
    better than nothing until the prompt explicitly asks for a
    summary field per page.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        if len(stripped) > max_chars:
            return stripped[: max_chars - 3].rstrip() + "..."
        return stripped
    return ""


def _all_chunk_ids_from_synthesis(synthesis: SynthesisResult) -> list[str]:
    """Every chunk_id that contributed to an ingest, deduped in order.

    Used for the source page's frontmatter so a reader can see every
    chunk that was synthesized from the underlying document, not just
    the chunks that happened to produce a create or update proposal.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for proposal in list(synthesis.pages_to_create) + list(synthesis.pages_to_update):
        cid = proposal.chunk_id
        if cid and cid not in seen:
            seen.add(cid)
            merged.append(cid)
    return merged
