"""Ingest processor — single entry point for adding a source to the wiki.

Runs the full ingest pipeline under ``FileLock`` so concurrent ingests
serialize cleanly. The ``IngestResult`` return shape is stable across
pipeline evolution: new stages fill in ``pages_created`` /
``pages_updated`` without changing the public API, and every stage that
produces files stages them into a single ingest git commit so history
stays atomic.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.events import EventType, append_event, create_event
from wikiloom.git_ops import GitOps
from wikiloom.ingest import router
from wikiloom.ingest.chunker import BudgetPlan, Chunker, plan_budget
from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.locking import FileLock
from wikiloom.registry import Registry
from wikiloom.search import IndexUpdater
from wikiloom.source_catalog import SourceCatalog, SourceEntry, hash_file
from wikiloom.utils import now_iso

# Default mapping from content_type → raw/ subdirectory
RAW_DEST_BY_CONTENT_TYPE: dict[str, str] = {
    "markdown": "articles",
    "pdf": "papers",
    "image": "images",
    "code": "code",
    "office": "articles",
    "web": "articles",
}


@dataclass
class IngestResult:
    """Result of a single ingest operation."""

    source_path: Path
    raw_path: Path | None
    content: ExtractedContent
    chunks: list[ExtractedContent]
    budget: BudgetPlan
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _raw_subdir_for(content: ExtractedContent) -> str:
    return RAW_DEST_BY_CONTENT_TYPE.get(content.content_type, "misc")


def copy_to_raw(source_path: Path, content: ExtractedContent, project_root: Path) -> Path | None:
    """Copy a local source file into raw/<subdir>/.

    Returns the destination path, or None for non-file sources (e.g. URLs).
    """
    if not source_path.exists() or not source_path.is_file():
        return None

    subdir = _raw_subdir_for(content)
    dest_dir = project_root / "raw" / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_path.name
    if dest.resolve() != source_path.resolve():
        shutil.copy2(source_path, dest)
    return dest


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def ingest(
    source: Path | str,
    project_root: Path,
    max_tokens_per_operation: int = 8000,
    force: bool = False,
) -> IngestResult:
    """Run the ingest pipeline for a single source.

    Args:
        source: Local file path or URL.
        project_root: Project root directory containing wiki/, raw/, _registry/.
        max_tokens_per_operation: Token budget for the LLM call (used for chunking).
        force: Re-run the full pipeline even if the source hash is already
            in the catalog. Without this flag, a repeat ingest of an
            identical local file is a cheap no-op that only bumps the
            catalog's ``ingest_count``.

    Returns:
        IngestResult describing what happened. Holds the lock for the
        duration of the operation so concurrent runs are serialized.

    URL sources: URLs are passed through to the WebExtractor and work
    end-to-end for extraction, but they bypass the raw/ copy (nothing
    to copy) and the content-hash dedup (we'd have to fetch before we
    could hash). URL dedup lands alongside Component 20.
    """
    project_root = Path(project_root)
    is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
    source_path = Path(source)

    with FileLock(project_root):
        # 0. Dedup check — only for local files. Cheap hash before we
        # pay the extraction cost (which will matter a lot more when
        # Component 20 wires LLM synthesis into the pipeline).
        catalog: SourceCatalog | None = None
        content_hash: str | None = None
        registry_dir = project_root / "_registry"
        if registry_dir.exists() and not is_url and source_path.is_file():
            catalog = SourceCatalog(registry_dir)
            content_hash = hash_file(source_path)
            if catalog.has(content_hash) and not force:
                catalog.touch(content_hash)
                catalog.save()
                existing = catalog.get(content_hash)
                return _already_ingested_result(
                    source_path, existing, content_hash
                )

        # 1. Extract
        extractor = router.route(source)
        content = extractor.extract(source_path)

        # 2. Copy to raw/
        raw_path = copy_to_raw(source_path, content, project_root)

        # 3. Plan budget
        budget = plan_budget(content, max_tokens_per_operation)

        # 4. Chunk if needed
        if budget.needs_chunking:
            chunks = Chunker().split(content, budget)
        else:
            chunks = [content]

        result = IngestResult(
            source_path=source_path,
            raw_path=raw_path,
            content=content,
            chunks=chunks,
            budget=budget,
        )

        # LLM synthesis, page writing, linking, and SQLite sync land
        # with Component 20 + the LLM enablement pass.

        # 10b. Rebuild backlink graph and sync inbound/outbound counts
        # back to the manifest. Full rebuild is wasteful at scale but
        # correct at every size; incremental comes in a later pass.
        backlinks_path: Path | None = None
        manifest_path: Path | None = None
        registry_dir = project_root / "_registry"
        if registry_dir.exists():
            backlinks = BacklinkRegistry(registry_dir, project_root / "wiki")
            backlinks.rebuild()
            backlinks.save()
            backlinks_path = backlinks.backlinks_path

            registry = Registry(registry_dir)
            counts = backlinks.link_counts()
            manifest_dirty = False
            for page_id, entry in registry.pages.items():
                inbound, outbound = counts.get(page_id, (0, 0))
                if (
                    entry.inbound_link_count != inbound
                    or entry.outbound_link_count != outbound
                ):
                    entry.inbound_link_count = inbound
                    entry.outbound_link_count = outbound
                    manifest_dirty = True
            if manifest_dirty:
                registry.save()
                manifest_path = registry.manifest_path

            # Regenerate indexes *after* manifest save so sub-indexes
            # read the fresh inbound/outbound counts. All written index
            # files are staged into the ingest commit below.
            index_paths: list[Path] = IndexUpdater(
                project_root / "wiki", registry=registry
            ).rebuild_all()
        else:
            index_paths = []

        # 11. Git commit. Empty staging no-ops to HEAD so this is safe
        # during early pipeline development when no pages are written yet.
        git_ops = GitOps(project_root)
        staged: list[Path] = []
        if raw_path is not None:
            staged.append(raw_path)
        for rel in result.pages_created + result.pages_updated:
            staged.append(project_root / rel)
        if backlinks_path is not None:
            staged.append(backlinks_path)
        if manifest_path is not None:
            staged.append(manifest_path)
        staged.extend(index_paths)
        commit_hash = git_ops.commit_ingest(
            source_name=source_path.name,
            files=staged,
            stats={
                "pages_created": result.pages_created,
                "pages_updated": result.pages_updated,
            },
        ) or None

        # 12b. Record / update the source catalog so future ingests of
        # the same content are cheap no-ops. URL sources skip this —
        # see the docstring note on URL dedup.
        if catalog is not None and content_hash is not None:
            existing = catalog.get(content_hash)
            if existing is None:
                size = source_path.stat().st_size if source_path.is_file() else 0
                raw_rel = (
                    str(raw_path.relative_to(project_root))
                    if raw_path is not None
                    else None
                )
                catalog.record(
                    SourceEntry(
                        content_hash=content_hash,
                        name=source_path.name,
                        content_type=content.content_type,
                        size_bytes=size,
                        raw_path=raw_rel,
                        first_ingested_at=now_iso(),
                        last_ingested_at=now_iso(),
                        ingest_count=1,
                        pages_produced=list(
                            result.pages_created + result.pages_updated
                        ),
                    )
                )
            else:
                existing.ingest_count += 1
                existing.last_ingested_at = now_iso()
                for page in result.pages_created + result.pages_updated:
                    if page not in existing.pages_produced:
                        existing.pages_produced.append(page)
            catalog.save()

        # 13. Log event. Written *after* the commit so the event can
        # carry the commit hash. The resulting log.md change is picked up
        # by the next commit (ingest, lint, etc.) — acceptable staleness.
        log_path = project_root / "wiki" / "log.md"
        if log_path.parent.exists():
            event = create_event(
                EventType.INGEST,
                description=source_path.name,
                pages_created=result.pages_created,
                pages_updated=result.pages_updated,
                git_commit_hash=commit_hash,
            )
            append_event(log_path, event)

        # 14. Sync the SQLite query cache (derived, git-ignored).
        # Runs after the commit so `wiki.db` mirrors the just-committed
        # state. `wikiloom rebuild-cache` is the manual recovery path.
        if registry_dir.exists():
            from wikiloom.cache import SQLiteCache

            SQLiteCache(registry_dir / "wiki.db").sync_from_files(
                project_root, staged
            )

        return result


def _already_ingested_result(
    source_path: Path,
    existing: SourceEntry | None,
    content_hash: str,
) -> IngestResult:
    """Build an IngestResult for a dedup hit without running extraction."""
    placeholder_content = ExtractedContent(
        text="",
        metadata={"content_hash": content_hash},
        source_path=source_path,
        content_type=existing.content_type if existing else "",
        extraction_method="dedup-skip",
        token_estimate=0,
    )
    result = IngestResult(
        source_path=source_path,
        raw_path=None,
        content=placeholder_content,
        chunks=[],
        budget=plan_budget(placeholder_content, 8000),
    )
    result.notes.append(
        f"Source already in catalog (hash={content_hash[:12]}, "
        f"ingested {existing.ingest_count if existing else 1}x). "
        f"Pass force=True to re-run the pipeline."
    )
    return result
