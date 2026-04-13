"""Ingest processor — orchestrates extraction, copy-to-raw, budgeting, and chunking.

The full pipeline (per spec) is:

    1. Extract text                          [implemented]
    2. Copy source to raw/                   [implemented]
    3. Plan context budget                   [implemented — placeholder]
    4. Chunk if needed                       [implemented]
    5. LLM synthesize each chunk             [TODO — depends on Component 5 (llm.py)]
    6. Merge chunk results                   [TODO — depends on llm.py]
    7. Write pages                           [TODO — depends on Component 5]
    8. Run linking engine                    [TODO — depends on Component 4 (linker.py)]
    9. Create source summary                 [TODO]
    10. Update manifest and indexes          [partial — registry + backlinks sync]
    10b. Rebuild backlink graph              [implemented]
    11. Git commit                           [implemented]
    12. SQLite sync                          [TODO — depends on Component 12 (cache.py)]
    13. Log event                            [implemented]

This module implements steps 1-4 and 13 today. The remaining steps will
be wired up as their owning components land. The shape of `ingest()` and
its return type are stable so downstream code can be added incrementally.
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
) -> IngestResult:
    """Run the ingest pipeline for a single source.

    Args:
        source: Local file path or URL.
        project_root: Project root directory containing wiki/, raw/, _registry/.
        max_tokens_per_operation: Token budget for the LLM call (used for chunking).

    Returns:
        IngestResult describing what happened. Holds the lock for the
        duration of the operation so concurrent runs are serialized.
    """
    project_root = Path(project_root)

    with FileLock(project_root):
        # 1. Extract
        extractor = router.route(source)
        source_path = Path(source) if not str(source).startswith(("http://", "https://")) else Path(str(source))
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

        # 5-10, 12. LLM / write / link / sync — pending later components.
        result.notes.append(
            "LLM synthesis, page writing, linking, and SQLite sync "
            "are pending Components 4-5 and 12."
        )

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
        commit_hash = git_ops.commit_ingest(
            source_name=source_path.name,
            files=staged,
            stats={
                "pages_created": result.pages_created,
                "pages_updated": result.pages_updated,
            },
        ) or None

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

        return result
