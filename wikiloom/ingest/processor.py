"""Ingest processor — single entry point for adding a source to the wiki.

Runs extract → chunk → synthesize → write pages → link → commit
under FileLock in one atomic pipeline.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from wikiloom.cli_output import (
    check as _check,
    dim as _dim,
    done_summary,
    format_tokens as _format_tokens,
)

from wikiloom.backlinks import BacklinkRegistry
from wikiloom.chunk_store import ChunkStore
from wikiloom.config import Config, IngestConfig
from wikiloom.events import EventType, append_event, create_event
from wikiloom.git_ops import GitOps
from wikiloom.ingest import router
from wikiloom.ingest.chunker import BudgetPlan, Chunker, plan_budget
from wikiloom.ingest.errors import (
    BudgetExceededError,
    EmptyExtractionError,
    FileTooLargeError,
)
from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.ingest.page_writer import PageWriter
from wikiloom.ingest.state import ChunkState, IngestState
from wikiloom.llm import LLMClient, estimate_cost
from wikiloom.locking import FileLock
from wikiloom.registry import Registry
from wikiloom.search import IndexUpdater
from wikiloom.source_catalog import SourceCatalog, SourceEntry, hash_file
from wikiloom.synthesis import run_synthesis
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
    # Synthesis metrics, populated after the LLM loop runs
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0


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


def _guard_file_size(source_path: Path, ingest_cfg: IngestConfig) -> None:
    """Fail fast when a local source exceeds the configured size cap."""
    if ingest_cfg.max_file_size_mb <= 0:
        return
    if not source_path.exists() or not source_path.is_file():
        return
    size_bytes = source_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > ingest_cfg.max_file_size_mb:
        raise FileTooLargeError(
            path=str(source_path),
            size_mb=size_mb,
            limit_mb=ingest_cfg.max_file_size_mb,
        )


def _guard_empty_extraction(
    content: ExtractedContent, ingest_cfg: IngestConfig
) -> None:
    """Fail fast when extraction returned too little text to be useful.

    Skips extractors that always emit a placeholder string (image,
    code) since their byte count reflects the wrapper, not real
    content. A scanned PDF or blank document gets caught here before
    the LLM loop wastes tokens on empty input.
    """
    if content.content_type in {"image", "code"}:
        return
    chars = len(content.text.strip())
    if chars < ingest_cfg.min_extracted_chars:
        raise EmptyExtractionError(
            path=str(content.source_path) if content.source_path else "<unknown>",
            content_type=content.content_type,
            extracted_chars=chars,
        )


def _load_ingest_config(project_root: Path) -> IngestConfig:
    """Read ``[ingest]`` from wikiloom.toml, falling back to defaults.

    A missing config file is fine — the defaults are safe. A malformed
    config surfaces as a ``tomllib.TOMLDecodeError`` to the caller;
    the CLI layer converts that to a friendly ``ClickException``.
    """
    try:
        return Config.load(project_root).ingest
    except FileNotFoundError:
        return IngestConfig()


def _load_full_config(project_root: Path) -> Config:
    """Load the full Config, falling back to defaults when no file exists."""
    try:
        return Config.load(project_root)
    except FileNotFoundError:
        return Config(project_root=project_root)


def _preflight_budget_check(
    chunks: list[ExtractedContent],
    cfg: Config,
) -> None:
    """Estimate the synthesis run's cost and refuse if it exceeds the monthly budget.

    Rough-and-conservative: sums per-chunk token_estimates as input
    tokens, assumes output tokens ~= half of input, then asks
    ``llm.estimate_cost`` for a USD figure. We intentionally over-
    estimate rather than under-estimate: a false positive is a user
    raising the budget knob; a false negative is money actually spent.
    """
    if not cfg.ingest.enable_budget_check:
        return
    if cfg.llm.monthly_budget_usd <= 0:
        return
    tokens_in = sum(max(1, c.token_estimate) for c in chunks)
    tokens_out = max(1, tokens_in // 2)
    estimated = estimate_cost(tokens_in, tokens_out, cfg.llm.for_ingest())
    if estimated > cfg.llm.monthly_budget_usd:
        raise BudgetExceededError(
            estimated_usd=estimated,
            budget_usd=cfg.llm.monthly_budget_usd,
        )


def ingest(
    source: Path | str,
    project_root: Path,
    max_tokens_per_operation: int = 8000,
    force: bool = False,
    use_page_context: bool | None = None,
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
        use_page_context: Overrides the project's ``[ingest] use_page_context``
            config value for this run. None leaves the config setting in
            place; True/False forces the behavior regardless of config.

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

    # Read ingest config before acquiring the lock; it's pure file I/O
    # and lets the boundary guards consult user settings.
    ingest_cfg = _load_ingest_config(project_root)

    # Guard 1: file-size cap. URLs skip this — nothing on disk to stat.
    if not is_url:
        _guard_file_size(source_path, ingest_cfg)

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
        click.echo(f"Extracting {source_path.name}...")
        extractor = router.route(source)
        content = extractor.extract(source_path)

        # Guard 2: empty-extraction. Raised *after* we've routed to an
        # extractor so the error message can cite the content_type.
        _guard_empty_extraction(content, ingest_cfg)

        click.echo(
            f"  {content.content_type}, "
            f"{content.token_estimate:,} tokens estimated"
        )

        # 2. Copy to raw/
        raw_path = copy_to_raw(source_path, content, project_root)
        if raw_path is not None:
            click.echo(f"  {raw_path.relative_to(project_root)}")

        # 3. Plan budget
        budget = plan_budget(content, max_tokens_per_operation)

        # 4. Chunk if needed
        if budget.needs_chunking:
            chunks = Chunker().split(content, budget)
        else:
            chunks = [content]
        click.echo(f"  {len(chunks)} chunk(s)")

        # 4b. Write the resume checkpoint. Records the chunk plan so a
        # crash mid-synthesis (once Component 20 lands) can pick up
        # where it left off. Cleared on successful completion below.
        ingest_state: IngestState | None = None
        if registry_dir.exists():
            source_key = content_hash or (source if is_url else str(source_path))
            chunk_states = [
                ChunkState(
                    index=int(chunk.metadata.get("chunk_index", i)),
                    total=int(chunk.metadata.get("chunk_total", len(chunks))),
                    token_estimate=chunk.token_estimate,
                )
                for i, chunk in enumerate(chunks)
            ]
            ingest_state = IngestState.begin(
                registry_dir=registry_dir,
                source_key=source_key,
                source_name=source_path.name if not is_url else str(source),
                content_type=content.content_type,
                chunks=chunk_states,
            )

        result = IngestResult(
            source_path=source_path,
            raw_path=raw_path,
            content=content,
            chunks=chunks,
            budget=budget,
        )

        # 5. LLM synthesis + page writing. File sources only for now;
        # URL sources bypass the chunk store until NOTES.local.md
        # item F lands. An empty synthesis run (all chunks failed)
        # still produces a commit for backlinks/manifest/indexes but
        # yields zero new pages.
        registry: Registry | None = None
        synthesis_written: list[Path] = []
        if (
            registry_dir.exists()
            and not is_url
            and content_hash is not None
            and source_path.is_file()
        ):
            full_cfg = _load_full_config(project_root)

            # 5a. Pre-flight budget check — refuse before the LLM loop
            # if the estimated cost would breach the monthly budget.
            # Silent on the happy path; raises if the estimate exceeds.
            _preflight_budget_check(chunks, full_cfg)

            # 5b. Persist chunks so their text is queryable via
            # `wikiloom source <chunk_id>` after the ingest commits.
            chunk_store = ChunkStore(registry_dir / "wiki.db")
            stored_chunks = chunk_store.persist_chunks(content_hash, chunks)
            chunk_ids = [s.chunk_id for s in stored_chunks]

            # 5c. Synthesis loop. The registry is loaded once here and
            # reused in step 10b for manifest sync — the page writer
            # mutates it in-place as pages are registered.
            registry = Registry(registry_dir)
            ingest_model = full_cfg.llm.for_ingest()
            llm_client = LLMClient(full_cfg, model=ingest_model)

            check_mark = _check()

            def _on_chunk_done(n: int, total: int, tokens: int, cost: float) -> None:
                # Columns: ✓  N/TOTAL   NN,NNN tok   $0.0045
                click.echo(
                    f"  {check_mark} {n}/{total}   "
                    f"{tokens:,} tok   ${cost:.4f}"
                )

            effective_workers = max(
                1, min(full_cfg.ingest.max_workers, len(chunks))
            )
            click.echo("")
            click.echo(
                f"Synthesizing via {click.style(ingest_model, fg='cyan')}  "
                f"{_dim(f'(max_workers={effective_workers})')}"
            )
            click.echo("")
            _synth_start = time.monotonic()
            effective_page_context = (
                full_cfg.ingest.use_page_context
                if use_page_context is None
                else use_page_context
            )
            synthesis_embedder = None
            if effective_page_context:
                from wikiloom.embeddings import load_embedder
                synthesis_embedder = load_embedder(project_root)
            synthesis = run_synthesis(
                chunks=chunks,
                chunk_ids=chunk_ids,
                registry=registry,
                llm_client=llm_client,
                project_root=project_root,
                state=ingest_state,
                progress_callback=_on_chunk_done,
                use_page_context=effective_page_context,
                page_context_top_k=full_cfg.ingest.page_context_top_k,
                embedder=synthesis_embedder,
                max_workers=full_cfg.ingest.max_workers,
            )

            result.total_tokens_in = synthesis.total_tokens_in
            result.total_tokens_out = synthesis.total_tokens_out
            result.total_cost_usd = synthesis.total_cost_usd
            result.notes.extend(synthesis.notes)

            # End-of-synthesis summary: one line, totals rolled up.
            _total_tok = synthesis.total_tokens_in + synthesis.total_tokens_out
            click.echo("")
            click.echo(
                done_summary(
                    [
                        f"{synthesis.chunks_processed}/{len(chunks)} chunks",
                        f"{_format_tokens(_total_tok)} tok",
                        f"${synthesis.total_cost_usd:.3f}",
                    ],
                    elapsed=time.monotonic() - _synth_start,
                )
            )
            click.echo("")

            # 5d. Write pages. Builds a source_entry on the fly so the
            # writer can emit the source summary page even on the very
            # first ingest of a file (before the catalog has recorded it).
            catalog_entry: SourceEntry | None = (
                catalog.get(content_hash) if catalog is not None else None
            )
            if catalog_entry is None and content_hash is not None:
                size_bytes = (
                    source_path.stat().st_size if source_path.is_file() else 0
                )
                raw_rel = (
                    str(raw_path.relative_to(project_root))
                    if raw_path is not None
                    else None
                )
                catalog_entry = SourceEntry(
                    content_hash=content_hash,
                    name=source_path.name,
                    content_type=content.content_type,
                    size_bytes=size_bytes,
                    raw_path=raw_rel,
                    first_ingested_at=now_iso(),
                    last_ingested_at=now_iso(),
                )

            click.echo("Writing pages...")
            writer = PageWriter(project_root, registry, force=force)
            write_result = writer.write(synthesis, source_entry=catalog_entry)

            result.pages_created.extend(write_result.created_page_ids)
            result.pages_updated.extend(write_result.updated_page_ids)
            result.notes.extend(write_result.notes)
            synthesis_written = (
                list(write_result.created_paths) + list(write_result.updated_paths)
            )
            click.echo(
                f"  {len(write_result.created_page_ids)} created, "
                f"{len(write_result.updated_page_ids)} updated"
            )

            # 6. Deterministic linking. Runs spaCy NER + rapidfuzz on
            # every just-written page, inserts [[wikilinks]], writes
            # stubs for unresolved entities, defers low-confidence
            # candidates to pending.json. Uses the same registry
            # instance so stubs are visible to step 10b's backlinks
            # rebuild. The linker modifies page files in-place — the
            # paths in synthesis_written are updated on disk.
            if synthesis_written:
                click.echo("")
                click.echo("Linking...")
                from wikiloom.linker import LinkingEngine

                linker = LinkingEngine(registry, config=full_cfg.linking)
                linked_pages = linker.link_all(synthesis_written)
                click.echo(f"  {len(linked_pages)} page(s) linked")

                # Stubs created by the linker are new files that need
                # to be staged in the commit. Scan the wiki dir for
                # stub-status pages the linker just registered.
                for page_id, entry in registry.pages.items():
                    if entry.status == "stub":
                        stub_path = project_root / "wiki" / f"{page_id}.md"
                        if stub_path.exists() and stub_path not in synthesis_written:
                            synthesis_written.append(stub_path)
                            result.pages_created.append(page_id)

                # pending.json gets staged too so the commit is atomic.
                pending_path = registry_dir / "pending.json"
                if pending_path.exists() and pending_path not in synthesis_written:
                    synthesis_written.append(pending_path)

        # 10b. Rebuild backlink graph and sync inbound/outbound counts
        # back to the manifest. Full rebuild is wasteful at scale but
        # correct at every size; incremental comes in a later pass.
        backlinks_path: Path | None = None
        manifest_path: Path | None = None
        if registry_dir.exists():
            backlinks = BacklinkRegistry(registry_dir, project_root / "wiki")
            backlinks.rebuild()
            backlinks.save()
            backlinks_path = backlinks.backlinks_path

            # Reuse the registry loaded in step 5c if the synthesis
            # block ran; otherwise load fresh. Either way, mutate in
            # place so the backlink counts + synthesized page entries
            # end up in a single save below.
            if registry is None:
                registry = Registry(registry_dir)
            counts = backlinks.link_counts()
            for page_id, entry in registry.pages.items():
                inbound, outbound = counts.get(page_id, (0, 0))
                entry.inbound_link_count = inbound
                entry.outbound_link_count = outbound

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
        # when synthesis produced no pages (e.g., every chunk failed).
        click.echo("")
        click.echo("Committing...")
        git_ops = GitOps(project_root)
        staged: list[Path] = []
        if raw_path is not None:
            staged.append(raw_path)
        # Stage every page written by the synthesis block (includes
        # source page + created + updated). Deduped against
        # pages_created / pages_updated which hold page_ids, not paths.
        for page_path in synthesis_written:
            if page_path not in staged:
                staged.append(page_path)
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
        if commit_hash:
            click.echo(
                f"  ingest: {source_path.name} "
                f"{_dim(f'({commit_hash[:7]})')}"
            )

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
        # Token + cost fields finally carry real numbers now that C20's
        # LLMClient populates LLMCallMetrics on every synthesize call.
        log_path = project_root / "wiki" / "log.md"
        if log_path.parent.exists():
            event = create_event(
                EventType.INGEST,
                description=source_path.name,
                pages_created=result.pages_created,
                pages_updated=result.pages_updated,
                git_commit_hash=commit_hash,
                tokens_used=result.total_tokens_in + result.total_tokens_out,
                cost_usd=result.total_cost_usd,
            )
            append_event(log_path, event)

        # 14. Sync the SQLite query cache (derived, git-ignored).
        # Runs after the commit so `wiki.db` mirrors the just-committed
        # state. `wikiloom rebuild-cache` is the manual recovery path.
        if registry_dir.exists():
            from wikiloom.cache import SQLiteCache
            from wikiloom.embeddings import load_embedder

            _embedder = load_embedder(project_root)
            if _embedder is None and full_cfg.embeddings.enabled:
                click.echo(
                    click.style(
                        "Warning: embeddings enabled in config but the "
                        "embedder failed to load. Rows will carry NULL "
                        "embeddings. Run `wikiloom rebuild-cache` after "
                        "this ingest, or check that your embedding "
                        "backend is installed (e.g. `pip install "
                        "fastembed`).",
                        fg="yellow",
                    ),
                    err=True,
                )
            SQLiteCache(registry_dir / "wiki.db").sync_from_files(
                project_root, staged, embedder=_embedder
            )

        # 15. Clear the resume checkpoint. Reaching this line means the
        # pipeline succeeded end-to-end, so the leftover state file is
        # no longer useful. Any crash before this point leaves the file
        # behind for the next run to inspect.
        if ingest_state is not None:
            ingest_state.clear()

        # 16. Commit the post-commit tail (log.md + sources.json). These
        # were written *after* step 11's ingest commit so the log event
        # could carry that commit's hash. Leaving them uncommitted makes
        # the working tree look dirty after every successful ingest, so
        # we land them in a small follow-up commit under the same
        # ``ingest:`` prefix. ``commit()`` no-ops if nothing changed.
        tail_files: list[Path] = []
        if log_path.parent.exists() and log_path.exists():
            tail_files.append(log_path)
        sources_path = registry_dir / "sources.json"
        if sources_path.exists():
            tail_files.append(sources_path)
        if tail_files:
            git_ops.commit(
                tail_files,
                f"ingest: log + catalog for {source_path.name}",
            )

        # 17. Post-ingest auto-merge (opt-in). Runs after the tail
        # commit so merges land in their own follow-up commit that
        # ``git revert`` can undo selectively. Scoped to pages this
        # ingest created or updated so we never merge far-field pairs
        # the user may not want touched.
        if full_cfg.ingest.post_merge != "off" and (
            result.pages_created or result.pages_updated
        ):
            _run_post_ingest_merge(
                project_root=project_root,
                full_cfg=full_cfg,
                result=result,
                git_ops=git_ops,
            )

        return result


def _run_post_ingest_merge(
    *,
    project_root: Path,
    full_cfg: Config,
    result: IngestResult,
    git_ops: GitOps,
) -> None:
    """Post-ingest merge pass (preview or safe mode).

    Scoped to pairs where at least one side is in this ingest's
    ``pages_created`` or ``pages_updated`` — avoids touching far-field
    pairs the user may have intentionally kept distinct. In preview
    mode, candidates are listed and the function returns. In safe
    mode, candidates are merged in a single batched commit, and if
    ``auto_relink`` is enabled and any merges applied, a follow-up
    relink pass runs so winners pick up new inbound links from any
    newly-unified aliases.
    """
    from wikiloom.duplicates import find_duplicates, suggest_winner

    touched = set(result.pages_created) | set(result.pages_updated)
    all_pairs = find_duplicates(project_root)
    scoped = [
        p for p in all_pairs if p.page_a in touched or p.page_b in touched
    ]
    plan: list[tuple[Any, Any]] = []
    for pair in scoped:
        sug = suggest_winner(pair)
        if sug.is_safe_to_auto:
            plan.append((pair, sug))

    if not plan:
        return

    mode = full_cfg.ingest.post_merge
    click.echo(f"\nPost-ingest merge candidates ({len(plan)} pair(s)):")
    for pair, sug in plan:
        emb = (
            f"{pair.embedding_score:.2f}"
            if pair.embedding_score >= 0
            else "n/a"
        )
        click.echo(
            f"  {sug.loser_page_id}  →  {sug.winner_page_id}  "
            f"(slug {pair.slug_score:.0f}%, emb {emb}, {sug.reason})"
        )

    if mode == "preview":
        click.echo(
            'Preview mode. Set ingest.post_merge = "safe" in wikiloom.toml '
            "to auto-merge these."
        )
        return

    from wikiloom.merge import merge_pages

    applied: list[tuple[str, str]] = []
    for pair, sug in plan:
        try:
            merge_pages(project_root, sug.winner_page_id, sug.loser_page_id)
            applied.append((sug.winner_page_id, sug.loser_page_id))
        except ValueError as exc:
            click.echo(f"  ✗ skipped {sug.loser_page_id}: {exc}")

    if not applied:
        return

    # One cache sync + one commit for the whole batch.
    registry_dir = project_root / "_registry"
    from wikiloom.cache import SQLiteCache
    from wikiloom.embeddings import load_embedder

    wiki_dir = project_root / "wiki"
    touched_paths: list[Path] = []
    for winner, loser in applied:
        touched_paths.append(wiki_dir / f"{winner}.md")
        touched_paths.append(wiki_dir / f"{loser}.md")
    # Rebuild indexes so archived losers disappear from category/root
    # indexes even when auto_relink is false (post-merge relink is the
    # other path that would otherwise cover this).
    registry = Registry(registry_dir)
    index_paths = IndexUpdater(wiki_dir, registry=registry).rebuild_all()
    SQLiteCache(registry_dir / "wiki.db").sync_from_files(
        project_root,
        changed_files=touched_paths + list(index_paths),
        embedder=load_embedder(project_root),
    )

    body = "\n".join(f"  {loser} → {winner}" for winner, loser in applied)
    message = (
        f"merge: post-ingest auto-merged {len(applied)} pair(s)\n\n{body}"
    )
    for scope in ("wiki", "_registry"):
        if (project_root / scope).exists():
            git_ops.repo.git.add("-A", "--", scope)
    git_ops.commit([], message)

    click.echo(f"Merged {len(applied)} pair(s).")

    if full_cfg.ingest.auto_relink:
        _run_post_merge_relink(project_root, full_cfg, git_ops)


def _run_post_merge_relink(
    project_root: Path, full_cfg: Config, git_ops: GitOps
) -> None:
    """Full-wiki relink after post-ingest merges.

    Merges can add aliases to winners (from losers' titles), and
    other pages may now match those aliases. A relink re-scans
    everything so the freshly-unified aliases produce new inbound
    links where appropriate. Runs a single follow-up commit only if
    any pages actually gained links.
    """
    import time as _time

    from wikiloom.linker import LinkingEngine

    wiki_dir = project_root / "wiki"
    all_pages = sorted(
        p for p in wiki_dir.rglob("*.md")
        if p.name != "index.md"
        and p.name != "log.md"
        and "archive" not in p.parts
    )
    if not all_pages:
        return

    total = len(all_pages)
    click.echo(f"Re-linking {total} page(s)...")
    start = _time.monotonic()
    step = max(25, max(1, total // 10))

    def _progress(done: int, total_inner: int) -> None:
        if done == total_inner or done % step == 0:
            click.echo(f"  {done}/{total_inner} pages linked...")

    registry = Registry(project_root / "_registry")
    linker = LinkingEngine(registry, config=full_cfg.linking)
    linked = linker.link_all(all_pages, progress=_progress)

    backlinks = BacklinkRegistry(project_root / "_registry")
    backlinks.rebuild()
    backlinks.save()
    IndexUpdater(wiki_dir, registry=registry).rebuild_all()

    from wikiloom.cache import SQLiteCache
    from wikiloom.embeddings import load_embedder

    SQLiteCache(project_root / "_registry" / "wiki.db").sync_from_files(
        project_root, embedder=load_embedder(project_root)
    )

    elapsed = _time.monotonic() - start
    click.echo(
        f"Re-linked {total} page(s) in {elapsed:.1f}s "
        f"({len(linked)} updated). Backlinks rebuilt."
    )

    if linked:
        for scope in ("wiki", "_registry"):
            if (project_root / scope).exists():
                git_ops.repo.git.add("-A", "--", scope)
        git_ops.commit(
            [], f"relink: updated wikilinks across {len(linked)} page(s)"
        )


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
