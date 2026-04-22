"""Synthesis loop: chunks → LLM → structured page proposals.

Calls LLMClient.synthesize per chunk, validates responses against
the ingest_response.json schema, deduplicates proposals, and tracks
per-chunk progress via IngestState. Aborts after 3 consecutive
failures.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Callable

from rapidfuzz import fuzz, process

from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.ingest.state import IngestState
from wikiloom.llm import LLMClient, SynthesizeResult
from wikiloom.llm_errors import LLMProviderError, LLMResponseFormatError
from wikiloom.registry import Registry

PROMPT_FILENAME = "ingest.md"
MAX_MANIFEST_PAGES_IN_CONTEXT = 100
# Threshold for flagging a proposed new page as a near-duplicate of an
# existing page. rapidfuzz token_sort_ratio, 0–100. Tight by design:
# we want near-identical slug variants (plural/singular, hyphen drift,
# token-drop) to convert to updates, but genuinely distinct concepts
# that happen to share a prefix should still get their own page.
SLUG_COLLISION_THRESHOLD = 95
_VALID_PAGE_TYPES = frozenset({"entity", "concept", "synthesis", "decision"})
_VALID_CONFIDENCES = frozenset({"high", "medium", "low"})


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------


class SynthesisError(Exception):
    """Base class for synthesis failures that the caller should handle."""


class PromptNotFoundError(SynthesisError):
    """Neither the project nor the package prompt template was found."""


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass
class PageProposal:
    """One page's worth of synthesized content, from one chunk.

    Produced by ``run_synthesis`` and consumed by the page writer.
    ``intent`` tells the writer which code path to take:
    ``"create"`` → write a new page or replace on --force,
    ``"update"`` → append to an existing page's auto region.
    """

    intent: str                   # "create" | "update"
    chunk_id: str                 # which chunk produced this proposal

    # Create-path fields
    title: str = ""
    type: str = ""
    suggested_slug: str = ""
    content_markdown: str = ""
    confidence: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)

    # Update-path fields
    existing_path: str | None = None
    additions_markdown: str = ""
    contradictions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SourceSummary:
    """The ``source_summary`` block from the LLM response.

    Used by the page writer to produce the ``wiki/sources/<slug>.md``
    source page that summarizes the ingested document.
    """

    title: str
    one_line: str
    content_markdown: str


@dataclass
class SynthesisResult:
    """Aggregated synthesis output across every chunk of an ingest."""

    source_summary: SourceSummary | None
    pages_to_create: list[PageProposal]
    pages_to_update: list[PageProposal]
    entities_mentioned: list[str]
    concepts_mentioned: list[str]
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    chunks_processed: int
    chunks_failed: int
    notes: list[str]


ProgressCallback = Callable[[int, int, int, float], None]
"""``(chunk_number, chunk_total, tokens_this_chunk, cost_this_chunk)``."""


# ----------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------


def load_prompt(project_root: Path) -> str:
    """Load the ingest prompt template.

    Prefers ``<project>/.wikiloom/prompts/ingest.md`` if present so
    users can customize prompts per-project. Falls back to the
    packaged template in ``wikiloom/templates/prompts/ingest.md``.
    """
    override = Path(project_root) / ".wikiloom" / "prompts" / PROMPT_FILENAME
    if override.exists():
        return override.read_text(encoding="utf-8")
    try:
        pkg = importlib_resources.files("wikiloom") / "templates" / "prompts" / PROMPT_FILENAME
        return pkg.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise PromptNotFoundError(
            f"No ingest prompt found at {override} or in the wikiloom package"
        ) from exc


# ----------------------------------------------------------------------
# Manifest context
# ----------------------------------------------------------------------


def render_manifest_context(
    registry: Registry,
    max_pages: int = MAX_MANIFEST_PAGES_IN_CONTEXT,
) -> str:
    """Render a compact snapshot of existing pages for LLM context.

    Produces a markdown table of the ``max_pages`` most-recently-
    modified pages so the LLM can prefer updates over creating
    duplicates. Capped so the context block stays token-bounded on
    large wikis.
    """
    # Include both active and dormant pages — dormant is informational,
    # not a quarantine. Only deprecated pages are excluded.
    live = [p for p in registry.pages.values() if p.status != "deprecated"]
    pages = sorted(live, key=lambda p: p.modified or "", reverse=True)[:max_pages]
    if not pages:
        return "(the manifest is empty — no existing pages yet)"

    lines = [
        "| page_id | type | title | summary |",
        "|---|---|---|---|",
    ]
    for entry in pages:
        summary = (entry.summary or "").replace("\n", " ").replace("|", "\\|")
        if len(summary) > 80:
            summary = summary[:77] + "..."
        title = entry.title.replace("|", "\\|")
        lines.append(f"| {entry.page_id} | {entry.type} | {title} | {summary} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------


def validate_ingest_response(data: Any) -> list[str]:
    """Validate an LLM response against the ingest_response.json schema.

    Returns a list of human-readable error messages. Empty list means
    the response is valid. Stops at structural failures so downstream
    checks don't produce confusing cascading errors.
    """
    if not isinstance(data, dict):
        return ["response must be a JSON object"]

    errors: list[str] = []
    required_top = [
        "source_summary",
        "pages_to_create",
        "pages_to_update",
        "entities_mentioned",
        "concepts_mentioned",
    ]
    for fname in required_top:
        if fname not in data:
            errors.append(f"missing required field: {fname}")
    if errors:
        return errors  # structural gaps — bail before type-checking

    # source_summary
    ss = data["source_summary"]
    if not isinstance(ss, dict):
        errors.append("source_summary must be an object")
    else:
        for fname in ("title", "one_line", "content_markdown"):
            if fname not in ss:
                errors.append(f"source_summary.{fname} missing")
            elif not isinstance(ss[fname], str):
                errors.append(f"source_summary.{fname} must be a string")

    # pages_to_create
    create_list = data["pages_to_create"]
    if not isinstance(create_list, list):
        errors.append("pages_to_create must be an array")
    else:
        for i, page in enumerate(create_list):
            prefix = f"pages_to_create[{i}]"
            if not isinstance(page, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for fname in ("type", "suggested_slug", "title", "content_markdown", "confidence"):
                if fname not in page:
                    errors.append(f"{prefix}.{fname} missing")
            if "type" in page and page["type"] not in _VALID_PAGE_TYPES:
                errors.append(f"{prefix}.type invalid: {page['type']!r}")
            if "confidence" in page and page["confidence"] not in _VALID_CONFIDENCES:
                errors.append(f"{prefix}.confidence invalid: {page['confidence']!r}")

    # pages_to_update
    update_list = data["pages_to_update"]
    if not isinstance(update_list, list):
        errors.append("pages_to_update must be an array")
    else:
        for i, page in enumerate(update_list):
            prefix = f"pages_to_update[{i}]"
            if not isinstance(page, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for fname in ("existing_path", "additions_markdown"):
                if fname not in page:
                    errors.append(f"{prefix}.{fname} missing")
                elif not isinstance(page[fname], str):
                    errors.append(f"{prefix}.{fname} must be a string")

    # entities_mentioned / concepts_mentioned
    for fname in ("entities_mentioned", "concepts_mentioned"):
        arr = data[fname]
        if not isinstance(arr, list):
            errors.append(f"{fname} must be an array")
            continue
        for i, item in enumerate(arr):
            if not isinstance(item, str):
                errors.append(f"{fname}[{i}] must be a string")

    return errors


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


# LLM emits singular type names ("concept", "entity"); manifest stores
# plural directory names ("concepts/", "entities/"). Mirror of the
# mapping in page_writer.py — keeping a local copy here avoids importing
# page_writer into synthesis, which would create a circular dependency.
_TYPE_TO_DIR = {
    "entity": "entities",
    "concept": "concepts",
    "synthesis": "syntheses",
    "decision": "decisions",
}


def _find_slug_collision(
    proposed_type: str,
    proposed_slug: str,
    existing_page_ids: list[str],
) -> str | None:
    """Return an existing page_id that near-matches the proposed one, if any.

    Scopes the comparison to the same type directory
    (``concepts/``, ``entities/``, ...) because cross-type collisions
    are typically meaningful — a concept and an entity that share a
    name usually refer to distinct things. Uses rapidfuzz
    token_sort_ratio with ``SLUG_COLLISION_THRESHOLD`` as the floor.
    Returns the matched existing id or None.
    """
    type_dir = _TYPE_TO_DIR.get(proposed_type)
    if not type_dir or not proposed_slug or not existing_page_ids:
        return None
    prefix = f"{type_dir}/"
    candidates = [p for p in existing_page_ids if p.startswith(prefix)]
    if not candidates:
        return None
    target = f"{prefix}{proposed_slug}"
    best = process.extractOne(
        target,
        candidates,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=SLUG_COLLISION_THRESHOLD,
    )
    if best is None:
        return None
    matched_id = best[0]
    if matched_id == target:
        return None  # exact identity isn't a "collision" to redirect
    return matched_id


def _format_abort_message(
    *,
    chunk_index: int,
    chunk_total: int,
    processed: int,
    consecutive: int,
    last_exc: Exception,
) -> str:
    """Build a user-facing message describing why a run aborted.

    Calls out quota exhaustion specifically (no point retrying, top up
    instead) versus generic transient errors (network/code bug). Always
    notes how many chunks succeeded and how to resume — today that's
    --force, since IngestState resume isn't wired into the synthesis
    loop yet.
    """
    from wikiloom.llm import _is_quota_exhausted

    remaining = chunk_total - chunk_index - 1
    underlying = getattr(last_exc, "original", last_exc)
    quota = _is_quota_exhausted(underlying)

    lines = [
        f"aborting after {consecutive} consecutive failure(s) at "
        f"chunk {chunk_index + 1}/{chunk_total}.",
        f"  {processed} chunk(s) succeeded and were saved as pages.",
        f"  {remaining} chunk(s) remain unprocessed.",
    ]
    if quota:
        lines.append(
            "  Cause: LLM provider account is out of credits/quota."
        )
        lines.append(
            "  Fix: top up your provider account, then re-run "
            "`wikiloom ingest <file> --force` to retry."
        )
    else:
        lines.append(
            "  Fix: investigate the error above, then re-run "
            "`wikiloom ingest <file> --force` to retry."
        )
    lines.append(
        "  Note: --force re-processes ALL chunks (including ones "
        "already saved) and re-pays for them. Auto-resume from the "
        "checkpoint is planned for a future release."
    )
    return "\n".join(lines)


def _render_user_prompt(
    chunk_text: str,
    chunk_id: str,
    chunk_index: int,
    chunk_total: int,
    manifest_context: str,
) -> str:
    """Build the user prompt for a single chunk's synthesis call."""
    parts = [
        f"## Source chunk {chunk_index + 1} of {chunk_total}",
        f"chunk_id: {chunk_id}",
        "",
        "### Existing pages in the wiki (for deduplication)",
        manifest_context,
        "",
        "### Source text",
        chunk_text,
    ]
    return "\n".join(parts)


def _process_one_chunk(
    *,
    chunk: ExtractedContent,
    chunk_id: str,
    chunk_index: int,
    chunk_total: int,
    llm_client: LLMClient,
    system_prompt: str,
    fallback_manifest_context: str,
    per_chunk_retrieval: bool,
    project_root: Path,
    embedder: Any | None,
    page_context_top_k: int,
) -> tuple[int, SynthesizeResult | None, Exception | None]:
    """Worker body for a single chunk's synthesis call.

    Runs in a ThreadPoolExecutor worker thread. Does prompt rendering
    (including optional per-chunk retrieval) and the LLM call. Returns
    errors rather than raising so the main thread can aggregate
    results from every chunk deterministically without losing any to
    an in-flight exception. The chunk_index is returned so the main
    thread can sort results back into source order.
    """
    try:
        if per_chunk_retrieval:
            from wikiloom.page_context import (
                render_candidates,
                retrieve_candidates_for_chunk,
            )

            candidates = retrieve_candidates_for_chunk(
                chunk_text=chunk.text,
                project_root=project_root,
                embedder=embedder,
                top_k=page_context_top_k,
            )
            chunk_context = render_candidates(candidates)
        else:
            chunk_context = fallback_manifest_context

        user_prompt = _render_user_prompt(
            chunk_text=chunk.text,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            manifest_context=chunk_context,
        )
        llm_result = llm_client.synthesize(system_prompt, user_prompt)
        return (chunk_index, llm_result, None)
    except (LLMProviderError, LLMResponseFormatError) as exc:
        return (chunk_index, None, exc)


def run_synthesis(
    chunks: list[ExtractedContent],
    chunk_ids: list[str],
    registry: Registry,
    llm_client: LLMClient,
    project_root: Path,
    state: IngestState | None = None,
    progress_callback: ProgressCallback | None = None,
    use_page_context: bool = True,
    page_context_top_k: int = 10,
    embedder: Any | None = None,
    max_workers: int = 2,
) -> SynthesisResult:
    """Run LLM synthesis across every chunk and return aggregated results.

    Args:
        chunks: The chunker's output for this ingest.
        chunk_ids: Stable ids returned by ``ChunkStore.persist_chunks``,
            in the same order as ``chunks``.
        registry: Current manifest, used to render existing-pages context.
        llm_client: Provider-agnostic LLM wrapper.
        project_root: Used to locate the project prompt override.
        state: Optional ingest-state checkpoint. If provided, each
            chunk's progress is recorded so failed runs can be
            inspected (and future sessions can resume).
        progress_callback: Optional hook for live progress feedback.
        use_page_context: When True and an ``embedder`` is supplied,
            retrieve the top-K most semantically similar existing pages
            per chunk and inject them into the prompt so the LLM can
            prefer UPDATE over CREATE on overlapping content. Falls
            back to the modified-order snapshot when disabled or when
            the cache has no embeddings yet.
        page_context_top_k: Cap on retrieved candidates per chunk.
        embedder: Optional embedder for per-chunk retrieval. Pass the
            same instance used elsewhere to avoid re-loading the model.

    Returns:
        ``SynthesisResult`` with aggregated page proposals, source
        summary, token + cost totals, and per-chunk failure notes.
    """
    if len(chunks) != len(chunk_ids):
        raise ValueError(
            f"chunks and chunk_ids length mismatch: "
            f"{len(chunks)} vs {len(chunk_ids)}"
        )

    system_prompt = load_prompt(project_root)
    fallback_manifest_context = render_manifest_context(registry)
    per_chunk_retrieval = use_page_context and embedder is not None

    pages_to_create: list[PageProposal] = []
    pages_to_update: list[PageProposal] = []
    entities: set[str] = set()
    concepts: set[str] = set()
    source_summary: SourceSummary | None = None

    total_in = 0
    total_out = 0
    total_cost = 0.0
    processed = 0
    failed = 0
    notes: list[str] = []
    seen_slugs: set[tuple[str, str]] = set()

    # Circuit breaker: stop scheduling new work once this many provider
    # errors have piled up. Cheaper than letting a dead provider burn
    # through every chunk. With parallelism, "consecutive" no longer
    # maps cleanly to wall-clock order, so we use a total-failure
    # threshold instead — equivalent in intent, simpler in behavior.
    max_provider_failures = 3

    # Task inputs, indexed by chunk_index so the post-gather pass can
    # walk results in source order regardless of completion order.
    indexed_inputs: list[tuple[int, int, ExtractedContent, str]] = []
    for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
        ci = int(chunk.metadata.get("chunk_index", i))
        ct = int(chunk.metadata.get("chunk_total", len(chunks)))
        indexed_inputs.append((ci, ct, chunk, chunk_id))

    results: dict[int, tuple[SynthesizeResult | None, Exception | None, str, int]] = {}
    provider_failures = 0
    abort_trigger: tuple[int, int, Exception] | None = None
    completed_count = 0
    effective_workers = max(1, min(max_workers, len(indexed_inputs)))

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_meta = {
            executor.submit(
                _process_one_chunk,
                chunk=chunk,
                chunk_id=chunk_id,
                chunk_index=ci,
                chunk_total=ct,
                llm_client=llm_client,
                system_prompt=system_prompt,
                fallback_manifest_context=fallback_manifest_context,
                per_chunk_retrieval=per_chunk_retrieval,
                project_root=project_root,
                embedder=embedder,
                page_context_top_k=page_context_top_k,
            ): (ci, ct, chunk_id)
            for (ci, ct, chunk, chunk_id) in indexed_inputs
        }

        for future in as_completed(future_to_meta):
            meta_ci, meta_ct, meta_chunk_id = future_to_meta[future]
            try:
                ci_returned, llm_result, err = future.result()
            except Exception as exc:  # unexpected worker crash / cancellation
                ci_returned, llm_result, err = meta_ci, None, exc

            results[ci_returned] = (llm_result, err, meta_chunk_id, meta_ct)
            completed_count += 1

            # Live progress: fire callback as each future finishes, using
            # the chunk's own (1-based) index and token/cost from its own
            # metrics. Cadence matches the old sequential loop.
            if progress_callback is not None and llm_result is not None:
                chunk_total_tokens = (
                    llm_result.metrics.tokens_in + llm_result.metrics.tokens_out
                )
                progress_callback(
                    ci_returned + 1,
                    meta_ct,
                    chunk_total_tokens,
                    llm_result.metrics.cost_usd,
                )

            if isinstance(err, LLMProviderError):
                provider_failures += 1
                if (
                    provider_failures >= max_provider_failures
                    and abort_trigger is None
                ):
                    abort_trigger = (ci_returned, meta_ct, err)
                    for f in future_to_meta:
                        if not f.done():
                            f.cancel()

    # Snapshot of existing manifest page_ids for the slug-collision
    # guard. Mutable — we extend it as this ingest's own creates land,
    # so two chunks proposing near-variants of a new page converge on
    # a single target rather than producing duplicates of each other.
    existing_page_ids: list[str] = list(registry.pages.keys())

    # Post-gather: walk results in chunk_index order so seen_slugs
    # dedup, state writes, and page-proposal ordering are deterministic
    # regardless of which worker finished first.
    for ci in sorted(results.keys()):
        llm_result, err, chunk_id, ct = results[ci]

        if isinstance(err, LLMProviderError) or isinstance(err, LLMResponseFormatError):
            failed += 1
            notes.append(f"chunk {ci + 1}/{ct}: {err}")
            if state is not None:
                state.mark_chunk_failed(ci, str(err))
            continue

        if err is not None:
            # Unexpected crash / cancellation — record and skip.
            failed += 1
            notes.append(f"chunk {ci + 1}/{ct}: unexpected worker error: {err}")
            if state is not None:
                state.mark_chunk_failed(ci, str(err))
            continue

        if llm_result is None:
            failed += 1
            notes.append(f"chunk {ci + 1}/{ct}: no result returned")
            if state is not None:
                state.mark_chunk_failed(ci, "no result returned")
            continue

        total_in += llm_result.metrics.tokens_in
        total_out += llm_result.metrics.tokens_out
        total_cost += llm_result.metrics.cost_usd

        validation_errors = validate_ingest_response(llm_result.result)
        if validation_errors:
            failed += 1
            short = "; ".join(validation_errors[:3])
            notes.append(
                f"chunk {ci + 1}/{ct}: schema validation failed: {short}"
            )
            if state is not None:
                state.mark_chunk_failed(ci, short)
            continue

        data = llm_result.result

        ss = data.get("source_summary") or {}
        if ss:
            source_summary = SourceSummary(
                title=ss.get("title", ""),
                one_line=ss.get("one_line", ""),
                content_markdown=ss.get("content_markdown", ""),
            )

        for page_data in data.get("pages_to_create", []) or []:
            ptype = page_data.get("type", "")
            slug = page_data.get("suggested_slug", "")
            key = (ptype, slug)
            if key in seen_slugs:
                continue
            seen_slugs.add(key)

            # Slug-collision guard: if a proposed new page's page_id
            # fuzzy-matches an existing one, divert to update instead.
            # Catches "account-closure-procedure" when the wiki already
            # has "account-closure-procedures", etc. The LLM can't
            # always see the existing page through semantic retrieval
            # alone, so this is a deterministic last-mile check.
            collision = _find_slug_collision(
                ptype, slug, existing_page_ids
            )
            if collision is not None:
                pages_to_update.append(
                    PageProposal(
                        intent="update",
                        chunk_id=chunk_id,
                        existing_path=collision,
                        additions_markdown=page_data.get(
                            "content_markdown", ""
                        ),
                        contradictions=[],
                    )
                )
                type_dir = _TYPE_TO_DIR.get(ptype, ptype)
                notes.append(
                    f"chunk {ci + 1}/{ct}: slug collision redirected "
                    f"{type_dir}/{slug} → {collision}"
                )
                continue

            pages_to_create.append(
                PageProposal(
                    intent="create",
                    chunk_id=chunk_id,
                    title=page_data.get("title", ""),
                    type=ptype,
                    suggested_slug=slug,
                    content_markdown=page_data.get("content_markdown", ""),
                    confidence=page_data.get("confidence", ""),
                    claims=list(page_data.get("claims", []) or []),
                )
            )
            # Track the just-created page_id so a later chunk in this
            # same ingest can collide with it and redirect, instead of
            # racing to create a second near-variant.
            type_dir = _TYPE_TO_DIR.get(ptype)
            if type_dir and slug:
                existing_page_ids.append(f"{type_dir}/{slug}")

        for page_data in data.get("pages_to_update", []) or []:
            pages_to_update.append(
                PageProposal(
                    intent="update",
                    chunk_id=chunk_id,
                    existing_path=page_data.get("existing_path"),
                    additions_markdown=page_data.get("additions_markdown", ""),
                    contradictions=list(page_data.get("contradictions", []) or []),
                )
            )

        for name in data.get("entities_mentioned", []) or []:
            if isinstance(name, str):
                entities.add(name)
        for name in data.get("concepts_mentioned", []) or []:
            if isinstance(name, str):
                concepts.add(name)

        processed += 1
        if state is not None:
            state.mark_chunk_done(ci)

    if abort_trigger is not None:
        abort_ci, abort_ct, abort_exc = abort_trigger
        notes.append(
            _format_abort_message(
                chunk_index=abort_ci,
                chunk_total=abort_ct,
                processed=processed,
                consecutive=provider_failures,
                last_exc=abort_exc,
            )
        )

    return SynthesisResult(
        source_summary=source_summary,
        pages_to_create=pages_to_create,
        pages_to_update=pages_to_update,
        entities_mentioned=sorted(entities),
        concepts_mentioned=sorted(concepts),
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        total_cost_usd=total_cost,
        chunks_processed=processed,
        chunks_failed=failed,
        notes=notes,
    )
