"""Chunk → LLM → structured page proposals.

The synthesis loop is the bridge between extracted chunks and written
wiki pages. It takes each chunk, builds a user prompt that includes
the existing manifest (so the LLM can prefer updates over duplicates),
calls ``LLMClient.synthesize``, validates the response against the
``ingest_response.json`` schema, and accumulates the results into a
structured ``SynthesisResult`` that the page writer can consume.

Design notes
------------
- **Manual schema validator, no dependency.** The schema is ~80 lines
  and the validation rules are: required fields present, types match,
  enum values in range. A handwritten validator is easier to reason
  about than importing ``jsonschema`` for one call site.
- **Per-chunk state updates.** ``IngestState.mark_chunk_done`` /
  ``mark_chunk_failed`` track progress across the run so a crashed
  ingest can (in a future session) resume from the first non-done
  chunk. Today we just use it to record progress for
  ``wikiloom status`` diagnostics.
- **Dedup across chunks.** Multi-chunk documents can redundantly
  propose the same new page from chunks 1 and 2. We dedupe by
  ``(type, suggested_slug)`` — first chunk wins. Future work: feed
  already-proposed pages into later chunks' manifest context so the
  LLM itself can prefer updates. For v1, first-wins is the cheap
  correct default.
- **Errors are isolated per chunk.** A provider error or schema
  failure on chunk 5 of 15 marks that chunk as failed and continues.
  The caller sees ``chunks_failed`` in the result and can decide
  whether to abort or continue. Silently dropping chunks is a real
  hazard — every failure is logged to ``notes``.
- **Source summary: last-chunk-wins.** Multi-chunk docs produce one
  ``source_summary`` per chunk; we keep the one from the last
  successfully-processed chunk since the LLM has the most context
  by then. Revisit in Session D when we see real output.
- **Prompt loading prefers project override.** ``.wikiloom/prompts/
  ingest.md`` in the project takes precedence over the packaged
  template so users can customize the prompt without editing the
  installed package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Callable

from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.ingest.state import IngestState
from wikiloom.llm import LLMClient
from wikiloom.llm_errors import LLMProviderError, LLMResponseFormatError
from wikiloom.registry import Registry

PROMPT_FILENAME = "ingest.md"
MAX_MANIFEST_PAGES_IN_CONTEXT = 100
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
    active = [p for p in registry.pages.values() if p.status == "active"]
    pages = sorted(active, key=lambda p: p.modified or "", reverse=True)[:max_pages]
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


def run_synthesis(
    chunks: list[ExtractedContent],
    chunk_ids: list[str],
    registry: Registry,
    llm_client: LLMClient,
    project_root: Path,
    state: IngestState | None = None,
    progress_callback: ProgressCallback | None = None,
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
    manifest_context = render_manifest_context(registry)

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
    consecutive_provider_failures = 0
    max_consecutive_failures = 3

    for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
        chunk_index = int(chunk.metadata.get("chunk_index", i))
        chunk_total = int(chunk.metadata.get("chunk_total", len(chunks)))

        user_prompt = _render_user_prompt(
            chunk_text=chunk.text,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            manifest_context=manifest_context,
        )

        try:
            llm_result = llm_client.synthesize(system_prompt, user_prompt)
        except LLMProviderError as exc:
            failed += 1
            consecutive_provider_failures += 1
            msg = f"chunk {chunk_index + 1}/{chunk_total}: {exc}"
            notes.append(msg)
            if state is not None:
                state.mark_chunk_failed(chunk_index, str(exc))
            if consecutive_provider_failures >= max_consecutive_failures:
                remaining = chunk_total - chunk_index - 1
                notes.append(
                    f"aborting: {consecutive_provider_failures} consecutive "
                    f"provider errors — skipping {remaining} remaining chunk(s). "
                    f"Fix the issue and re-run with --force."
                )
                break
            continue
        except LLMResponseFormatError as exc:
            failed += 1
            msg = f"chunk {chunk_index + 1}/{chunk_total}: {exc}"
            notes.append(msg)
            if state is not None:
                state.mark_chunk_failed(chunk_index, str(exc))
            continue

        consecutive_provider_failures = 0

        total_in += llm_result.metrics.tokens_in
        total_out += llm_result.metrics.tokens_out
        total_cost += llm_result.metrics.cost_usd

        validation_errors = validate_ingest_response(llm_result.result)
        if validation_errors:
            failed += 1
            short = "; ".join(validation_errors[:3])
            notes.append(
                f"chunk {chunk_index + 1}/{chunk_total}: schema validation failed: {short}"
            )
            if state is not None:
                state.mark_chunk_failed(chunk_index, short)
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
            key = (page_data.get("type", ""), page_data.get("suggested_slug", ""))
            if key in seen_slugs:
                continue
            seen_slugs.add(key)
            pages_to_create.append(
                PageProposal(
                    intent="create",
                    chunk_id=chunk_id,
                    title=page_data.get("title", ""),
                    type=page_data.get("type", ""),
                    suggested_slug=page_data.get("suggested_slug", ""),
                    content_markdown=page_data.get("content_markdown", ""),
                    confidence=page_data.get("confidence", ""),
                    claims=list(page_data.get("claims", []) or []),
                )
            )

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
            state.mark_chunk_done(chunk_index)

        if progress_callback is not None:
            chunk_total_tokens = (
                llm_result.metrics.tokens_in + llm_result.metrics.tokens_out
            )
            progress_callback(
                chunk_index + 1,
                chunk_total,
                chunk_total_tokens,
                llm_result.metrics.cost_usd,
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
