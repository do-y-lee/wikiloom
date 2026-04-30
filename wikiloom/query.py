"""Query pipeline: question → retrieval → LLM → structured answer.

Retrieves relevant pages via FTS5 keyword search (with semantic
embedding fallback), injects them as LLM context, and returns a
structured answer with source citations and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from wikiloom.cache import SQLiteCache
from wikiloom.config import Config
from wikiloom.frontmatter import read_page
from wikiloom.llm import LLMCallMetrics, LLMClient
from wikiloom.llm_errors import LLMProviderError, LLMResponseFormatError

PROMPT_FILENAME = "query.md"
MAX_CONTEXT_PAGES = 5
# Body caps differ by tier so primary pages (where the answer is
# grounded) dominate the prompt while linked pages supply bounded
# supporting context. Starting values; tune from real queries.
MAX_PRIMARY_BODY_CHARS = 20000
MAX_SECONDARY_BODY_CHARS = 5000
# Legacy alias kept so external callers that still pass
# ``max_body_chars`` keep working. New code should use the tier-
# specific constants above.
MAX_BODY_CHARS = MAX_PRIMARY_BODY_CHARS
MAX_LINKED_PAGES = 5
# Page types that never show up as secondary context. ``source``
# pages are provenance summaries (not content the LLM should reason
# from); ``index`` pages are derived navigation.
_SECONDARY_EXCLUDE_TYPES: frozenset[str] = frozenset({"source", "index"})
_VALID_RELEVANCES = frozenset({"primary", "supporting", "tangential"})
_VALID_CONFIDENCES = frozenset({"high", "medium", "low"})


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass
class PageContext:
    page_id: str
    title: str
    body: str
    kind: str = "primary"  # "primary" | "secondary"


@dataclass
class SourceConsulted:
    page_path: str
    relevance: str


@dataclass
class QueryAnswer:
    answer: str
    sources_consulted: list[SourceConsulted]
    confidence: str
    suggest_synthesis: bool
    suggested_followups: list[str]
    metrics: LLMCallMetrics


# ----------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------


def load_query_prompt(project_root: Path) -> str:
    override = Path(project_root) / ".wikiloom" / "prompts" / PROMPT_FILENAME
    if override.exists():
        return override.read_text(encoding="utf-8")
    try:
        pkg = importlib_resources.files("wikiloom") / "templates" / "prompts" / PROMPT_FILENAME
        return pkg.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return "You are a knowledge base assistant. Answer using the provided wiki pages."


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------


def retrieve_context(
    question: str,
    cache: SQLiteCache,
    wiki_dir: Path,
    max_pages: int = MAX_CONTEXT_PAGES,
    max_body_chars: int | None = None,
    embedder: Any | None = None,
    *,
    registry_dir: Path | None = None,
    include_linked: bool = True,
    max_linked: int = MAX_LINKED_PAGES,
    max_primary_body_chars: int = MAX_PRIMARY_BODY_CHARS,
    max_secondary_body_chars: int = MAX_SECONDARY_BODY_CHARS,
) -> list[PageContext]:
    """Retrieve primary pages + (optional) linked secondary pages.

    Primary retrieval is unchanged: FTS5 keyword search first, with
    a semantic top-up from ``embedder`` if FTS5 underdelivers.

    Secondary retrieval (``include_linked=True``, on by default):
    walks outbound wikilinks from every primary page, ranks targets
    by how many primaries reference them (ties broken by recency),
    filters out deprecated, archived, already-primary, and
    source/index types, and caps at ``max_linked`` — so the LLM
    gets directly-matched context first and bounded link-hop
    context after.

    ``max_body_chars``, when provided, pins both tiers to the same
    cap — kept for backward compatibility with older callers.
    Otherwise the tiers use independent caps so primaries dominate
    the prompt while secondaries stay bounded.

    ``registry_dir`` is needed to walk backlinks and filter by
    manifest status. When omitted, it's inferred from ``wiki_dir``
    (``wiki_dir.parent / '_registry'``). Callers that want to
    disable the expansion can pass ``include_linked=False``.
    """
    wiki_dir = Path(wiki_dir)
    if registry_dir is None:
        registry_dir = wiki_dir.parent / "_registry"
    else:
        registry_dir = Path(registry_dir)

    if max_body_chars is not None:
        max_primary_body_chars = max_body_chars
        max_secondary_body_chars = max_body_chars

    # Primary: FTS5 keyword search
    hits = cache.search(question, limit=max_pages)

    # Top-up: if FTS5 didn't fill the quota, use embeddings
    if len(hits) < max_pages and embedder is not None:
        try:
            query_vectors = embedder.embed_texts([question])
            if query_vectors:
                fts_ids = {h["page_id"] for h in hits}
                remaining = max_pages - len(hits)
                semantic_hits = cache.semantic_search(
                    query_vectors[0], limit=remaining + len(fts_ids)
                )
                for sh in semantic_hits:
                    if sh["page_id"] not in fts_ids and len(hits) < max_pages:
                        hits.append(sh)
                        fts_ids.add(sh["page_id"])
        except Exception:
            pass  # embedding failure is non-fatal

    primary_ids: list[str] = []
    contexts: list[PageContext] = []
    for hit in hits:
        page_id = hit["page_id"]
        ctx = _load_page_context(
            page_id, wiki_dir, max_primary_body_chars, kind="primary"
        )
        if ctx is not None:
            contexts.append(ctx)
            primary_ids.append(page_id)

    if include_linked and primary_ids and max_linked > 0:
        secondary_ids = _rank_linked_pages(
            primary_ids=primary_ids,
            cache=cache,
            limit=max_linked,
        )
        for pid in secondary_ids:
            ctx = _load_page_context(
                pid, wiki_dir, max_secondary_body_chars, kind="secondary"
            )
            if ctx is not None:
                contexts.append(ctx)

    return contexts


def _load_page_context(
    page_id: str,
    wiki_dir: Path,
    max_chars: int,
    *,
    kind: str,
) -> PageContext | None:
    """Load a page's frontmatter + body, truncating the body if needed."""
    page_path = wiki_dir / f"{page_id}.md"
    if not page_path.exists():
        return None
    fm, body = read_page(page_path)
    title = fm.title if fm else page_id
    if len(body) > max_chars:
        body = body[:max_chars] + "\n\n[... truncated]"
    return PageContext(page_id=page_id, title=title, body=body, kind=kind)


def _rank_linked_pages(
    *,
    primary_ids: list[str],
    cache: SQLiteCache,
    limit: int,
) -> list[str]:
    """Return up to ``limit`` secondary page_ids ranked by link-hop relevance.

    Process:
    1. Collect outbound wikilink targets from every primary page.
    2. Dedup by target page_id, summing the number of primaries that
       link to each target.
    3. Drop: already-primary, deprecated, archived, missing-from-
       manifest, and types in ``_SECONDARY_EXCLUDE_TYPES`` (source,
       index).
    4. Sort by ``(reference_count desc, modified desc)`` so shared
       concepts rise and the freshest page breaks ties.
    5. Return the top ``limit``.

    Reads edges and page metadata from the SQLite cache instead of
    parsing ``backlinks.json`` and ``manifest.json`` per query — the
    cache mirrors both files via ``sync_from_files``, so the data is
    equivalent and orders of magnitude cheaper to retrieve.
    """
    if limit <= 0 or not primary_ids:
        return []

    primary_set = set(primary_ids)
    edges = cache.get_outbound_edges(primary_ids)

    ref_counts: dict[str, int] = {}
    for edge in edges:
        target = edge["target_page"]
        if target in primary_set:
            continue
        ref_counts[target] = ref_counts.get(target, 0) + 1

    if not ref_counts:
        return []

    target_pages = cache.get_pages(list(ref_counts))

    ranked: list[tuple[int, str, str]] = []
    for pid, count in ref_counts.items():
        page = target_pages.get(pid)
        if page is None:
            continue
        if page.get("status") in ("deprecated", "archived"):
            continue
        if page.get("type") in _SECONDARY_EXCLUDE_TYPES:
            continue
        ranked.append((count, page.get("modified") or "", pid))

    # Sort by (count desc, modified desc) — reference_count is the
    # primary signal; recency breaks ties.
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [pid for _, _, pid in ranked[:limit]]


# ----------------------------------------------------------------------
# Context assembly
# ----------------------------------------------------------------------


def _render_user_prompt(question: str, contexts: list[PageContext]) -> str:
    """Split contexts into primary/secondary sections so the LLM weights them accordingly.

    Primary pages come from FTS5/embedding hits — directly matched to
    the question, full body. Secondary pages come from outbound
    wikilinks on the primaries — supporting context, shorter body.
    Separating them in the prompt gives the LLM explicit hierarchy:
    ground the answer in primaries, lean on secondaries only when
    they reinforce.
    """
    parts = [f"## Your question\n\n{question}"]

    primaries = [c for c in contexts if c.kind == "primary"]
    secondaries = [c for c in contexts if c.kind == "secondary"]

    if not primaries and not secondaries:
        parts.append(
            "\n## Wiki pages\n\n"
            "(No relevant pages found. Answer based on your knowledge "
            "and note that the wiki doesn't cover this topic yet.)\n"
        )
        return "\n".join(parts)

    if primaries:
        parts.append("\n## Primary pages (directly matched)\n")
        for ctx in primaries:
            parts.append(f"### {ctx.page_id} — {ctx.title}\n\n{ctx.body}\n")

    if secondaries:
        parts.append(
            "\n## Linked pages (outbound links from primaries, "
            "for additional context)\n"
        )
        for ctx in secondaries:
            parts.append(f"### {ctx.page_id} — {ctx.title}\n\n{ctx.body}\n")

    return "\n".join(parts)


# ----------------------------------------------------------------------
# Response validation
# ----------------------------------------------------------------------


def validate_query_response(data: Any) -> list[str]:
    """Validate against query_response.json schema. Returns error list."""
    if not isinstance(data, dict):
        return ["response must be a JSON object"]
    errors: list[str] = []
    for fname in ("answer", "sources_consulted", "confidence"):
        if fname not in data:
            errors.append(f"missing required field: {fname}")
    if errors:
        return errors

    if not isinstance(data["answer"], str):
        errors.append("answer must be a string")
    if not isinstance(data["sources_consulted"], list):
        errors.append("sources_consulted must be an array")
    else:
        for i, src in enumerate(data["sources_consulted"]):
            if not isinstance(src, dict):
                errors.append(f"sources_consulted[{i}] must be an object")
                continue
            if "page_path" not in src:
                errors.append(f"sources_consulted[{i}].page_path missing")
            if "relevance" not in src:
                errors.append(f"sources_consulted[{i}].relevance missing")
            elif src["relevance"] not in _VALID_RELEVANCES:
                errors.append(f"sources_consulted[{i}].relevance invalid: {src['relevance']!r}")
    if "confidence" in data and data["confidence"] not in _VALID_CONFIDENCES:
        errors.append(f"confidence invalid: {data['confidence']!r}")

    return errors


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def run_query(
    question: str,
    project_root: Path,
    llm_client: LLMClient,
    cache: SQLiteCache | None = None,
    max_context_pages: int = MAX_CONTEXT_PAGES,
    embedder: Any | None = None,
) -> QueryAnswer:
    """Full query pipeline: retrieve → assemble → LLM → validate → return.

    If ``embedder`` is provided, semantic similarity search is used as
    a fallback when FTS5 finds no keyword matches.

    Raises ``LLMProviderError`` on provider failures,
    ``LLMResponseFormatError`` on unparseable responses, and
    ``ValueError`` on schema validation failures.
    """
    project_root = Path(project_root)
    wiki_dir = project_root / "wiki"
    registry_dir = project_root / "_registry"

    if cache is None:
        cache = SQLiteCache(registry_dir / "wiki.db")

    system_prompt = load_query_prompt(project_root)
    contexts = retrieve_context(
        question, cache, wiki_dir,
        max_pages=max_context_pages,
        embedder=embedder,
    )
    user_prompt = _render_user_prompt(question, contexts)

    result = llm_client.synthesize(system_prompt, user_prompt)

    errors = validate_query_response(result.result)
    if errors:
        raise ValueError(
            f"Query response validation failed: {'; '.join(errors)}"
        )

    data = result.result
    sources = [
        SourceConsulted(
            page_path=src.get("page_path", ""),
            relevance=src.get("relevance", "tangential"),
        )
        for src in data.get("sources_consulted", [])
    ]

    return QueryAnswer(
        answer=data["answer"],
        sources_consulted=sources,
        confidence=data.get("confidence", "low"),
        suggest_synthesis=bool(data.get("suggest_synthesis", False)),
        suggested_followups=list(data.get("suggested_followups", [])),
        metrics=result.metrics,
    )
