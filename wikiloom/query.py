"""Query pipeline: question → retrieval → LLM → structured answer.

Closes the read side of WikiLoom: given a user's natural-language
question, retrieve the most relevant wiki pages via FTS5, inject them
as LLM context, and return a structured answer with source citations.

Design notes
------------
- **Retrieval is FTS5-only for v1.** The SQLite cache's ``pages_fts``
  table covers title + summary. Body search is a fallback grep.
  Semantic (embedding-based) retrieval is a v2 consideration.
- **Uses ``LLMClient.synthesize`` not ``query``.** The response
  schema (``query_response.json``) is structured JSON with
  ``sources_consulted``, ``confidence``, ``suggest_synthesis`` — so
  we need JSON mode. ``LLMClient.query`` returns plain text.
- **Page bodies are read from disk, not the cache.** FTS5 gives us
  page_ids; we read the full markdown body so the LLM has real
  content, not just summaries.
- **Context cap.** To avoid blowing the token budget, we cap the
  number of pages injected (default 5) and truncate each page body
  to ``max_body_chars`` (default 4000 chars ≈ 1000 tokens).
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
MAX_BODY_CHARS = 4000
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
    max_body_chars: int = MAX_BODY_CHARS,
    embedder: Any | None = None,
) -> list[PageContext]:
    """Retrieve relevant pages: FTS5 first, semantic top-up if needed.

    FTS5 keyword search runs first. If it returns fewer than
    ``max_pages`` results and an ``embedder`` is provided, semantic
    similarity fills the remaining slots — so the LLM always gets
    the richest context available without duplicates.
    """
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

    contexts: list[PageContext] = []
    for hit in hits:
        page_id = hit["page_id"]
        page_path = Path(wiki_dir) / f"{page_id}.md"
        if not page_path.exists():
            continue
        fm, body = read_page(page_path)
        title = fm.title if fm else page_id
        if len(body) > max_body_chars:
            body = body[:max_body_chars] + "\n\n[... truncated]"
        contexts.append(PageContext(page_id=page_id, title=title, body=body))
    return contexts


# ----------------------------------------------------------------------
# Context assembly
# ----------------------------------------------------------------------


def _render_user_prompt(question: str, contexts: list[PageContext]) -> str:
    parts = [f"## Your question\n\n{question}"]
    if contexts:
        parts.append("\n## Wiki pages (for reference)\n")
        for ctx in contexts:
            parts.append(f"### {ctx.page_id} — {ctx.title}\n\n{ctx.body}\n")
    else:
        parts.append(
            "\n## Wiki pages\n\n"
            "(No relevant pages found. Answer based on your knowledge "
            "and note that the wiki doesn't cover this topic yet.)\n"
        )
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
