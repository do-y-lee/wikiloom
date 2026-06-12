"""FastMCP server exposing WikiLoom's retrieval surface as 6 agent-callable tools.

The tools follow a 3-layer pattern:

- **Cheap routers** (small payloads): ``search_pages``, ``search_chunks``.
- **Expensive payloads** (full text): ``get_pages``, ``get_chunks``.
- **One-shot orchestrator**: ``get_context`` (page router → token-budgeted chunks).
- **Graph hop**: ``get_outbound_links``.

Cache and embedder are loaded once at startup and closed over by every tool
so we never reopen connections or reload the embedder per call.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from wikiloom.cache import SQLiteCache
from wikiloom.chunk_store import ChunkStore
from wikiloom.context import get_context as wikiloom_get_context
from wikiloom.embeddings import load_embedder
from wikiloom.frontmatter import read_page
from wikiloom.mcp.models import (
    BacklinkOut,
    CitationOut,
    ContextResultOut,
    OutboundLinkOut,
    PageBodyOut,
    PageHitOut,
    StoredChunkOut,
    short_summary,
)
from wikiloom.retrieval import search_chunks as wikiloom_search_chunks

# Pinned internal defaults for the get_context orchestrator. The MCP
# surface intentionally hides these from the agent — the 3-layer pattern
# says drop down to search_pages/search_chunks for finer control.
_DEFAULT_TOP_PAGES = 5
_DEFAULT_K = 20
_DEFAULT_BUDGET = 2000


def build_server(
    cache: SQLiteCache, embedder: Any, project: Path
) -> FastMCP:
    """Wire the 6 tools over a given cache + embedder. Testable in isolation."""
    mcp = FastMCP("wikiloom")
    store = ChunkStore(cache)
    wiki_dir = project / "wiki"

    @mcp.tool()
    def search_pages(query: str, k: int = 5) -> list[PageHitOut]:
        """Cheap router. Returns up to k synthesized pages most similar to the goal.

        Each result includes a short (~30-token) summary preview and a
        cosine similarity score — NOT the full page body. Use when the
        goal is conceptual ("how does auth work?"). Call this before
        get_pages to decide which full bodies are worth pulling.

        If results don't fit your goal: refine the query and call again,
        pivot to search_chunks for literal/keyword hits, or follow links
        from a near-match page via get_outbound_links.
        """
        query_vecs = embedder.embed_texts([query])
        if not query_vecs:
            return []
        hits = cache.semantic_search(
            list(query_vecs[0]),
            limit=k,
            exclude_statuses=("deprecated",),
        )
        return [
            PageHitOut(
                page_id=h["page_id"],
                type=h["type"],
                title=h["title"],
                summary=short_summary(h.get("summary") or ""),
                similarity=h["similarity"],
            )
            for h in hits
        ]

    @mcp.tool()
    def search_chunks(query: str, k: int = 10) -> list[CitationOut]:
        """Cheap router (chunk-level). Returns up to k chunks ranked by BM25 + vector.

        Each result includes a ~200-char snippet and provenance — NOT the
        full chunk text. Use when the goal has a literal phrase, quoted
        string, or proper noun unlikely to appear in page summaries.
        Call this before get_chunks to decide which full texts to pull.

        If results don't fit: refine the query, or back off to
        search_pages for broader topic routing.
        """
        cites = wikiloom_search_chunks(cache, embedder, query, k=k)
        return [CitationOut(**asdict(c)) for c in cites]

    @mcp.tool()
    def get_pages(ids: list[str]) -> list[PageBodyOut]:
        """Expensive payload. Returns full body markdown for each page id.

        Call after search_pages or get_context when the truncated
        summary isn't enough. Missing ids are silently omitted from
        the result. Cost scales with the number and size of pages.
        """
        out: list[PageBodyOut] = []
        for page_id in ids:
            path = wiki_dir / f"{page_id}.md"
            if not path.exists():
                continue
            fm, body = read_page(path)
            if fm is None:
                continue
            out.append(
                PageBodyOut(
                    page_id=page_id,
                    title=fm.title or page_id,
                    type=fm.type or "",
                    status=fm.status or "",
                    summary=fm.summary or "",
                    body=body,
                )
            )
        return out

    @mcp.tool()
    def get_chunks(ids: list[str]) -> list[StoredChunkOut]:
        """Expensive payload (chunk-level). Returns full text for each chunk id.

        Call after search_chunks when the snippet isn't enough.
        Provenance (page_id, source_path, parent_heading) you already
        have from the prior search_chunks call; this tool returns the
        text and a token estimate so you can budget downstream calls.
        Missing ids are silently omitted.
        """
        rows = store.get_chunks(ids)
        out: list[StoredChunkOut] = []
        for cid in ids:
            row = rows.get(cid)
            if row is None:
                continue
            out.append(
                StoredChunkOut(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    token_estimate=row.token_estimate,
                    content_type=row.content_type,
                )
            )
        return out

    @mcp.tool()
    def get_context(goal: str, budget: int = _DEFAULT_BUDGET) -> ContextResultOut:
        """One-shot orchestrator. Embeds the goal, routes to top pages, packs chunks.

        Returns both the routed pages (for explainability — *why* these
        chunks) and the citations (the actual payload to read).
        ``budget`` clamps the citation list by approximate token count;
        the top-ranked chunk always lands even if oversized.

        Use when you'd otherwise call search_pages + search_chunks
        yourself. If results don't fit: refine the goal, increase
        budget, or drop to the cheaper tools for finer control.
        """
        result = wikiloom_get_context(
            cache, embedder, goal,
            top_pages=_DEFAULT_TOP_PAGES,
            k=_DEFAULT_K,
            budget=budget,
        )
        return ContextResultOut(
            pages=[
                PageHitOut(
                    page_id=p.page_id,
                    type=p.type,
                    title=p.title,
                    summary=short_summary(p.summary),
                    similarity=p.similarity,
                )
                for p in result.pages
            ],
            citations=[CitationOut(**asdict(c)) for c in result.citations],
        )

    @mcp.tool()
    def get_outbound_links(page_id: str) -> list[OutboundLinkOut]:
        """Graph hop. Returns outbound wikilink targets from one page.

        Call after search_pages finds a near-match page to discover
        related topics, then pass targets to get_pages to follow the hop.
        Empty list if the page has no outbound links or doesn't exist.
        """
        edges = cache.get_outbound_edges([page_id])
        return [
            OutboundLinkOut(
                source_page=e["source_page"],
                target_page=e["target_page"],
            )
            for e in edges
        ]

    @mcp.tool()
    def get_backlinks(page_id: str, limit: int = 10) -> list[BacklinkOut]:
        """Graph hop. Returns pages that link TO ``page_id`` (inbound edges).

        Call after search_pages finds a relevant page to discover what
        else cites it; pass each result's ``source_page`` to get_pages to
        read the citing context. Empty list if no page cites it or it
        doesn't exist.

        Set ``limit`` to cap the number of edges returned (default 10).
        Popular pages can have hundreds of backlinks — raise the limit
        deliberately when you need breadth, lower it to stay cheap.
        """
        edges = cache.get_inbound_edges([page_id], limit=limit)
        return [
            BacklinkOut(
                target_page=e["target_page"],
                source_page=e["source_page"],
            )
            for e in edges
        ]

    return mcp


def serve(project: Path) -> None:
    """Boot a stdio MCP server for the WikiLoom project at ``project``.

    Fails loud if the project has no configured embedder — the agent
    surface is useless without retrieval, and degraded-mode emptiness
    would mask a config error.
    """
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    embedder = load_embedder(project)
    if embedder is None:
        raise RuntimeError(
            f"No embedder configured for project at {project}. "
            f"The MCP server requires retrieval; enable embeddings in "
            f"wikiloom.toml before starting."
        )
    mcp = build_server(cache, embedder, project)
    mcp.run(transport="stdio")
