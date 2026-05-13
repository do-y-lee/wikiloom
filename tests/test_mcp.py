"""Tests for the MCP server surface (wikiloom/mcp/server.py)."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import pytest

from wikiloom.cache import SQLiteCache
from wikiloom.chunk_store import ChunkStore
from wikiloom.embeddings import serialize_embedding
from wikiloom.frontmatter import Frontmatter, write_page
from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.mcp.models import short_summary
from wikiloom.mcp.server import build_server, serve


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _FakeEmbedder:
    DEFAULT_MODEL = "fake-model-1"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            for ch in t.lower():
                v[ord(ch) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _chunk(
    text: str, index: int, total: int, *,
    parent_heading: str | None = None,
    token_estimate: int = 10,
) -> ExtractedContent:
    meta: dict = {"chunk_index": index, "chunk_total": total}
    if parent_heading is not None:
        meta["parent_heading"] = parent_heading
    return ExtractedContent(
        text=text,
        metadata=meta,
        source_path=Path("/tmp/test-doc.md"),
        content_type="markdown",
        extraction_method="test-fixture",
        token_estimate=token_estimate,
    )


def _seed_pages(
    cache: SQLiteCache,
    embedder: _FakeEmbedder,
    specs: list[tuple[str, str, str, str, str, str]],
) -> None:
    """``specs`` items: ``(page_id, title, type, status, summary, embed_text)``."""
    with cache._connect() as conn:
        for page_id, title, type_, status, summary, embed_text in specs:
            vec = embedder.embed_texts([embed_text])[0]
            blob = serialize_embedding(vec)
            conn.execute(
                "INSERT INTO pages "
                "(page_id, title, type, status, summary, "
                "created, modified, embedding) "
                "VALUES (?, ?, ?, ?, ?, "
                "'2026-01-01', '2026-01-01', ?)",
                (page_id, title, type_, status, summary, blob),
            )
    cache._invalidate_embeddings()


def _seed_backlinks(cache: SQLiteCache, edges: list[tuple[str, str]]) -> None:
    with cache._connect() as conn:
        for src, tgt in edges:
            conn.execute(
                "INSERT INTO backlinks (source_page, target_page, linked_at) "
                "VALUES (?, ?, '2026-01-01')",
                (src, tgt),
            )


def _write_wiki_files(wiki_dir: Path, pages: list[tuple[str, str, str]]) -> None:
    """``pages`` items: ``(page_id, summary, body)``."""
    for page_id, summary, body in pages:
        fm = Frontmatter(
            title=page_id.split("/")[-1].title(),
            type="concept",
            status="active",
            summary=summary,
            created="2026-01-01",
            modified="2026-01-01",
        )
        write_page(wiki_dir / f"{page_id}.md", fm, body)


@pytest.fixture
def project_setup(tmp_path: Path) -> tuple[Any, Path, _FakeEmbedder]:
    """Cache + chunks + page rows + wiki files + backlinks; returns (mcp, project, embedder)."""
    project = tmp_path
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    embedder = _FakeEmbedder(dim=8)
    store = ChunkStore(cache)

    _seed_pages(cache, embedder, [
        ("concepts/auth", "Authentication", "concept", "active",
         "Authentication and login flow summary that is intentionally "
         "much longer than the boundary truncation threshold so the "
         "test can confidently assert the ellipsis suffix is added.",
         "authentication login flow logout endpoint"),
        ("concepts/billing", "Billing", "concept", "active",
         "Payment provider configuration summary.",
         "payment provider configuration invoice subtotal"),
        ("concepts/dead", "Dead Auth", "concept", "deprecated",
         "Old auth content.",
         "authentication ancient deprecated content"),
    ])

    chunks = [
        _chunk("authentication and login flow", 0, 4, parent_heading="Auth"),
        _chunk("logout endpoint reference", 1, 4, parent_heading="Auth"),
        _chunk("payment provider configuration", 2, 4, parent_heading="Billing"),
        _chunk("invoice subtotal handling", 3, 4, parent_heading="Billing"),
    ]
    stored = store.persist_chunks(
        "src-1", chunks,
        embedder=embedder,
        embedder_provider="fake",
        embedder_model="fake-model-1",
    )
    store.set_page_ids({
        stored[0].chunk_id: "concepts/auth",
        stored[1].chunk_id: "concepts/auth",
        stored[2].chunk_id: "concepts/billing",
        stored[3].chunk_id: "concepts/billing",
    })

    _write_wiki_files(project / "wiki", [
        ("concepts/auth",
         "Authentication and login flow summary that is intentionally "
         "much longer than the boundary truncation threshold so the "
         "test can confidently assert the ellipsis suffix is added.",
         "# Authentication\n\nFull body of the auth page.\n"),
        ("concepts/billing",
         "Payment provider configuration summary.",
         "# Billing\n\nFull body of the billing page.\n"),
    ])

    _seed_backlinks(cache, [
        ("concepts/auth", "concepts/billing"),
    ])

    mcp = build_server(cache, embedder, project)
    return mcp, project, embedder


def _call(mcp: Any, name: str, args: dict[str, Any]) -> Any:
    """Invoke an MCP tool synchronously and return its structured payload.

    FastMCP wraps ``list[X]`` returns as ``{"result": [...]}`` but a
    pydantic-model return comes back as the model's dict directly.
    Normalize so callers always get the payload (list or dict).
    """
    _, structured = asyncio.run(mcp.call_tool(name, args))
    if isinstance(structured, dict) and list(structured) == ["result"]:
        return next(iter(structured.values()))
    return structured


# ----------------------------------------------------------------------
# Registration / schemas
# ----------------------------------------------------------------------


def test_six_tools_are_registered(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "search_pages", "search_chunks",
        "get_pages", "get_chunks",
        "get_context", "get_outbound_links",
    }


def test_tool_descriptions_teach_router_vs_payload(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    # Descriptions are load-bearing copy: the agent needs to learn the
    # cheap-vs-expensive pattern from the tool surface alone.
    mcp, _, _ = project_setup
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert "Cheap router" in tools["search_pages"].description
    assert "Cheap router" in tools["search_chunks"].description
    assert "Expensive payload" in tools["get_pages"].description
    assert "Expensive payload" in tools["get_chunks"].description
    assert "orchestrator" in tools["get_context"].description.lower()
    assert "hop" in tools["get_outbound_links"].description.lower()


def test_search_pages_schema_describes_summary_truncation(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    # PageHitOut.summary description tells the agent the field is
    # NOT the full body — that's how it learns to call get_pages.
    mcp, _, _ = project_setup
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    blob = str(tools["search_pages"].outputSchema)
    assert "NOT the full body" in blob
    assert "get_pages" in blob


# ----------------------------------------------------------------------
# search_pages
# ----------------------------------------------------------------------


def test_search_pages_returns_page_hits(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    pages = _call(mcp, "search_pages", {"query": "authentication login", "k": 5})
    assert pages
    expected_keys = {"page_id", "type", "title", "summary", "similarity"}
    assert all(expected_keys <= set(p) for p in pages)


def test_search_pages_excludes_deprecated(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    pages = _call(mcp, "search_pages", {"query": "authentication", "k": 10})
    page_ids = {p["page_id"] for p in pages}
    assert "concepts/dead" not in page_ids


def test_search_pages_truncates_long_summaries(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    # The auth page summary is >120 chars; must come back truncated.
    # k widened so the FakeEmbedder's ordering noise doesn't drop it.
    mcp, _, _ = project_setup
    pages = _call(mcp, "search_pages", {"query": "authentication", "k": 10})
    auth = next(p for p in pages if p["page_id"] == "concepts/auth")
    assert auth["summary"].endswith("…")
    assert len(auth["summary"]) <= 121  # 120 + ellipsis


# ----------------------------------------------------------------------
# search_chunks
# ----------------------------------------------------------------------


def test_search_chunks_returns_citations(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    cites = _call(mcp, "search_chunks", {"query": "authentication", "k": 5})
    assert cites
    expected_keys = {"chunk_id", "snippet", "score"}
    assert all(expected_keys <= set(c) for c in cites)


def test_search_chunks_carries_token_estimate_through_boundary(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    cites = _call(mcp, "search_chunks", {"query": "authentication", "k": 5})
    assert all(c["token_estimate"] == 10 for c in cites)


# ----------------------------------------------------------------------
# get_pages
# ----------------------------------------------------------------------


def test_get_pages_returns_full_bodies(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    pages = _call(mcp, "get_pages", {"ids": ["concepts/auth", "concepts/billing"]})
    assert len(pages) == 2
    by_id = {p["page_id"]: p for p in pages}
    assert "Full body of the auth page" in by_id["concepts/auth"]["body"]
    assert "Full body of the billing page" in by_id["concepts/billing"]["body"]


def test_get_pages_silently_skips_missing(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    pages = _call(mcp, "get_pages", {"ids": ["concepts/auth", "concepts/does-not-exist"]})
    assert [p["page_id"] for p in pages] == ["concepts/auth"]


# ----------------------------------------------------------------------
# get_chunks
# ----------------------------------------------------------------------


def test_get_chunks_returns_full_text(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    cites = _call(mcp, "search_chunks", {"query": "authentication", "k": 1})
    chunk_id = cites[0]["chunk_id"]
    full = _call(mcp, "get_chunks", {"ids": [chunk_id]})
    assert len(full) == 1
    assert full[0]["chunk_id"] == chunk_id
    assert full[0]["text"]


def test_get_chunks_silently_skips_missing(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    cites = _call(mcp, "search_chunks", {"query": "authentication", "k": 1})
    real = cites[0]["chunk_id"]
    result = _call(mcp, "get_chunks", {"ids": [real, "nope-1", "nope-2"]})
    assert [c["chunk_id"] for c in result] == [real]


# ----------------------------------------------------------------------
# get_context
# ----------------------------------------------------------------------


def test_get_context_returns_pages_and_citations(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    result = _call(mcp, "get_context", {"goal": "authentication login", "budget": 2000})
    assert "pages" in result and "citations" in result
    assert result["pages"]
    assert result["citations"]


def test_get_context_summaries_are_truncated(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    # Same truncation rule as search_pages — the pages field here is
    # explainability, not the full payload.
    mcp, _, _ = project_setup
    result = _call(mcp, "get_context", {"goal": "authentication"})
    auth = next((p for p in result["pages"] if p["page_id"] == "concepts/auth"), None)
    assert auth is not None
    assert auth["summary"].endswith("…")


def test_get_context_budget_clamps_citations(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    # 4 chunks * 10 tokens = 40 tokens total. budget=15 → 1 chunk
    # (10 fits, second would push to 20 > 15).
    result = _call(mcp, "get_context", {"goal": "authentication", "budget": 15})
    assert len(result["citations"]) == 1


# ----------------------------------------------------------------------
# get_outbound_links
# ----------------------------------------------------------------------


def test_get_outbound_links_returns_targets(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    edges = _call(mcp, "get_outbound_links", {"page_id": "concepts/auth"})
    assert len(edges) == 1
    assert edges[0]["source_page"] == "concepts/auth"
    assert edges[0]["target_page"] == "concepts/billing"


def test_get_outbound_links_empty_for_isolated_page(
    project_setup: tuple[Any, Path, _FakeEmbedder],
) -> None:
    mcp, _, _ = project_setup
    assert _call(mcp, "get_outbound_links", {"page_id": "concepts/billing"}) == []


# ----------------------------------------------------------------------
# Summary truncation helper
# ----------------------------------------------------------------------


def test_short_summary_passes_short_strings_through() -> None:
    assert short_summary("short") == "short"
    assert short_summary("") == ""


def test_short_summary_truncates_at_word_boundary_with_ellipsis() -> None:
    s = "alpha beta gamma delta epsilon zeta eta theta " * 5
    out = short_summary(s, max_chars=30)
    assert out.endswith("…")
    assert len(out) <= 31
    # Word-boundary: no half-word before the ellipsis.
    assert " " in out
    assert not out[:-1].rstrip().endswith(("alph", "bet", "gam"))


# ----------------------------------------------------------------------
# serve() startup contract
# ----------------------------------------------------------------------


def test_serve_raises_when_project_has_no_embedder(tmp_path: Path) -> None:
    # An empty project dir has no wikiloom.toml → load_embedder returns
    # None. The server must fail loud rather than booting with a
    # degraded surface that would silently return empty results.
    with pytest.raises(RuntimeError, match="No embedder configured"):
        serve(tmp_path)
