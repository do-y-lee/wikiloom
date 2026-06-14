"""Tests for the hybrid context lane (wikiloom/context.py)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from wikiloom.cache import SQLiteCache
from wikiloom.chunk_store import ChunkStore
from wikiloom.context import ContextResult, PageHit, get_context
from wikiloom.embeddings import serialize_embedding
from wikiloom.ingest.extractors.base import ExtractedContent


# ----------------------------------------------------------------------
# Helpers (mirrors tests/test_retrieval.py — kept local so each test
# file is self-contained)
# ----------------------------------------------------------------------


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


class _FakeEmbedder:
    """Deterministic 8-dim hashed embedder so tests don't need fastembed."""

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


def _seed_pages(
    cache: SQLiteCache,
    embedder: _FakeEmbedder,
    specs: list[tuple[str, str, str, str, str]],
) -> None:
    """Insert pages with embeddings derived from ``embed_text``.

    ``specs`` items: ``(page_id, title, type, status, embed_text)``.
    """
    with cache._connect() as conn:
        for page_id, title, type_, status, embed_text in specs:
            vec = embedder.embed_texts([embed_text])[0]
            blob = serialize_embedding(vec)
            conn.execute(
                "INSERT INTO pages "
                "(page_id, title, type, status, summary, "
                "created, modified, embedding) "
                "VALUES (?, ?, ?, ?, ?, "
                "'2026-01-01', '2026-01-01', ?)",
                (page_id, title, type_, status, f"summary of {title}", blob),
            )
    # Bust the cached embedding matrix so semantic_search reloads.
    cache._invalidate_embeddings()


def _seed_chunk_pages(cache: SQLiteCache, mapping: dict[str, str]) -> None:
    """Mirror a chunk->page mapping into chunk_pages, as the frontmatter
    projection in cache sync does. Scoped retrieval reads this table."""
    with cache._connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO chunk_pages (chunk_id, page_id) VALUES (?, ?)",
            list(mapping.items()),
        )


@pytest.fixture
def project(
    tmp_path: Path,
) -> tuple[SQLiteCache, _FakeEmbedder, ChunkStore]:
    """Six chunks across three active pages + one deprecated page."""
    cache = SQLiteCache(tmp_path / "_registry" / "wiki.db")
    embedder = _FakeEmbedder(dim=8)
    store = ChunkStore(cache)

    _seed_pages(cache, embedder, [
        ("concepts/auth", "Authentication", "concept", "active",
         "authentication login flow logout endpoint"),
        ("concepts/billing", "Billing", "concept", "active",
         "payment provider configuration invoice subtotal"),
        ("concepts/other", "Other", "concept", "active",
         "alpha keyword content bravo charlie delta"),
        ("concepts/dead", "Dead Auth", "concept", "deprecated",
         "authentication ancient deprecated content"),
    ])

    chunks = [
        _chunk("authentication and login flow", 0, 6, parent_heading="Auth"),
        _chunk("logout endpoint reference", 1, 6, parent_heading="Auth"),
        _chunk("payment provider configuration", 2, 6, parent_heading="Billing"),
        _chunk("invoice subtotal handling", 3, 6, parent_heading="Billing"),
        _chunk("alpha keyword content", 4, 6, parent_heading="Other"),
        _chunk("bravo charlie delta", 5, 6, parent_heading="Other"),
    ]
    stored = store.persist_chunks(
        "src-multi", chunks,
        embedder=embedder,
        embedder_provider="fake",
        embedder_model="fake-model-1",
    )
    mapping = {
        stored[0].chunk_id: "concepts/auth",
        stored[1].chunk_id: "concepts/auth",
        stored[2].chunk_id: "concepts/billing",
        stored[3].chunk_id: "concepts/billing",
        stored[4].chunk_id: "concepts/other",
        stored[5].chunk_id: "concepts/other",
    }
    store.set_page_ids(mapping)
    _seed_chunk_pages(cache, mapping)
    return cache, embedder, store


# ----------------------------------------------------------------------
# Shape
# ----------------------------------------------------------------------


def test_get_context_returns_context_result(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication login")
    assert isinstance(result, ContextResult)
    assert all(isinstance(p, PageHit) for p in result.pages)


def test_page_hit_is_frozen() -> None:
    p = PageHit(
        page_id="concepts/auth",
        type="concept",
        title="Authentication",
        summary="auth summary",
        similarity=0.9,
    )
    with pytest.raises(Exception):
        p.similarity = 0.0  # type: ignore[misc]


def test_context_result_is_frozen() -> None:
    r = ContextResult(pages=[], citations=[])
    with pytest.raises(Exception):
        r.pages = []  # type: ignore[misc]


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def test_get_context_routes_to_relevant_pages(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication login flow")
    page_ids = [p.page_id for p in result.pages]
    # Auth page should be the top router hit.
    assert page_ids[0] == "concepts/auth"


def test_get_context_excludes_deprecated_pages(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication", top_pages=10)
    page_ids = {p.page_id for p in result.pages}
    assert "concepts/dead" not in page_ids


def test_get_context_respects_top_pages(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication", top_pages=2)
    assert len(result.pages) <= 2


# ----------------------------------------------------------------------
# Citations sourced only from routed pages
# ----------------------------------------------------------------------


def test_citations_come_only_from_routed_pages(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication", top_pages=1)
    routed = {p.page_id for p in result.pages}
    citation_pages = {c.page_id for c in result.citations}
    assert citation_pages.issubset(routed)


def test_get_context_respects_k(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    result = get_context(
        cache, embedder, "authentication", top_pages=5, k=3,
    )
    assert len(result.citations) <= 3


def test_page_hit_carries_provenance_fields(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication login flow", top_pages=3)
    assert result.pages
    p = result.pages[0]
    # Field shape — not asserting which page tops the list (that's
    # covered by `test_get_context_routes_to_relevant_pages`).
    assert p.page_id.startswith("concepts/")
    assert p.title
    assert p.type == "concept"
    assert p.summary
    assert isinstance(p.similarity, float)
    assert p.similarity > 0.0


# ----------------------------------------------------------------------
# Degraded modes
# ----------------------------------------------------------------------


def test_get_context_empty_goal_returns_empty(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    assert get_context(cache, embedder, "") == ContextResult(
        pages=[], citations=[]
    )
    assert get_context(cache, embedder, "   ") == ContextResult(
        pages=[], citations=[]
    )


def test_get_context_zero_k_returns_empty(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    assert get_context(cache, embedder, "auth", k=0).citations == []


def test_get_context_zero_top_pages_returns_empty(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    assert get_context(cache, embedder, "auth", top_pages=0).pages == []


def test_get_context_no_embedder_returns_empty(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, _, _ = project
    result = get_context(cache, None, "authentication")
    assert result.pages == []
    assert result.citations == []


def test_get_context_empty_cache_returns_empty(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "_registry" / "wiki.db")
    embedder = _FakeEmbedder(dim=8)
    result = get_context(cache, embedder, "authentication")
    assert result.pages == []
    assert result.citations == []


# ----------------------------------------------------------------------
# Fingerprint enforcement
# ----------------------------------------------------------------------


def test_get_context_raises_on_fingerprint_mismatch(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    cache, embedder, _ = project
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        get_context(
            cache, embedder, "authentication",
            embedder_provider="other",
            embedder_model="other-model",
        )


# ----------------------------------------------------------------------
# Token-budget packing
# ----------------------------------------------------------------------


def test_get_context_default_budget_none_is_byte_identical(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    # budget=None must leave the citation list exactly as search_chunks
    # produced it — same shape, same order, same count.
    cache, embedder, _ = project
    baseline = get_context(cache, embedder, "authentication", k=5)
    with_none = get_context(cache, embedder, "authentication", k=5, budget=None)
    assert [c.chunk_id for c in baseline.citations] == [
        c.chunk_id for c in with_none.citations
    ]


def test_get_context_budget_caps_token_sum(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    # Each fixture chunk costs 10 tokens. budget=25 fits two chunks
    # (10 + 10 = 20, next would push to 30 > 25 and stop).
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication", k=5, budget=25)
    assert len(result.citations) == 2
    spent = sum((c.token_estimate or 1) for c in result.citations)
    assert spent <= 25


def test_get_context_budget_includes_oversized_first_chunk(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    # budget=5 < the first chunk's cost (10). Must still include it —
    # returning empty would be worse for the agent than a slight over-cap.
    cache, embedder, _ = project
    result = get_context(cache, embedder, "authentication", k=5, budget=5)
    assert len(result.citations) == 1


def test_get_context_budget_counts_null_token_estimate_as_one(
    tmp_path: Path,
) -> None:
    # Legacy rows can have NULL token_estimate. The packer must count
    # them as 1 (not 0) so the loop still advances and we don't pull in
    # an unbounded number of "free" chunks.
    cache = SQLiteCache(tmp_path / "_registry" / "wiki.db")
    embedder = _FakeEmbedder(dim=8)
    store = ChunkStore(cache)

    _seed_pages(cache, embedder, [
        ("concepts/auth", "Authentication", "concept", "active",
         "authentication login flow logout"),
    ])
    chunks = [
        _chunk(f"authentication chunk {i}", i, 5, parent_heading="Auth")
        for i in range(5)
    ]
    stored = store.persist_chunks(
        "src-null-tok", chunks,
        embedder=embedder,
        embedder_provider="fake",
        embedder_model="fake-model-1",
    )
    mapping = {s.chunk_id: "concepts/auth" for s in stored}
    store.set_page_ids(mapping)
    _seed_chunk_pages(cache, mapping)
    with cache._connect() as conn:
        conn.execute(
            "UPDATE chunks SET token_estimate = NULL WHERE source_hash = ?",
            ("src-null-tok",),
        )
        conn.commit()

    # budget=2 → two NULL chunks (counted as 1 each) fit; third would
    # push spent (2) + cost (1) = 3 > 2 and break.
    result = get_context(cache, embedder, "authentication", k=5, budget=2)
    assert all(c.token_estimate is None for c in result.citations)
    assert len(result.citations) == 2


def test_get_context_budget_preserves_pages_when_citations_packed(
    project: tuple[SQLiteCache, _FakeEmbedder, ChunkStore],
) -> None:
    # Packing trims citations, not pages — the router result must
    # still be observable for explainability.
    cache, embedder, _ = project
    result = get_context(
        cache, embedder, "authentication", top_pages=3, k=5, budget=15,
    )
    assert result.pages
    assert len(result.citations) <= 2
