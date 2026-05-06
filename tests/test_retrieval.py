"""Tests for the chunk-direct retrieval lane (wikiloom/retrieval.py)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from wikiloom.cache import SQLiteCache
from wikiloom.chunk_store import ChunkStore
from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.retrieval import (
    _SNIPPET_MAX_CHARS,
    Citation,
    _make_snippet,
    _rrf_fuse,
    search_chunks,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _chunk(text: str, index: int, total: int, *, parent_heading: str | None = None) -> ExtractedContent:
    meta: dict = {"chunk_index": index, "chunk_total": total}
    if parent_heading is not None:
        meta["parent_heading"] = parent_heading
    return ExtractedContent(
        text=text,
        metadata=meta,
        source_path=Path("/tmp/test-doc.md"),
        content_type="markdown",
        extraction_method="test-fixture",
        token_estimate=10,
    )


class _FakeEmbedder:
    """Deterministic 8-dim hashed embedder so tests don't need fastembed."""

    DEFAULT_MODEL = "fake-model-1"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            # Char-level bag — same words → similar vector.
            v = [0.0] * self.dim
            for ch in t.lower():
                v[ord(ch) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


@pytest.fixture
def populated(tmp_path: Path) -> tuple[SQLiteCache, _FakeEmbedder]:
    cache = SQLiteCache(tmp_path / "_registry" / "wiki.db")
    store = ChunkStore(cache)
    embedder = _FakeEmbedder(dim=8)
    chunks = [
        _chunk("authentication and login flow", 0, 3, parent_heading="Auth"),
        _chunk("logout endpoint reference", 1, 3, parent_heading="Auth"),
        _chunk("payment provider configuration", 2, 3, parent_heading="Billing"),
    ]
    store.persist_chunks(
        "src-1", chunks,
        embedder=embedder,
        embedder_provider="fake",
        embedder_model="fake-model-1",
    )
    return cache, embedder


# ----------------------------------------------------------------------
# Citation shape
# ----------------------------------------------------------------------


def test_citation_is_frozen_and_has_expected_fields() -> None:
    c = Citation(
        chunk_id="abc",
        page_id="concepts/auth",
        source_path="/tmp/foo.md",
        parent_heading="Auth",
        snippet="auth body preview",
        score=0.123,
    )
    assert c.chunk_id == "abc"
    assert c.page_id == "concepts/auth"
    assert c.source_path == "/tmp/foo.md"
    assert c.parent_heading == "Auth"
    assert c.snippet == "auth body preview"
    assert c.score == 0.123
    with pytest.raises(Exception):
        c.score = 9.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# _make_snippet helper
# ----------------------------------------------------------------------


def test_make_snippet_passes_short_text_through() -> None:
    assert _make_snippet("short text") == "short text"


def test_make_snippet_collapses_whitespace() -> None:
    assert _make_snippet("a\n\nb\t c   d") == "a b c d"


def test_make_snippet_truncates_with_ellipsis() -> None:
    long = "x" * (_SNIPPET_MAX_CHARS + 50)
    s = _make_snippet(long)
    assert len(s) == _SNIPPET_MAX_CHARS
    assert s.endswith("…")


def test_make_snippet_handles_empty() -> None:
    assert _make_snippet("") == ""


# ----------------------------------------------------------------------
# RRF math
# ----------------------------------------------------------------------


def test_rrf_fuse_rank_1_in_one_lane() -> None:
    fused = _rrf_fuse({100: 1}, {})
    assert fused == {100: 1.0 / (60 + 1)}


def test_rrf_fuse_sums_across_lanes() -> None:
    fused = _rrf_fuse({100: 1}, {100: 2})
    assert fused == {100: 1.0 / (60 + 1) + 1.0 / (60 + 2)}


def test_rrf_fuse_unioned_rowids() -> None:
    fused = _rrf_fuse({100: 1, 200: 5}, {200: 1, 300: 3})
    # 200 appears in both lanes → highest fused score.
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    assert ranked[0][0] == 200


# ----------------------------------------------------------------------
# search_chunks — happy paths
# ----------------------------------------------------------------------


def test_search_returns_citations(populated: tuple[SQLiteCache, _FakeEmbedder]) -> None:
    cache, embedder = populated
    cites = search_chunks(cache, embedder, "authentication", k=5)
    assert cites
    assert all(isinstance(c, Citation) for c in cites)
    # The auth chunk should rank above billing.
    chunk_ids_by_rank = [c.chunk_id for c in cites]
    auth_idx = next(i for i, c in enumerate(cites) if "auth" in (c.parent_heading or "").lower())
    billing_idx = next(
        (i for i, c in enumerate(cites) if c.parent_heading == "Billing"),
        len(cites),
    )
    assert auth_idx < billing_idx, chunk_ids_by_rank


def test_search_populates_provenance(populated: tuple[SQLiteCache, _FakeEmbedder]) -> None:
    cache, embedder = populated
    cites = search_chunks(cache, embedder, "logout", k=3)
    assert cites
    top = cites[0]
    assert top.source_path == "/tmp/test-doc.md"
    assert top.parent_heading in ("Auth", "Billing")
    assert top.snippet  # always populated, even if parent_heading is set
    assert top.score > 0.0


def test_search_populates_snippet_for_non_markdown_chunks(
    tmp_path: Path,
) -> None:
    # Non-markdown chunk → parent_heading is None, but snippet must
    # still give the agent a textual preview.
    cache = SQLiteCache(tmp_path / "_registry" / "wiki.db")
    store = ChunkStore(cache)
    chunks = [
        ExtractedContent(
            text="A long document body discussing payment configuration.",
            metadata={"chunk_index": 0, "chunk_total": 1},
            source_path=Path("/tmp/api.pdf"),
            content_type="pdf",
            extraction_method="pymupdf",
            token_estimate=10,
        ),
    ]
    store.persist_chunks("src-pdf", chunks)
    cites = search_chunks(cache, None, "payment", k=3)
    assert cites
    assert cites[0].parent_heading is None
    assert "payment" in cites[0].snippet.lower()


def test_search_respects_k(populated: tuple[SQLiteCache, _FakeEmbedder]) -> None:
    cache, embedder = populated
    cites = search_chunks(cache, embedder, "payment", k=1)
    assert len(cites) == 1


def test_search_min_score_filters(populated: tuple[SQLiteCache, _FakeEmbedder]) -> None:
    cache, embedder = populated
    high_floor = 10.0  # impossibly high
    cites = search_chunks(cache, embedder, "authentication", k=5, min_score=high_floor)
    assert cites == []


# ----------------------------------------------------------------------
# search_chunks — degraded modes
# ----------------------------------------------------------------------


def test_search_works_without_embedder_bm25_only(
    populated: tuple[SQLiteCache, _FakeEmbedder],
) -> None:
    cache, _ = populated
    cites = search_chunks(cache, None, "authentication", k=5)
    assert cites
    assert any("auth" in (c.parent_heading or "").lower() for c in cites)


def test_search_empty_query_returns_empty(
    populated: tuple[SQLiteCache, _FakeEmbedder],
) -> None:
    cache, embedder = populated
    assert search_chunks(cache, embedder, "", k=5) == []
    assert search_chunks(cache, embedder, "   ", k=5) == []


def test_search_zero_k_returns_empty(
    populated: tuple[SQLiteCache, _FakeEmbedder],
) -> None:
    cache, embedder = populated
    assert search_chunks(cache, embedder, "auth", k=0) == []


def test_search_no_match_returns_empty(
    populated: tuple[SQLiteCache, _FakeEmbedder],
) -> None:
    cache, _ = populated
    # Nonsense query, no embedder → BM25-only, no FTS hits.
    assert search_chunks(cache, None, "zzzzqqq", k=5) == []


# ----------------------------------------------------------------------
# search_chunks — fingerprint enforcement
# ----------------------------------------------------------------------


def test_search_uses_vector_lane_to_recall_chunks_bm25_misses(
    tmp_path: Path,
) -> None:
    # Locks the fusion contract: the vector lane must contribute rowids
    # that BM25 alone wouldn't return. A regression to single-lane would
    # see hybrid == bm25_only.
    cache = SQLiteCache(tmp_path / "_registry" / "wiki.db")
    store = ChunkStore(cache)
    embedder = _FakeEmbedder(dim=8)
    chunks = [
        _chunk("alpha keyword", 0, 3),
        _chunk("alpha alpha keyword", 1, 3),
        _chunk("bravo charlie delta", 2, 3),  # no FTS match for 'alpha'
    ]
    store.persist_chunks(
        "src-fuse", chunks,
        embedder=embedder,
        embedder_provider="fake",
        embedder_model="fake-model-1",
    )

    bm25_only = search_chunks(cache, None, "alpha", k=5)
    hybrid = search_chunks(cache, embedder, "alpha", k=5)

    bm25_ids = {c.chunk_id for c in bm25_only}
    hybrid_ids = {c.chunk_id for c in hybrid}
    # Hybrid recalls at least one chunk BM25 missed (chunk 2, no 'alpha').
    assert hybrid_ids - bm25_ids, (
        f"hybrid should recall vector-only chunks; "
        f"bm25={bm25_ids}, hybrid={hybrid_ids}"
    )


def test_search_with_embedder_falls_back_when_no_vector_index(
    tmp_path: Path,
) -> None:
    # Persist without an embedder → no chunk_vec, no fingerprint. A
    # later search call given an embedder should not raise; it should
    # silently skip the vector lane and return BM25-only results.
    cache = SQLiteCache(tmp_path / "_registry" / "wiki.db")
    store = ChunkStore(cache)
    store.persist_chunks(
        "src-bm25-only",
        [_chunk("alpha keyword", 0, 2), _chunk("bravo keyword", 1, 2)],
    )
    embedder = _FakeEmbedder(dim=8)
    cites = search_chunks(cache, embedder, "alpha", k=5)
    # Only the chunk containing 'alpha' should come back — vector
    # lane is silently skipped, so the second 'bravo' chunk doesn't
    # pile in via similarity.
    assert len(cites) == 1
    assert cites[0].score > 0.0


def test_search_raises_on_dim_mismatch(
    populated: tuple[SQLiteCache, _FakeEmbedder],
) -> None:
    cache, _ = populated
    bigger = _FakeEmbedder(dim=16)
    with pytest.raises(RuntimeError, match="dim mismatch"):
        search_chunks(cache, bigger, "auth", k=3)


def test_search_raises_on_provider_model_mismatch(
    populated: tuple[SQLiteCache, _FakeEmbedder],
) -> None:
    cache, embedder = populated
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        search_chunks(
            cache, embedder, "auth", k=3,
            embedder_provider="other",
            embedder_model="other-model",
        )
