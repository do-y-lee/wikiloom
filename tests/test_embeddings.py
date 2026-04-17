"""Tests for the embedding module + semantic retrieval fallback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import git
import pytest

from wikiloom.cache import SQLiteCache
from wikiloom.embeddings import (
    EmbeddingConfig,
    cosine_similarity,
    deserialize_embedding,
    get_embedder,
    serialize_embedding,
)
from wikiloom.frontmatter import Frontmatter, write_page
from wikiloom.query import retrieve_context
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project


# ----------------------------------------------------------------------
# Mock embedder
# ----------------------------------------------------------------------


class MockEmbedder:
    """Deterministic embedder that hashes text into a fixed-dim vector.

    Not meaningful for real similarity, but stable and testable —
    same input always produces the same output, different inputs
    produce different outputs.
    """

    def __init__(self, dims: int = 8) -> None:
        self.dims = dims
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        vectors = []
        for text in texts:
            h = hash(text) & 0xFFFFFFFF
            vec = []
            for i in range(self.dims):
                val = ((h + i * 7919) % 1000) / 1000.0 - 0.5
                vec.append(val)
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = init_project(name="embed-test", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    repo.index.add(
        [
            "wikiloom.toml",
            ".gitignore",
            "wiki/index.md",
            "_registry/manifest.json",
            "_registry/backlinks.json",
        ]
    )
    repo.index.commit("initial scaffold")
    return project_dir


def _add_page(project: Path, page_id: str, title: str, body: str, summary: str = "") -> None:
    fm = Frontmatter(
        title=title,
        type=page_id.split("/")[0].rstrip("s") if "/" in page_id else "concept",
        status="active",
        created="2026-01-01T00:00:00Z",
        modified="2026-01-01T00:00:00Z",
        summary=summary or f"Summary of {title}.",
    )
    write_page(project / "wiki" / f"{page_id}.md", fm, body)
    registry = Registry(project / "_registry")
    registry.register_page(
        page_id,
        PageEntry(title=title, type=fm.type, summary=summary or f"Summary of {title}."),
    )
    registry.save()


# ----------------------------------------------------------------------
# Vector utilities
# ----------------------------------------------------------------------


def test_serialize_deserialize_roundtrip() -> None:
    vec = [0.1, -0.5, 0.9, 0.0, -1.0]
    blob = serialize_embedding(vec)
    result = deserialize_embedding(blob)
    assert len(result) == len(vec)
    for a, b in zip(vec, result):
        assert abs(a - b) < 1e-6


def test_cosine_similarity_identical_vectors() -> None:
    v = [1.0, 0.0, 0.5]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_safe() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ----------------------------------------------------------------------
# get_embedder
# ----------------------------------------------------------------------


def test_get_embedder_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder(EmbeddingConfig(provider="nonexistent"))


# ----------------------------------------------------------------------
# Cache embedding storage
# ----------------------------------------------------------------------


def test_full_rebuild_stores_embeddings(project: Path) -> None:
    _add_page(project, "concepts/alpha", "Alpha", "Alpha is a concept.")
    _add_page(project, "concepts/beta", "Beta", "Beta is another concept.")

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    embedder = MockEmbedder()
    cache.full_rebuild(project, embedder=embedder)

    assert embedder.call_count == 2

    page = cache.get_page("concepts/alpha")
    assert page is not None


def test_full_rebuild_without_embedder_leaves_null(project: Path) -> None:
    _add_page(project, "concepts/alpha", "Alpha", "Alpha is a concept.")

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)  # no embedder

    # Should still work — pages table has embedding column but it's NULL
    stats = cache.get_stats()
    assert stats["total_pages"] == 1


# ----------------------------------------------------------------------
# Semantic search
# ----------------------------------------------------------------------


def test_semantic_search_returns_similar_pages(project: Path) -> None:
    _add_page(project, "concepts/alpha", "Alpha", "Alpha is a concept about X.")
    _add_page(project, "concepts/beta", "Beta", "Beta is a concept about Y.")

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    embedder = MockEmbedder()
    cache.full_rebuild(project, embedder=embedder)

    query_vec = embedder.embed_texts(["Tell me about Alpha"])[0]
    results = cache.semantic_search(query_vec, limit=2)

    assert len(results) == 2
    assert all("similarity" in r for r in results)
    assert results[0]["similarity"] >= results[1]["similarity"]


def test_semantic_search_empty_when_no_embeddings(project: Path) -> None:
    _add_page(project, "concepts/alpha", "Alpha", "Alpha is a concept.")

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)  # no embedder → no embeddings

    query_vec = [0.1] * 8
    results = cache.semantic_search(query_vec)
    assert results == []


# ----------------------------------------------------------------------
# Query retrieval fallback
# ----------------------------------------------------------------------


def test_retrieve_falls_back_to_semantic_when_fts_empty(project: Path) -> None:
    """If FTS5 finds nothing, semantic search should be tried."""
    _add_page(
        project,
        "concepts/overdraft-protection",
        "Overdraft Protection",
        "When your account becomes overdrawn, the bank may transfer funds.",
        summary="Overdraft protection service.",
    )

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    embedder = MockEmbedder()
    cache.full_rebuild(project, embedder=embedder)

    # This query uses words that FTS5 might not match
    # but the mock embedder will return results via cosine similarity
    contexts = retrieve_context(
        "zzzzz_nonexistent_keyword",
        cache,
        project / "wiki",
        embedder=embedder,
    )
    # With semantic fallback, should find at least one page
    assert len(contexts) >= 1


def test_retrieve_fts_match_appears_first(project: Path) -> None:
    """FTS5 keyword matches should appear before semantic top-ups."""
    _add_page(
        project,
        "concepts/overdraft-protection",
        "Overdraft Protection",
        "When your account becomes overdrawn.",
        summary="Overdraft protection.",
    )
    _add_page(
        project,
        "concepts/other-topic",
        "Other Topic",
        "Something unrelated to overdrafts.",
        summary="Unrelated topic.",
    )

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    embedder = MockEmbedder()
    cache.full_rebuild(project, embedder=embedder)

    # "overdraft" is a direct FTS5 match — should be first result
    contexts = retrieve_context(
        "overdraft",
        cache,
        project / "wiki",
        embedder=embedder,
    )
    assert len(contexts) >= 1
    assert contexts[0].page_id == "concepts/overdraft-protection"
