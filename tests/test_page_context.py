"""Tests for per-chunk page context retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikiloom.cache import SQLiteCache
from wikiloom.embeddings import serialize_embedding
from wikiloom.frontmatter import Frontmatter, write_page
from wikiloom.page_context import (
    PageCandidate,
    render_candidates,
    retrieve_candidates_for_chunk,
)
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project


class _FakeEmbedder:
    """Deterministic embedder that maps keywords to fixed vectors."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            matched: list[float] | None = None
            for keyword, vector in self._mapping.items():
                if keyword.lower() in text.lower():
                    matched = vector
                    break
            if matched is None:
                # Default vector — orthogonal-ish to all mapped ones.
                matched = [0.0, 0.0, 0.0, 1.0]
            out.append(matched)
        return out


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return init_project(name="testproj", path=tmp_path, domain="test")


def _register_page(
    project_root: Path,
    page_id: str,
    title: str,
    summary: str,
) -> None:
    """Write a page file + manifest entry without touching the cache."""
    wiki_dir = project_root / "wiki"
    registry = Registry(project_root / "_registry", wiki_dir=wiki_dir)
    entry = PageEntry(
        title=title,
        type=page_id.split("/", 1)[0].rstrip("s"),  # concepts → concept
        page_id=page_id,
        summary=summary,
        created="2026-04-14T10:00:00Z",
        modified="2026-04-14T10:00:00Z",
    )
    registry.register_page(page_id, entry)
    registry.save()

    page_path = wiki_dir / f"{page_id}.md"
    write_page(
        page_path,
        Frontmatter(
            title=title,
            type=entry.type,
            created=entry.created,
            modified=entry.modified,
            summary=summary,
        ),
        f"# {title}\n\n{summary}\n",
    )


def _seed_pages_with_embeddings(
    project_root: Path, pages: list[tuple[str, str, str, list[float]]]
) -> None:
    """Register every page, rebuild the cache once, then set embeddings.

    Rebuilding the cache per-page wipes the embedding column, so the
    pattern is: register all, rebuild once, then poke embeddings in.
    """
    for page_id, title, summary, _ in pages:
        _register_page(project_root, page_id, title, summary)

    cache = SQLiteCache(project_root / "_registry" / "wiki.db")
    cache.full_rebuild(project_root)
    with cache._connect() as conn:
        for page_id, _, _, vector in pages:
            conn.execute(
                "UPDATE pages SET embedding = ? WHERE page_id = ?",
                (serialize_embedding(vector), page_id),
            )


def test_empty_cache_returns_empty(project: Path) -> None:
    """A fresh project with no pages returns no candidates."""
    embedder = _FakeEmbedder({"foo": [1.0, 0.0, 0.0, 0.0]})
    assert retrieve_candidates_for_chunk(
        chunk_text="foo bar baz", project_root=project, embedder=embedder
    ) == []


def test_retrieves_most_similar_page(project: Path) -> None:
    """Top-K returns pages ordered by cosine similarity."""
    _seed_pages_with_embeddings(
        project,
        [
            (
                "concepts/transactions",
                "Transactions",
                "Banking transactions",
                [1.0, 0.0, 0.0, 0.0],
            ),
            (
                "concepts/interest-rates",
                "Interest Rates",
                "Bank interest",
                [0.0, 1.0, 0.0, 0.0],
            ),
        ],
    )

    embedder = _FakeEmbedder(
        {
            "pending": [1.0, 0.0, 0.0, 0.0],
            "interest": [0.0, 1.0, 0.0, 0.0],
        }
    )

    pending_hits = retrieve_candidates_for_chunk(
        chunk_text="pending transaction details",
        project_root=project,
        embedder=embedder,
        top_k=5,
    )
    assert pending_hits, "expected at least one candidate"
    assert pending_hits[0].page_id == "concepts/transactions"

    interest_hits = retrieve_candidates_for_chunk(
        chunk_text="interest rate schedule",
        project_root=project,
        embedder=embedder,
        top_k=5,
    )
    assert interest_hits[0].page_id == "concepts/interest-rates"


def test_respects_top_k_limit(project: Path) -> None:
    """Never returns more candidates than requested."""
    _seed_pages_with_embeddings(
        project,
        [
            (
                f"concepts/topic-{i}",
                f"Topic {i}",
                f"Summary {i}",
                [1.0, i * 0.01, 0.0, 0.0],
            )
            for i in range(5)
        ],
    )
    embedder = _FakeEmbedder({"topic": [1.0, 0.0, 0.0, 0.0]})

    results = retrieve_candidates_for_chunk(
        chunk_text="topic discussion",
        project_root=project,
        embedder=embedder,
        top_k=2,
    )
    assert len(results) == 2


def test_respects_min_similarity_floor(project: Path) -> None:
    """Pages below the similarity floor are excluded."""
    _seed_pages_with_embeddings(
        project,
        [
            (
                "concepts/unrelated",
                "Unrelated",
                "Something else",
                [0.0, 0.0, 1.0, 0.0],
            )
        ],
    )
    embedder = _FakeEmbedder({"query": [1.0, 0.0, 0.0, 0.0]})
    assert (
        retrieve_candidates_for_chunk(
            chunk_text="query about something",
            project_root=project,
            embedder=embedder,
        )
        == []
    )


def test_render_candidates_empty_placeholder() -> None:
    """Empty candidate list renders a clear placeholder string."""
    out = render_candidates([])
    assert "no semantically related pages" in out


def test_render_candidates_table_format() -> None:
    """Candidates render as a markdown table with the expected columns."""
    cands = [
        PageCandidate(
            page_id="concepts/foo",
            type="concept",
            title="Foo",
            summary="A short summary",
            similarity=0.95,
        )
    ]
    out = render_candidates(cands)
    assert "| page_id | type | title | summary |" in out
    assert "concepts/foo" in out
    assert "A short summary" in out


def test_embedder_failure_returns_empty(project: Path) -> None:
    """Embedder errors surface as empty list, not an exception."""

    class BrokenEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("model failed to load")

    _seed_pages_with_embeddings(
        project,
        [
            (
                "concepts/anything",
                "Anything",
                "Summary",
                [1.0, 0.0, 0.0, 0.0],
            )
        ],
    )
    assert (
        retrieve_candidates_for_chunk(
            chunk_text="foo", project_root=project, embedder=BrokenEmbedder()
        )
        == []
    )
