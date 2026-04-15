"""Tests for the chunk persistence layer (wikiloom/chunk_store.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikiloom.chunk_store import (
    CHUNK_ID_LENGTH,
    ChunkStore,
    derive_chunk_id,
)
from wikiloom.ingest.extractors.base import ExtractedContent


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _chunk(
    text: str,
    index: int,
    total: int,
    content_type: str = "markdown",
    token_estimate: int = 100,
) -> ExtractedContent:
    return ExtractedContent(
        text=text,
        metadata={"chunk_index": index, "chunk_total": total},
        source_path=None,
        content_type=content_type,
        extraction_method="test-fixture",
        token_estimate=token_estimate,
    )


@pytest.fixture
def store(tmp_path: Path) -> ChunkStore:
    return ChunkStore(tmp_path / "_registry" / "wiki.db")


# ----------------------------------------------------------------------
# derive_chunk_id
# ----------------------------------------------------------------------


def test_derive_chunk_id_is_deterministic() -> None:
    a = derive_chunk_id("abc123", 0)
    b = derive_chunk_id("abc123", 0)
    assert a == b
    assert len(a) == CHUNK_ID_LENGTH


def test_derive_chunk_id_varies_by_index() -> None:
    a = derive_chunk_id("abc123", 0)
    b = derive_chunk_id("abc123", 1)
    assert a != b


def test_derive_chunk_id_varies_by_source_hash() -> None:
    a = derive_chunk_id("abc123", 0)
    b = derive_chunk_id("def456", 0)
    assert a != b


# ----------------------------------------------------------------------
# persist_chunks
# ----------------------------------------------------------------------


def test_persist_chunks_writes_rows(store: ChunkStore) -> None:
    chunks = [
        _chunk("first chunk text", 0, 3, token_estimate=150),
        _chunk("second chunk text", 1, 3, token_estimate=180),
        _chunk("third chunk text", 2, 3, token_estimate=90),
    ]
    stored = store.persist_chunks("source-hash-a", chunks)

    assert len(stored) == 3
    assert stored[0].chunk_index == 0
    assert stored[0].text == "first chunk text"
    assert stored[0].source_hash == "source-hash-a"
    assert stored[0].chunk_id == derive_chunk_id("source-hash-a", 0)


def test_persist_chunks_is_empty_safe(store: ChunkStore) -> None:
    stored = store.persist_chunks("empty-source", [])
    assert stored == []
    assert store.count() == 0


def test_persist_chunks_replaces_prior_rows(store: ChunkStore) -> None:
    """A second persist for the same source_hash wipes the earlier rows."""
    store.persist_chunks(
        "source-hash-a",
        [_chunk("old chunk 0", 0, 1)],
    )
    assert store.count() == 1

    store.persist_chunks(
        "source-hash-a",
        [
            _chunk("new chunk 0", 0, 2),
            _chunk("new chunk 1", 1, 2),
        ],
    )
    assert store.count() == 2

    chunks = store.get_chunks_for_source("source-hash-a")
    assert [c.text for c in chunks] == ["new chunk 0", "new chunk 1"]


def test_persist_chunks_from_different_sources_coexist(
    store: ChunkStore,
) -> None:
    store.persist_chunks("source-a", [_chunk("A0", 0, 1)])
    store.persist_chunks("source-b", [_chunk("B0", 0, 1)])

    assert store.count() == 2
    a_chunks = store.get_chunks_for_source("source-a")
    b_chunks = store.get_chunks_for_source("source-b")
    assert len(a_chunks) == 1
    assert len(b_chunks) == 1
    assert a_chunks[0].text == "A0"
    assert b_chunks[0].text == "B0"


# ----------------------------------------------------------------------
# get_chunk / get_chunks_for_source
# ----------------------------------------------------------------------


def test_get_chunk_returns_stored_chunk(store: ChunkStore) -> None:
    chunks = [_chunk("hello world", 0, 1, token_estimate=42)]
    stored = store.persist_chunks("source-x", chunks)
    chunk_id = stored[0].chunk_id

    fetched = store.get_chunk(chunk_id)
    assert fetched is not None
    assert fetched.text == "hello world"
    assert fetched.token_estimate == 42
    assert fetched.source_hash == "source-x"


def test_get_chunk_returns_none_for_unknown_id(store: ChunkStore) -> None:
    assert store.get_chunk("does-not-exist") is None


def test_get_chunks_for_source_returns_ordered_rows(store: ChunkStore) -> None:
    # Insert out of order to prove the query orders by chunk_index.
    store.persist_chunks(
        "ordered",
        [
            _chunk("two", 2, 3),
            _chunk("zero", 0, 3),
            _chunk("one", 1, 3),
        ],
    )
    chunks = store.get_chunks_for_source("ordered")
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.text for c in chunks] == ["zero", "one", "two"]


# ----------------------------------------------------------------------
# delete_by_source
# ----------------------------------------------------------------------


def test_delete_by_source_removes_matching_rows(store: ChunkStore) -> None:
    store.persist_chunks("target", [_chunk("A", 0, 2), _chunk("B", 1, 2)])
    store.persist_chunks("other", [_chunk("C", 0, 1)])

    removed = store.delete_by_source("target")
    assert removed == 2
    assert store.count() == 1
    assert store.get_chunks_for_source("target") == []
    assert len(store.get_chunks_for_source("other")) == 1


def test_delete_by_source_missing_source_is_noop(store: ChunkStore) -> None:
    assert store.delete_by_source("never-seen") == 0


# ----------------------------------------------------------------------
# Re-ingest stability
# ----------------------------------------------------------------------


def test_reingest_same_source_produces_stable_chunk_ids(
    store: ChunkStore,
) -> None:
    """Chunk IDs should survive a re-ingest of the same source.

    This is the invariant that makes page-frontmatter chunk_ids
    reliable across --force re-ingests: as long as the source bytes
    are unchanged, every chunk_id references the same chunk.
    """
    first = store.persist_chunks(
        "stable-source",
        [_chunk("one", 0, 2), _chunk("two", 1, 2)],
    )
    first_ids = [c.chunk_id for c in first]

    second = store.persist_chunks(
        "stable-source",
        [_chunk("one", 0, 2), _chunk("two", 1, 2)],
    )
    second_ids = [c.chunk_id for c in second]

    assert first_ids == second_ids
