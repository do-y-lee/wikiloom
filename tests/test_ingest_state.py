"""Tests for the per-ingest resume checkpoint (wikiloom/ingest/state.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wikiloom.ingest.state import STATE_FILENAME, ChunkState, IngestState


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    d = tmp_path / "_registry"
    d.mkdir()
    return d


def _sample_chunks() -> list[ChunkState]:
    return [
        ChunkState(index=0, total=3, token_estimate=100),
        ChunkState(index=1, total=3, token_estimate=120),
        ChunkState(index=2, total=3, token_estimate=90),
    ]


# ----------------------------------------------------------------------
# begin / save / load
# ----------------------------------------------------------------------


def test_begin_creates_state_file(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="abc123",
        source_name="paper.pdf",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    assert state.state_path.exists()
    assert state.state_path.name == STATE_FILENAME

    data = json.loads(state.state_path.read_text())
    assert data["source_key"] == "abc123"
    assert data["source_name"] == "paper.pdf"
    assert data["content_type"] == "pdf"
    assert len(data["chunks"]) == 3
    assert data["chunks"][0]["done"] is False


def test_load_returns_none_when_missing(registry_dir: Path) -> None:
    assert IngestState.load(registry_dir) is None


def test_load_returns_none_on_corrupt_file(registry_dir: Path) -> None:
    (registry_dir / STATE_FILENAME).write_text("{not json")
    assert IngestState.load(registry_dir) is None


def test_load_roundtrips_state(registry_dir: Path) -> None:
    IngestState.begin(
        registry_dir=registry_dir,
        source_key="hash-xyz",
        source_name="doc.md",
        content_type="markdown",
        chunks=_sample_chunks(),
    )
    loaded = IngestState.load(registry_dir)
    assert loaded is not None
    assert loaded.source_key == "hash-xyz"
    assert loaded.source_name == "doc.md"
    assert loaded.content_type == "markdown"
    assert len(loaded.chunks) == 3
    assert loaded.chunks[1].token_estimate == 120


# ----------------------------------------------------------------------
# mark_chunk_done / failed / pending / is_complete
# ----------------------------------------------------------------------


def test_mark_chunk_done_persists(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="k",
        source_name="n",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    state.mark_chunk_done(1, page_id="concepts/foo")

    reloaded = IngestState.load(registry_dir)
    assert reloaded is not None
    assert reloaded.chunks[1].done is True
    assert reloaded.chunks[1].page_id == "concepts/foo"
    assert reloaded.chunks[0].done is False


def test_mark_chunk_failed_records_error(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="k",
        source_name="n",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    state.mark_chunk_failed(2, "RateLimit: 429")

    reloaded = IngestState.load(registry_dir)
    assert reloaded is not None
    assert reloaded.chunks[2].error == "RateLimit: 429"
    assert reloaded.chunks[2].done is False


def test_pending_indices_returns_unfinished_chunks(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="k",
        source_name="n",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    state.mark_chunk_done(0)
    state.mark_chunk_done(1)
    assert state.pending_indices() == [2]


def test_is_complete_true_only_when_all_done(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="k",
        source_name="n",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    assert not state.is_complete()
    state.mark_chunk_done(0)
    state.mark_chunk_done(1)
    assert not state.is_complete()
    state.mark_chunk_done(2)
    assert state.is_complete()


def test_is_complete_true_for_empty_chunks(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="k",
        source_name="n",
        content_type="pdf",
        chunks=[],
    )
    assert state.is_complete()


# ----------------------------------------------------------------------
# clear
# ----------------------------------------------------------------------


def test_clear_removes_state_file(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="k",
        source_name="n",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    assert state.state_path.exists()
    state.clear()
    assert not state.state_path.exists()


def test_clear_is_idempotent(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="k",
        source_name="n",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    state.clear()
    state.clear()  # no error on second call


# ----------------------------------------------------------------------
# matches
# ----------------------------------------------------------------------


def test_matches_detects_same_source(registry_dir: Path) -> None:
    state = IngestState.begin(
        registry_dir=registry_dir,
        source_key="hash-A",
        source_name="paper.pdf",
        content_type="pdf",
        chunks=_sample_chunks(),
    )
    assert state.matches("hash-A")
    assert not state.matches("hash-B")
    assert not state.matches("")
