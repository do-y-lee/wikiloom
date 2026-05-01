"""Shared pytest fixtures and isolation hooks."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_embedder_cache() -> None:
    """Isolate ``load_embedder``'s module-level cache between tests."""
    from wikiloom.embeddings import clear_embedder_cache

    clear_embedder_cache()
    yield
    clear_embedder_cache()
