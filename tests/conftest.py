"""Shared pytest fixtures and isolation hooks."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_embedder_cache() -> None:
    """Reset the module-level embedder cache between tests.

    ``load_embedder`` caches the loaded backend keyed by
    ``(provider, model)`` so a single process pays the cold-load tax
    once. Tests run in one process, so without this fixture a cached
    embedder from an earlier test would persist into a later test that
    points at a different project config and expects a fresh load.
    """
    from wikiloom.embeddings import clear_embedder_cache

    clear_embedder_cache()
    yield
    clear_embedder_cache()
