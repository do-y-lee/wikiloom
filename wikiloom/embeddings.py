"""Embedding provider abstraction.

Turns text into dense vectors for semantic similarity search. Used
by the query pipeline as a fallback when FTS5 keyword search finds
no results — the query is embedded and compared against page
embeddings via cosine similarity.

Three backends, configured via ``[embeddings]`` in ``wikiloom.toml``:

- **fastembed** (default): local ONNX model, ~100MB install, free,
  no API key. ``BAAI/bge-small-en-v1.5`` by default.
- **openai**: API-based, ~1MB install, needs ``OPENAI_API_KEY``.
  ``text-embedding-3-small`` by default. Industry standard.
- **sentence-transformers**: local PyTorch model, ~800MB+ install,
  free. ``all-MiniLM-L6-v2`` by default. Most popular local lib.

Design notes
------------
- **Lazy imports.** Each backend is imported only when selected so
  users don't pay the install cost for providers they don't use. A
  missing dependency produces a clear error with install instructions.
- **Single interface.** ``get_embedder(config)`` returns an
  ``Embedder`` with one method: ``embed_texts(texts) → vectors``.
  The query and cache modules don't know which backend is active.
- **Vectors are lists of floats, not numpy arrays.** Avoids a numpy
  dependency for serialization to SQLite BLOBs. Cosine similarity
  is computed with plain Python math — fast enough at wiki scale.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Protocol


@dataclass
class EmbeddingConfig:
    """Mirrors the ``[embeddings]`` section of ``wikiloom.toml``."""

    provider: str = "fastembed"
    model: str = ""  # empty = use the provider's default
    enabled: bool = True


class Embedder(Protocol):
    """Interface that all embedding backends implement."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


# ----------------------------------------------------------------------
# Backend implementations
# ----------------------------------------------------------------------


class FastEmbedBackend:
    """ONNX-based local embeddings via fastembed."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model: str = "") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is not installed. Run: pip install fastembed"
            )
        self._model = TextEmbedding(model or self.DEFAULT_MODEL)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(v) for v in self._model.embed(texts)]


class OpenAIBackend:
    """API-based embeddings via OpenAI."""

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, model: str = "") -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            )
        self._client = OpenAI()
        self._model = model or self.DEFAULT_MODEL

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]


class SentenceTransformersBackend:
    """PyTorch-based local embeddings via sentence-transformers."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. Run: "
                "pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model or self.DEFAULT_MODEL)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

_BACKENDS = {
    "fastembed": FastEmbedBackend,
    "openai": OpenAIBackend,
    "sentence-transformers": SentenceTransformersBackend,
}


def get_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    """Create an embedder from config. Raises on unknown provider."""
    if config is None:
        config = EmbeddingConfig()
    provider = config.provider.lower().strip()
    backend_cls = _BACKENDS.get(provider)
    if backend_cls is None:
        raise ValueError(
            f"Unknown embedding provider {provider!r}. "
            f"Options: {', '.join(sorted(_BACKENDS))}"
        )
    return backend_cls(model=config.model)


# ----------------------------------------------------------------------
# Vector utilities
# ----------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns -1.0 to 1.0."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def serialize_embedding(vector: list[float]) -> bytes:
    """Pack a float vector into bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack a BLOB back into a float vector."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))
