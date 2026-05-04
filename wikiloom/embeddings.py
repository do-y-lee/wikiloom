"""Embedding provider abstraction.

Three backends (fastembed, openai, sentence-transformers) configured
via [embeddings] in wikiloom.toml. Lazy-imported so only the selected
provider needs to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


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


def fastembed_cache_dir() -> Path:
    """Durable, per-user cache for fastembed ONNX models.

    fastembed's default cache is ``tempfile.gettempdir()`` which on
    macOS resolves to ``/var/folders/.../T/`` — a directory the OS
    reaps on a schedule. Reaping is per-file, so the snapshot dir can
    survive while the large ONNX file inside it disappears, leaving
    fastembed to crash with a cryptic ``NO_SUCHFILE`` on next load.
    Routing the cache through ``platformdirs`` lands it in
    ``~/Library/Caches/wikiloom/fastembed`` (macOS),
    ``~/.cache/wikiloom/fastembed`` (Linux), or
    ``%LOCALAPPDATA%\\wikiloom\\Cache\\fastembed`` (Windows) — none
    of which the OS auto-cleans.
    """
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("wikiloom")) / "fastembed"


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
        cache_dir = fastembed_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(
            model or self.DEFAULT_MODEL,
            cache_dir=str(cache_dir),
        )

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


_cached_embedder: Embedder | None = None
_cached_embedder_key: tuple[str, str] | None = None


def load_embedder(project: Path) -> Embedder | None:
    """Read project config and return an embedder if embeddings are enabled.

    Returns ``None`` when embeddings are disabled, the config file is
    missing, or the backend fails to import. Callers can pass the result
    straight into ``SQLiteCache.full_rebuild`` / ``sync_from_files``.

    Cached at module level keyed by ``(provider, model)``;
    ``clear_embedder_cache()`` resets for tests.
    """
    global _cached_embedder, _cached_embedder_key

    try:
        from wikiloom.config import Config

        cfg = Config.load(project)
    except (FileNotFoundError, ValueError):
        return None

    if not cfg.embeddings.enabled:
        return None

    key = (cfg.embeddings.provider, cfg.embeddings.model)
    if _cached_embedder is not None and _cached_embedder_key == key:
        return _cached_embedder

    try:
        embedder = get_embedder(cfg.embeddings)
    except (ImportError, ValueError):
        return None

    _cached_embedder = embedder
    _cached_embedder_key = key
    return embedder


def clear_embedder_cache() -> None:
    """Reset the module-level ``load_embedder`` cache."""
    global _cached_embedder, _cached_embedder_key
    _cached_embedder = None
    _cached_embedder_key = None


# ----------------------------------------------------------------------
# Vector utilities
# ----------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns -1.0 to 1.0."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def serialize_embedding(vector: list[float]) -> bytes:
    """Pack a float vector into bytes for SQLite BLOB storage."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack a BLOB back into a float vector."""
    return np.frombuffer(blob, dtype=np.float32).tolist()
