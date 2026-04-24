"""Tests for wikiloom.linker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
    HAS_MODEL = True
except OSError:  # pragma: no cover
    HAS_MODEL = False

requires_model = pytest.mark.skipif(
    not HAS_MODEL, reason="en_core_web_sm not installed"
)

from wikiloom.config import LinkingConfig
from wikiloom.linker import (
    LinkingEngine,
    PendingLink,
    ResolvedLink,
    UnresolvedEntity,
)
from wikiloom.registry import PageEntry, Registry


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "_registry").mkdir()
    (tmp_path / "wiki" / "entities").mkdir(parents=True)
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def registry(project: Path) -> Registry:
    reg = Registry(project / "_registry")
    reg.register_page(
        "entities/google-brain",
        PageEntry(
            title="Google Brain",
            type="entity",
            aliases=["google brain", "google ai research"],
            summary="AI research division of Google.",
        ),
    )
    reg.register_page(
        "concepts/flash-attention",
        PageEntry(
            title="Flash Attention",
            type="concept",
            aliases=["flash attention", "flash-attn"],
            summary="Memory-efficient attention algorithm.",
        ),
    )
    reg.register_page(
        "concepts/attention",
        PageEntry(
            title="Attention",
            type="concept",
            aliases=["attention mechanism"],
            summary="Self-attention mechanism.",
        ),
    )
    return reg


class _StubEmbedder:
    """Cheapest possible embedder — returns a fixed vector for any input.

    The linker requires an embedder, but most tests don't actually
    exercise the cosine rerank path — they poke helpers like
    ``_resolve`` or ``_compute_safe_zones`` directly, or they
    exercise exact-alias hits that short-circuit before the embedder
    runs. This stub satisfies the constructor without pulling any
    real model.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _StubCache:
    """Matches the SQLiteCache surface the linker actually touches.

    When ``from_registry`` is passed, every registered page gets a
    unit vector so cosine rerank picks up the fuzzy top-1. That's
    what most non-hybrid-specific tests want — they only care about
    which bucket the match lands in, not the numerical cosine value.
    Tests exercising the rerank logic pass explicit vectors instead.
    """

    def __init__(
        self,
        embeddings: dict[str, list[float]] | None = None,
        *,
        from_registry: Registry | None = None,
    ) -> None:
        self._embeddings = dict(embeddings or {})
        if from_registry is not None:
            for page_id in from_registry.pages:
                self._embeddings.setdefault(page_id, [1.0, 0.0, 0.0])

    def load_page_embeddings(self) -> dict[str, list[float]]:
        return dict(self._embeddings)


@pytest.fixture
def engine(registry: Registry) -> LinkingEngine:
    if not HAS_MODEL:
        pytest.skip("en_core_web_sm not installed")
    return LinkingEngine(
        registry,
        embedder=_StubEmbedder(),
        cache=_StubCache(from_registry=registry),
        config=LinkingConfig(auto_create_stubs=False),
    )


# ----------------------------------------------------------------------
# _resolve — exact + fuzzy + thresholds
# ----------------------------------------------------------------------


@requires_model
def test_resolve_exact_match(engine: LinkingEngine) -> None:
    match = engine._resolve("Flash Attention")
    assert match is not None
    assert match.page_id == "concepts/flash-attention"
    assert match.score == 100
    assert match.method == "exact"


@requires_model
def test_resolve_alias_exact(engine: LinkingEngine) -> None:
    match = engine._resolve("flash-attn")
    assert match is not None
    assert match.page_id == "concepts/flash-attention"


@requires_model
def test_resolve_fuzzy_match(engine: LinkingEngine) -> None:
    match = engine._resolve("Google Brain Team")
    assert match is not None
    assert match.page_id == "entities/google-brain"
    assert match.method == "fuzzy"


@requires_model
def test_resolve_returns_none_below_threshold(engine: LinkingEngine) -> None:
    match = engine._resolve("Completely Unrelated Term Salad")
    assert match is None


# ----------------------------------------------------------------------
# Safe zones — code blocks, wikilinks, headings, etc.
# ----------------------------------------------------------------------


@requires_model
def test_safe_zones_excludes_fenced_code(engine: LinkingEngine) -> None:
    body = "Hello Flash Attention.\n\n```\ncode block with Flash Attention\n```\n\nMore text."
    zones = engine._compute_safe_zones(body)
    # Find character index of "code block"
    code_pos = body.find("code block")
    assert not any(start <= code_pos < end for start, end in zones)


@requires_model
def test_safe_zones_excludes_inline_code(engine: LinkingEngine) -> None:
    body = "Use `Flash Attention` here."
    zones = engine._compute_safe_zones(body)
    inline_pos = body.find("`Flash") + 1
    assert not any(start <= inline_pos < end for start, end in zones)


@requires_model
def test_safe_zones_excludes_existing_wikilinks(engine: LinkingEngine) -> None:
    body = "See [[concepts/flash-attention|Flash Attention]] for details."
    zones = engine._compute_safe_zones(body)
    pos = body.find("flash-attention")
    assert not any(start <= pos < end for start, end in zones)


@requires_model
def test_safe_zones_excludes_headings(engine: LinkingEngine) -> None:
    body = "# Flash Attention\n\nBody content."
    zones = engine._compute_safe_zones(body)
    heading_pos = body.find("Flash")
    assert not any(start <= heading_pos < end for start, end in zones)


@requires_model
def test_safe_zones_includes_normal_text(engine: LinkingEngine) -> None:
    body = "Just some regular paragraph text."
    zones = engine._compute_safe_zones(body)
    assert zones == [(0, len(body))]


@requires_model
def test_safe_zones_excludes_tilde_fence(engine: LinkingEngine) -> None:
    body = "Before.\n\n~~~python\nFlash Attention here\n~~~\n\nAfter."
    zones = engine._compute_safe_zones(body)
    pos = body.find("Flash Attention here")
    assert not any(start <= pos < end for start, end in zones)


@requires_model
def test_safe_zones_excludes_language_tagged_fence(engine: LinkingEngine) -> None:
    body = "Intro.\n\n```python\ndef foo(): return 'Flash Attention'\n```\n\nOutro."
    zones = engine._compute_safe_zones(body)
    pos = body.find("'Flash Attention'") + 1
    assert not any(start <= pos < end for start, end in zones)


@requires_model
def test_safe_zones_excludes_reference_link(engine: LinkingEngine) -> None:
    body = "See [Flash Attention][fa] for details.\n\n[fa]: https://example.com/fa"
    zones = engine._compute_safe_zones(body)
    # Text inside [Flash Attention][fa] must not be a safe zone
    pos = body.find("Flash Attention")
    assert not any(start <= pos < end for start, end in zones)
    # Reference definition line is also unsafe
    def_pos = body.find("[fa]: ")
    assert not any(start <= def_pos < end for start, end in zones)


@requires_model
def test_safe_zones_excludes_html_pre_block(engine: LinkingEngine) -> None:
    body = "Para.\n\n<pre>\nFlash Attention in pre\n</pre>\n\nMore."
    zones = engine._compute_safe_zones(body)
    pos = body.find("Flash Attention in pre")
    assert not any(start <= pos < end for start, end in zones)


@requires_model
def test_safe_zones_excludes_inline_html_code(engine: LinkingEngine) -> None:
    body = "Use <code>Flash Attention</code> here."
    zones = engine._compute_safe_zones(body)
    pos = body.find("Flash Attention")
    assert not any(start <= pos < end for start, end in zones)


# ----------------------------------------------------------------------
# Stub-duplication regression tests
# ----------------------------------------------------------------------


@requires_model
def test_first_mention_variant_does_not_create_duplicate_stub(
    registry: Registry, project: Path
) -> None:
    """A second surface form of an already-linked entity must not become a stub."""
    eng = LinkingEngine(
        registry,
        embedder=_StubEmbedder(),
        cache=_StubCache(from_registry=registry),
        config=LinkingConfig(auto_create_stubs=True),
    )
    body = (
        "Flash Attention is fast. We benchmarked Flash Attentions on long sequences."
    )
    result = eng._link_text(body, source_page_id="sources/paper")
    # The plural "Flash Attentions" must not appear in unresolved
    assert not any("flash attention" in u.text.lower() for u in result.unresolved)


@requires_model
def test_self_mention_variant_does_not_create_duplicate_stub(
    registry: Registry,
) -> None:
    """The page about an entity must not generate a stub for its own variants."""
    eng = LinkingEngine(
        registry,
        embedder=_StubEmbedder(),
        cache=_StubCache(from_registry=registry),
        config=LinkingConfig(auto_create_stubs=True),
    )
    body = "Flash Attention is fast. People also call it flash-attn."
    result = eng._link_text(body, source_page_id="concepts/flash-attention")
    # Neither variant of Flash Attention should be unresolved
    assert not any("flash" in u.text.lower() for u in result.unresolved)


# ----------------------------------------------------------------------
# _insert_links
# ----------------------------------------------------------------------


@requires_model
def test_insert_links_single(engine: LinkingEngine) -> None:
    body = "We use Flash Attention here."
    link = ResolvedLink(
        original_text="Flash Attention",
        page_id="concepts/flash-attention",
        score=100,
        confidence="high",
        start=body.index("Flash Attention"),
        end=body.index("Flash Attention") + len("Flash Attention"),
    )
    out = engine._insert_links(body, [link])
    assert out == "We use [[concepts/flash-attention|Flash Attention]] here."


@requires_model
def test_insert_links_multiple_reverse_order(engine: LinkingEngine) -> None:
    body = "Google Brain built Flash Attention."
    g_start = body.index("Google Brain")
    f_start = body.index("Flash Attention")
    links = [
        ResolvedLink("Google Brain", "entities/google-brain", 100, "high", g_start, g_start + 12),
        ResolvedLink("Flash Attention", "concepts/flash-attention", 100, "high", f_start, f_start + 15),
    ]
    out = engine._insert_links(body, links)
    assert "[[entities/google-brain|Google Brain]]" in out
    assert "[[concepts/flash-attention|Flash Attention]]" in out
    assert "built" in out


# ----------------------------------------------------------------------
# End-to-end _link_text
# ----------------------------------------------------------------------


@requires_model
def test_link_text_inserts_high_confidence_link(engine: LinkingEngine) -> None:
    body = "We used Flash Attention to speed up training."
    result = engine._link_text(body, source_page_id="sources/some-paper")
    assert any(l.page_id == "concepts/flash-attention" for l in result.high_links)
    assert "[[concepts/flash-attention|" in result.body


@requires_model
def test_link_text_no_self_links(engine: LinkingEngine) -> None:
    body = "Flash Attention is described here."
    result = engine._link_text(body, source_page_id="concepts/flash-attention")
    assert all(l.page_id != "concepts/flash-attention" for l in result.high_links)
    assert "[[concepts/flash-attention" not in result.body


@requires_model
def test_link_text_first_mention_only(engine: LinkingEngine) -> None:
    body = (
        "Flash Attention is fast. Flash Attention is also memory-efficient. "
        "We love Flash Attention."
    )
    result = engine._link_text(body, source_page_id="sources/paper")
    assert result.body.count("[[concepts/flash-attention|") == 1


@requires_model
def test_link_text_skips_code_blocks(engine: LinkingEngine) -> None:
    body = "```\nFlash Attention in code\n```\nNo links here."
    result = engine._link_text(body, source_page_id="sources/paper")
    assert "[[concepts/flash-attention" not in result.body


@requires_model
def test_link_text_filters_stopwords(engine: LinkingEngine) -> None:
    """Common words should never get linked even if they match."""
    # Add a dummy page titled "data" to the registry to ensure stop-list works
    engine.registry.register_page(
        "concepts/data",
        PageEntry(title="data", type="concept", aliases=["data"]),
    )
    engine.refresh()
    body = "The data is important."
    result = engine._link_text(body, source_page_id="sources/paper")
    assert "[[concepts/data" not in result.body


# ----------------------------------------------------------------------
# Pending link persistence
# ----------------------------------------------------------------------


@requires_model
def test_save_pending_writes_file(engine: LinkingEngine, project: Path) -> None:
    pending = [
        PendingLink(
            source_page="sources/paper",
            matched_text="Flash Attention v2",
            candidate_page_id="concepts/flash-attention",
            score=78,
            label="CONCEPT",
        )
    ]
    engine._save_pending(pending)
    pending_path = project / "_registry" / "pending.json"
    assert pending_path.exists()
    data = json.loads(pending_path.read_text())
    assert len(data["pending"]) == 1
    assert data["pending"][0]["score"] == 78
    assert data["pending"][0]["candidate_page_id"] == "concepts/flash-attention"


@requires_model
def test_save_pending_appends(engine: LinkingEngine, project: Path) -> None:
    pending_path = project / "_registry" / "pending.json"
    pending_path.write_text(json.dumps({"version": 1, "pending": [{"existing": True}]}))

    engine._save_pending([
        PendingLink("p", "x", "concepts/y", 75, "CONCEPT")
    ])
    data = json.loads(pending_path.read_text())
    assert len(data["pending"]) == 2


@requires_model
def test_save_pending_serializes_numpy_scores(
    engine: LinkingEngine, project: Path
) -> None:
    """Cosine scores come from numpy ops (float32) and must serialize.

    Regression: round() on a numpy scalar returns another numpy scalar,
    which json.dumps rejects with TypeError. Mid-batch ingest in a
    real project hit this on the second file, after the first file
    populated the page-embeddings cache.
    """
    import numpy as np

    pending = [
        PendingLink(
            source_page="sources/paper",
            matched_text="span",
            candidate_page_id="concepts/x",
            score=np.int64(82),  # rapidfuzz can return numpy scalars
            label="CONCEPT",
            cosine_score=np.float32(0.8765),
        )
    ]
    engine._save_pending(pending)
    pending_path = project / "_registry" / "pending.json"
    data = json.loads(pending_path.read_text())
    assert data["pending"][0]["score"] == 82
    assert data["pending"][0]["cosine_score"] == pytest.approx(0.8765, abs=1e-4)


# ----------------------------------------------------------------------
# Stub creation
# ----------------------------------------------------------------------


@requires_model
def test_create_stubs_writes_files_and_registers(registry: Registry, project: Path) -> None:
    eng = LinkingEngine(
        registry,
        embedder=_StubEmbedder(),
        cache=_StubCache(from_registry=registry),
        config=LinkingConfig(auto_create_stubs=True),
    )
    unresolved = [UnresolvedEntity(text="LoRA", label="ORG")]
    created = eng._create_stubs(unresolved)
    assert created == 1
    stub_path = project / "wiki" / "entities" / "lora.md"
    assert stub_path.exists()
    content = stub_path.read_text()
    assert "LoRA" in content
    assert "stub" in content
    assert registry.get_page("entities/lora") is not None


# ----------------------------------------------------------------------
# Path → page_id helper
# ----------------------------------------------------------------------


def test_path_to_id_extracts_relative_id(tmp_path: Path) -> None:
    from wikiloom.utils import page_id_from_path

    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    p = tmp_path / "wiki" / "concepts" / "attention.md"
    p.write_text("body")
    assert page_id_from_path(tmp_path / "wiki", p) == "concepts/attention"


# ----------------------------------------------------------------------
# Full link_page integration
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Hybrid linker (fuzzy pre-filter → cosine rerank)
# ----------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic embedder: maps substrings to pre-set vectors.

    Falls back to a zero vector so any input that matches no rule
    scores cosine 0 — an unambiguous "no match" for the linker.
    """

    def __init__(self, rules: list[tuple[str, list[float]]]) -> None:
        self._rules = rules

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0, 0.0, 0.0]
            for needle, v in self._rules:
                if needle.lower() in text.lower():
                    vec = v
                    break
            out.append(vec)
        return out


class _FakeCache:
    """Stand-in for SQLiteCache exposing only load_page_embeddings."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def load_page_embeddings(self) -> dict[str, list[float]]:
        return dict(self._embeddings)


@requires_model
def test_hybrid_cosine_drops_fuzzy_false_positive(
    registry: Registry, project: Path
) -> None:
    """Cosine rerank should kill a fuzzy-matching but semantically-wrong candidate.

    Exercises ``_resolve_with_rerank`` directly so spaCy's entity
    ruler (which would exact-hit registered aliases) isn't in the
    way. The span here is a deliberately-fuzzy variant of a
    registered alias; the page's embedding is orthogonal to the
    span-context embedding. Cosine lands at 0.0, ``_bucket_for``
    returns "drop", and the match carries a near-zero cosine_score
    so callers can see why.
    """
    embedder = _FakeEmbedder(
        rules=[("google brain team", [1.0, 0.0, 0.0])]
    )
    cache = _FakeCache(
        embeddings={"entities/google-brain": [0.0, 1.0, 0.0]}
    )
    eng = LinkingEngine(
        registry,
        LinkingConfig(auto_create_stubs=False),
        embedder=embedder,
        cache=cache,
    )

    body = "the Google Brain Team announced results today"
    match = eng._resolve_with_rerank("Google Brain Team", body, 4, 21)
    assert match is not None
    assert match.method == "hybrid"
    assert match.page_id == "entities/google-brain"
    assert match.cosine_score == pytest.approx(0.0)
    assert eng._bucket_for(match) == "drop"


@requires_model
def test_hybrid_cosine_promotes_strong_semantic_match(
    registry: Registry, project: Path
) -> None:
    """Cosine near 1.0 should promote a fuzzy-only pending candidate to a link.

    A cosine of 1.0 lands the candidate in the high bucket regardless
    of where ``cosine_high_threshold`` is set, even when the fuzzy
    score alone would have fallen under the fuzzy high threshold.
    """
    # Span context and page embedding point the same direction →
    # cosine similarity is 1.0, clearly above any configured cosine
    # threshold.
    embedder = _FakeEmbedder(
        rules=[("google brain team", [1.0, 0.0, 0.0])]
    )
    cache = _FakeCache(
        embeddings={"entities/google-brain": [1.0, 0.0, 0.0]}
    )
    eng = LinkingEngine(
        registry,
        LinkingConfig(auto_create_stubs=False),
        embedder=embedder,
        cache=cache,
    )

    body = "The Google Brain Team published a paper on attention mechanisms."
    result = eng._link_text(body, source_page_id="sources/paper")

    high_targets = {l.page_id for l in result.high_links}
    assert "entities/google-brain" in high_targets
    # Cosine score should be populated on the resolved link.
    matched = next(
        l for l in result.high_links if l.page_id == "entities/google-brain"
    )
    assert matched.cosine_score is not None
    assert matched.cosine_score > 0.99


@requires_model
def test_hybrid_exact_alias_skips_cosine(
    registry: Registry, project: Path
) -> None:
    """Exact alias hits shouldn't burn an embedder call.

    The embedder below would throw on any call, proving the exact-
    match fast path short-circuits before reaching it.
    """

    class _ExplodingEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError(
                "embed_texts should not be called on an exact alias hit"
            )

    cache = _FakeCache(embeddings={})
    eng = LinkingEngine(
        registry,
        LinkingConfig(auto_create_stubs=False),
        embedder=_ExplodingEmbedder(),
        cache=cache,
    )

    # "Flash Attention" is a registered page title — exact alias hit.
    match = eng._resolve_with_rerank("Flash Attention", "body", 0, 15)
    assert match is not None
    assert match.method == "exact"
    assert match.cosine_score is None


@requires_model
def test_link_page_writes_back_with_frontmatter(registry: Registry, project: Path) -> None:
    eng = LinkingEngine(
        registry,
        embedder=_StubEmbedder(),
        cache=_StubCache(from_registry=registry),
        config=LinkingConfig(auto_create_stubs=False),
    )

    page_path = project / "wiki" / "sources" / "paper.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        """---
title: "Paper"
type: source
status: active
created: "2026-04-12T00:00:00Z"
modified: "2026-04-12T00:00:00Z"
summary: "A paper"
---

We tried Flash Attention and it sped up training significantly.
""",
        encoding="utf-8",
    )

    result = eng.link_page(page_path)
    assert result.high_confidence_links + result.medium_confidence_links >= 1

    written = page_path.read_text()
    # Frontmatter preserved
    assert "title: Paper" in written or 'title: "Paper"' in written
    # Wikilink inserted
    assert "[[concepts/flash-attention|" in written
