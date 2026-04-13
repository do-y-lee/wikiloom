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


@pytest.fixture
def engine(registry: Registry) -> LinkingEngine:
    if not HAS_MODEL:
        pytest.skip("en_core_web_sm not installed")
    return LinkingEngine(registry, LinkingConfig(auto_create_stubs=False))


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
    eng = LinkingEngine(registry, LinkingConfig(auto_create_stubs=True))
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
    eng = LinkingEngine(registry, LinkingConfig(auto_create_stubs=True))
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


# ----------------------------------------------------------------------
# Stub creation
# ----------------------------------------------------------------------


@requires_model
def test_create_stubs_writes_files_and_registers(registry: Registry, project: Path) -> None:
    eng = LinkingEngine(registry, LinkingConfig(auto_create_stubs=True))
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
    p = tmp_path / "wiki" / "concepts" / "attention.md"
    assert LinkingEngine._path_to_id(p) == "concepts/attention"


# ----------------------------------------------------------------------
# Full link_page integration
# ----------------------------------------------------------------------


@requires_model
def test_link_page_writes_back_with_frontmatter(registry: Registry, project: Path) -> None:
    eng = LinkingEngine(registry, LinkingConfig(auto_create_stubs=False))

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
