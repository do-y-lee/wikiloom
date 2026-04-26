"""Tests for the synthesis loop (wikiloom/synthesis.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.ingest.state import ChunkState, IngestState
from wikiloom.llm import LLMCallMetrics, SynthesizeResult
from wikiloom.llm_errors import LLMProviderError, LLMResponseFormatError
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project
from wikiloom.synthesis import (
    PageProposal,
    SourceSummary,
    SynthesisResult,
    load_prompt,
    render_manifest_context,
    render_namespace_list,
    run_synthesis,
    validate_ingest_response,
)


# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return init_project(name="synth-test", path=tmp_path, domain="test")


@pytest.fixture
def registry(project: Path) -> Registry:
    return Registry(project / "_registry")


def _chunk(text: str, index: int, total: int, content_type: str = "markdown") -> ExtractedContent:
    return ExtractedContent(
        text=text,
        metadata={"chunk_index": index, "chunk_total": total},
        source_path=None,
        content_type=content_type,
        extraction_method="test",
        token_estimate=max(1, len(text) // 4),
    )


def _ok_response(
    *,
    source_title: str = "Some Doc",
    creates: list[dict[str, Any]] | None = None,
    updates: list[dict[str, Any]] | None = None,
    entities: list[str] | None = None,
    concepts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source_summary": {
            "title": source_title,
            "one_line": f"A one-line description of {source_title}.",
            "content_markdown": f"## About {source_title}\n\nSome content.",
        },
        "pages_to_create": creates or [],
        "pages_to_update": updates or [],
        "entities_mentioned": entities or [],
        "concepts_mentioned": concepts or [],
    }


def _create_page(
    slug: str,
    title: str | None = None,
    type_: str = "concept",
    confidence: str = "high",
) -> dict[str, Any]:
    return {
        "type": type_,
        "suggested_slug": slug,
        "title": title or slug.replace("-", " ").title(),
        "content_markdown": f"# {title or slug}\n\nSynthesized body for {slug}.",
        "confidence": confidence,
    }


def _mock_llm_client(responses: list[dict[str, Any] | Exception]) -> MagicMock:
    """Build a mock LLMClient whose synthesize returns each response in order."""
    client = MagicMock()
    call_results: list[Any] = []
    for r in responses:
        if isinstance(r, Exception):
            call_results.append(r)
        else:
            call_results.append(
                SynthesizeResult(
                    result=r,
                    metrics=LLMCallMetrics(
                        tokens_in=100,
                        tokens_out=200,
                        cost_usd=0.0015,
                        model="claude-sonnet-4-20250514",
                    ),
                )
            )

    def side_effect(*args: Any, **kwargs: Any) -> Any:
        nxt = call_results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    client.synthesize.side_effect = side_effect
    return client


# ----------------------------------------------------------------------
# load_prompt
# ----------------------------------------------------------------------


def test_load_prompt_uses_package_template_by_default(project: Path) -> None:
    """init_project copies the template into .wikiloom/prompts/ — that counts as the override."""
    prompt = load_prompt(project)
    assert "Ingestion Prompt" in prompt


def test_load_prompt_prefers_project_override(project: Path) -> None:
    override = project / ".wikiloom" / "prompts" / "ingest.md"
    override.write_text("# Custom project override\n\nDo different things.", encoding="utf-8")
    prompt = load_prompt(project)
    assert "Custom project override" in prompt


# ----------------------------------------------------------------------
# render_manifest_context
# ----------------------------------------------------------------------


def test_render_manifest_context_empty(project: Path, registry: Registry) -> None:
    ctx = render_manifest_context(registry)
    assert "empty" in ctx.lower()


def test_render_manifest_context_with_pages(project: Path, registry: Registry) -> None:
    registry.register_page(
        "concepts/transformer",
        PageEntry(
            title="Transformer",
            type="concept",
            summary="Attention-based sequence model.",
            aliases=["transformers"],
        ),
    )
    registry.register_page(
        "entities/openai",
        PageEntry(
            title="OpenAI",
            type="entity",
            summary="AI research organization.",
        ),
    )
    registry.save()

    ctx = render_manifest_context(registry)
    assert "concepts/transformer" in ctx
    assert "Transformer" in ctx
    assert "entities/openai" in ctx


def test_render_manifest_context_skips_deprecated(project: Path, registry: Registry) -> None:
    registry.register_page(
        "concepts/alive",
        PageEntry(title="Alive", type="concept"),
    )
    registry.register_page(
        "concepts/dead",
        PageEntry(title="Dead", type="concept", status="deprecated"),
    )
    registry.save()

    ctx = render_manifest_context(registry)
    assert "concepts/alive" in ctx
    assert "concepts/dead" not in ctx


def test_render_manifest_context_caps_at_max_pages(
    project: Path, registry: Registry
) -> None:
    for i in range(50):
        registry.register_page(
            f"concepts/page-{i:02d}",
            PageEntry(title=f"Page {i}", type="concept"),
        )
    registry.save()

    ctx = render_manifest_context(registry, max_pages=5)
    # header + separator + 5 rows
    assert ctx.count("\n") == 6


# ----------------------------------------------------------------------
# render_namespace_list
# ----------------------------------------------------------------------


def test_render_namespace_list_empty(project: Path, registry: Registry) -> None:
    rendered = render_namespace_list(registry)
    assert "no existing pages" in rendered.lower()


def test_render_namespace_list_covers_every_active_page(
    project: Path, registry: Registry
) -> None:
    registry.register_page(
        "concepts/alpha",
        PageEntry(title="Alpha", type="concept"),
    )
    registry.register_page(
        "entities/zeta-corp",
        PageEntry(title="Zeta Corp", type="entity"),
    )
    # Deprecated pages must be skipped — they're not valid targets.
    registry.register_page(
        "concepts/gone",
        PageEntry(title="Gone", type="concept", status="deprecated"),
    )
    registry.save()

    rendered = render_namespace_list(registry)
    lines = rendered.splitlines()

    # page_id only, no title column, no separator.
    assert "concepts/alpha" in lines
    assert "entities/zeta-corp" in lines
    assert "concepts/gone" not in lines
    assert "|" not in rendered  # titles dropped to halve token cost
    # Sorted for cache stability.
    assert lines.index("concepts/alpha") < lines.index("entities/zeta-corp")


def test_run_synthesis_embeds_namespace_in_system_prompt(
    project: Path, registry: Registry
) -> None:
    """The namespace list reaches the LLM as part of the (cached)
    system prompt so every chunk's synthesis call sees every
    existing page without paying per-chunk token cost."""
    registry.register_page(
        "concepts/alpha",
        PageEntry(title="Alpha", type="concept"),
    )
    registry.save()

    chunks = [_chunk("text", 0, 1)]
    chunk_ids = ["c1"]
    llm = _mock_llm_client([_ok_response()])

    run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    call = llm.synthesize.call_args_list[0]
    system_prompt_arg = call.args[0] if call.args else call.kwargs["system_prompt"]
    assert "## NAMESPACE" in system_prompt_arg
    # Namespace lines carry page_ids only (titles dropped to halve cost).
    assert "concepts/alpha" in system_prompt_arg
    assert "concepts/alpha | Alpha" not in system_prompt_arg


# ----------------------------------------------------------------------
# validate_ingest_response
# ----------------------------------------------------------------------


def test_validate_accepts_minimal_valid_response() -> None:
    response = _ok_response()
    assert validate_ingest_response(response) == []


def test_validate_rejects_non_dict() -> None:
    assert validate_ingest_response("not a dict") != []
    assert validate_ingest_response([1, 2, 3]) != []


def test_validate_rejects_missing_top_level_field() -> None:
    response = _ok_response()
    del response["pages_to_create"]
    errors = validate_ingest_response(response)
    assert any("pages_to_create" in e for e in errors)


def test_validate_rejects_bad_page_type_enum() -> None:
    response = _ok_response(
        creates=[_create_page("foo") | {"type": "invalid-type"}]
    )
    errors = validate_ingest_response(response)
    assert any("type" in e and "invalid" in e for e in errors)


def test_validate_rejects_bad_confidence_enum() -> None:
    response = _ok_response(
        creates=[_create_page("foo") | {"confidence": "absolute"}]
    )
    errors = validate_ingest_response(response)
    assert any("confidence" in e for e in errors)


def test_validate_rejects_missing_source_summary_field() -> None:
    response = _ok_response()
    del response["source_summary"]["one_line"]
    errors = validate_ingest_response(response)
    assert any("source_summary.one_line" in e for e in errors)


def test_validate_rejects_missing_update_fields() -> None:
    response = _ok_response(updates=[{"existing_path": "concepts/foo"}])
    errors = validate_ingest_response(response)
    assert any("additions_markdown" in e for e in errors)


# ----------------------------------------------------------------------
# run_synthesis — happy path
# ----------------------------------------------------------------------


def test_run_synthesis_single_chunk_produces_page(
    project: Path, registry: Registry
) -> None:
    chunks = [_chunk("Some source text about transformers.", 0, 1)]
    chunk_ids = ["aaa111"]
    llm = _mock_llm_client(
        [
            _ok_response(
                creates=[_create_page("transformer", "Transformer")],
                entities=["Google"],
                concepts=["attention"],
            )
        ]
    )

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert result.chunks_processed == 1
    assert result.chunks_failed == 0
    assert len(result.pages_to_create) == 1
    assert result.pages_to_create[0].title == "Transformer"
    assert result.pages_to_create[0].chunk_id == "aaa111"
    assert result.source_summary is not None
    assert result.entities_mentioned == ["Google"]
    assert result.concepts_mentioned == ["attention"]
    assert result.total_tokens_in == 100
    assert result.total_tokens_out == 200
    assert result.total_cost_usd == pytest.approx(0.0015)


def test_run_synthesis_aggregates_across_chunks(
    project: Path, registry: Registry
) -> None:
    chunks = [
        _chunk("chunk 1 text", 0, 2),
        _chunk("chunk 2 text", 1, 2),
    ]
    chunk_ids = ["c1", "c2"]
    llm = _mock_llm_client(
        [
            _ok_response(creates=[_create_page("alpha", "Alpha")]),
            _ok_response(
                creates=[_create_page("beta", "Beta")],
                updates=[
                    {
                        "existing_path": "concepts/alpha",
                        "additions_markdown": "More from chunk 2.",
                    }
                ],
            ),
        ]
    )

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert result.chunks_processed == 2
    titles = {p.title for p in result.pages_to_create}
    assert titles == {"Alpha", "Beta"}
    assert len(result.pages_to_update) == 1
    assert result.pages_to_update[0].existing_path == "concepts/alpha"
    assert result.pages_to_update[0].additions_markdown == "More from chunk 2."
    assert result.total_tokens_in == 200  # 2 chunks × 100
    assert result.total_cost_usd == pytest.approx(0.003)


def test_run_synthesis_dedupes_create_proposals_by_slug(
    project: Path, registry: Registry
) -> None:
    """Two chunks proposing the same (type, slug) — only the first wins."""
    chunks = [_chunk("text 1", 0, 2), _chunk("text 2", 1, 2)]
    chunk_ids = ["c1", "c2"]
    llm = _mock_llm_client(
        [
            _ok_response(creates=[_create_page("transformer", "Transformer")]),
            _ok_response(creates=[_create_page("transformer", "Transformer Duplicate")]),
        ]
    )

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert len(result.pages_to_create) == 1
    assert result.pages_to_create[0].title == "Transformer"
    assert result.pages_to_create[0].chunk_id == "c1"


def test_run_synthesis_redirects_near_duplicate_create_to_update(
    project: Path, registry: Registry
) -> None:
    """A proposed create whose page_id fuzzy-matches an existing manifest
    entry gets converted to an update of that entry — prevents duplicate
    pages like ``account-closure-procedure`` / ``account-closure-procedures``
    from coexisting just because the LLM emitted a slight slug variant."""
    # Seed an existing page in the manifest.
    registry.register_page(
        "concepts/account-closure-procedures",
        PageEntry(
            title="Account Closure Procedures",
            type="concept",
            status="active",
            created="2026-01-01T00:00:00Z",
            modified="2026-01-01T00:00:00Z",
            summary="Procedures for closing an account.",
        ),
    )
    registry.save()

    chunks = [_chunk("text 1", 0, 1)]
    chunk_ids = ["c1"]
    llm = _mock_llm_client(
        [
            _ok_response(
                creates=[
                    _create_page(
                        "account-closure-procedure",
                        title="Account Closure Procedure",
                    )
                ]
            ),
        ]
    )

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert result.pages_to_create == []  # create was diverted
    assert len(result.pages_to_update) == 1
    update = result.pages_to_update[0]
    assert update.intent == "update"
    assert update.existing_path == "concepts/account-closure-procedures"
    assert "Synthesized body for account-closure-procedure" in update.additions_markdown
    # Note records the redirect so users can see it in ingest output.
    assert any("slug collision redirected" in n for n in result.notes)


def test_run_synthesis_does_not_redirect_distinct_concepts(
    project: Path, registry: Registry
) -> None:
    """Pages that share a prefix but are genuinely different concepts
    stay as creates — the 95% threshold is tight enough that e.g.
    ``card-liability-limits`` doesn't collide with ``bank-liability-limits``.
    """
    registry.register_page(
        "concepts/bank-liability-limits",
        PageEntry(
            title="Bank Liability Limits",
            type="concept",
            status="active",
            created="2026-01-01T00:00:00Z",
            modified="2026-01-01T00:00:00Z",
            summary="Limits on bank liability.",
        ),
    )
    registry.save()

    chunks = [_chunk("text 1", 0, 1)]
    chunk_ids = ["c1"]
    llm = _mock_llm_client(
        [
            _ok_response(
                creates=[
                    _create_page(
                        "card-liability-limits",
                        title="Card Liability Limits",
                    )
                ]
            ),
        ]
    )

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert len(result.pages_to_create) == 1
    assert result.pages_to_create[0].suggested_slug == "card-liability-limits"
    assert result.pages_to_update == []


def test_run_synthesis_update_proposals_accumulate(
    project: Path, registry: Registry
) -> None:
    """Multiple chunks proposing updates to the same page — all kept."""
    chunks = [_chunk("text 1", 0, 2), _chunk("text 2", 1, 2)]
    chunk_ids = ["c1", "c2"]
    llm = _mock_llm_client(
        [
            _ok_response(
                updates=[
                    {
                        "existing_path": "concepts/foo",
                        "additions_markdown": "First addition",
                    }
                ]
            ),
            _ok_response(
                updates=[
                    {
                        "existing_path": "concepts/foo",
                        "additions_markdown": "Second addition",
                    }
                ]
            ),
        ]
    )

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert len(result.pages_to_update) == 2
    additions = [u.additions_markdown for u in result.pages_to_update]
    # Order is not guaranteed: ThreadPoolExecutor lets chunks race for the
    # mock's FIFO response queue. The contract under test is "both updates
    # are kept", not which chunk consumed which response.
    assert sorted(additions) == ["First addition", "Second addition"]


# ----------------------------------------------------------------------
# run_synthesis — failures
# ----------------------------------------------------------------------


def test_run_synthesis_isolates_provider_error_per_chunk(
    project: Path, registry: Registry
) -> None:
    """A failing chunk doesn't abort the whole run."""
    chunks = [_chunk("good 1", 0, 3), _chunk("bad", 1, 3), _chunk("good 2", 2, 3)]
    chunk_ids = ["c1", "c2", "c3"]
    fake_exc = LLMProviderError(
        model="test", call_type="synthesize", original=RuntimeError("rate limit")
    )
    llm = _mock_llm_client(
        [
            _ok_response(creates=[_create_page("alpha")]),
            fake_exc,
            _ok_response(creates=[_create_page("gamma")]),
        ]
    )

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert result.chunks_processed == 2
    assert result.chunks_failed == 1
    assert len(result.pages_to_create) == 2  # alpha + gamma
    assert any("rate limit" in n for n in result.notes)


def test_run_synthesis_records_schema_failure(
    project: Path, registry: Registry
) -> None:
    chunks = [_chunk("text", 0, 1)]
    chunk_ids = ["c1"]
    # Missing required pages_to_create field
    bad_response = {
        "source_summary": {"title": "T", "one_line": "o", "content_markdown": "m"},
        "pages_to_update": [],
        "entities_mentioned": [],
        "concepts_mentioned": [],
    }
    llm = _mock_llm_client([bad_response])

    result = run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
    )

    assert result.chunks_processed == 0
    assert result.chunks_failed == 1
    assert any("pages_to_create" in n for n in result.notes)


# ----------------------------------------------------------------------
# run_synthesis — IngestState integration
# ----------------------------------------------------------------------


def test_run_synthesis_marks_state_done_per_chunk(
    project: Path, registry: Registry
) -> None:
    chunks = [_chunk("a", 0, 2), _chunk("b", 1, 2)]
    chunk_ids = ["c1", "c2"]
    state = IngestState.begin(
        registry_dir=project / "_registry",
        source_key="source-hash",
        source_name="doc.md",
        content_type="markdown",
        chunks=[
            ChunkState(index=0, total=2, token_estimate=10),
            ChunkState(index=1, total=2, token_estimate=10),
        ],
    )
    llm = _mock_llm_client([_ok_response(), _ok_response()])

    run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
        state=state,
    )

    reloaded = IngestState.load(project / "_registry")
    assert reloaded is not None
    assert reloaded.is_complete()
    assert reloaded.pending_indices() == []


def test_run_synthesis_marks_state_failed_on_provider_error(
    project: Path, registry: Registry
) -> None:
    chunks = [_chunk("a", 0, 1)]
    chunk_ids = ["c1"]
    state = IngestState.begin(
        registry_dir=project / "_registry",
        source_key="source-hash",
        source_name="doc.md",
        content_type="markdown",
        chunks=[ChunkState(index=0, total=1, token_estimate=10)],
    )
    llm = _mock_llm_client(
        [
            LLMProviderError(
                model="test",
                call_type="synthesize",
                original=RuntimeError("network"),
            )
        ]
    )

    run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
        state=state,
    )

    reloaded = IngestState.load(project / "_registry")
    assert reloaded is not None
    assert not reloaded.is_complete()
    assert reloaded.chunks[0].error is not None
    assert "network" in reloaded.chunks[0].error


# ----------------------------------------------------------------------
# Progress callback
# ----------------------------------------------------------------------


def test_run_synthesis_invokes_progress_callback(
    project: Path, registry: Registry
) -> None:
    chunks = [_chunk("a", 0, 2), _chunk("b", 1, 2)]
    chunk_ids = ["c1", "c2"]
    llm = _mock_llm_client([_ok_response(), _ok_response()])

    calls: list[tuple[int, int, int, float]] = []

    def cb(n: int, total: int, tokens: int, cost: float) -> None:
        calls.append((n, total, tokens, cost))

    run_synthesis(
        chunks=chunks,
        chunk_ids=chunk_ids,
        registry=registry,
        llm_client=llm,
        project_root=project,
        progress_callback=cb,
    )

    assert len(calls) == 2
    assert calls[0][0] == 1  # chunk 1
    assert calls[1][0] == 2  # chunk 2
    assert calls[0][1] == 2  # total


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_run_synthesis_rejects_length_mismatch(
    project: Path, registry: Registry
) -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        run_synthesis(
            chunks=[_chunk("a", 0, 1)],
            chunk_ids=["c1", "c2"],
            registry=registry,
            llm_client=_mock_llm_client([]),
            project_root=project,
        )
