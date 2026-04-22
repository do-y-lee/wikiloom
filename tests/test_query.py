"""Tests for the query pipeline (wikiloom/query.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import git
import pytest

from wikiloom.cache import SQLiteCache
from wikiloom.config import Config, LLMConfig
from wikiloom.frontmatter import Frontmatter, read_page, write_page
from wikiloom.llm import LLMCallMetrics, LLMClient, SynthesizeResult
from wikiloom.query import (
    QueryAnswer,
    load_query_prompt,
    retrieve_context,
    run_query,
    validate_query_response,
)
from wikiloom.registry import PageEntry, Registry
from wikiloom.scaffold import init_project


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = init_project(name="query-test", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    repo.index.add(
        [
            "wikiloom.toml",
            ".gitignore",
            "wiki/index.md",
            "_registry/manifest.json",
            "_registry/backlinks.json",
        ]
    )
    repo.index.commit("initial scaffold")
    return project_dir


def _add_page(
    project: Path, page_id: str, title: str, body: str, summary: str = ""
) -> None:
    fm = Frontmatter(
        title=title,
        type=page_id.split("/")[0].rstrip("s"),
        status="active",
        created="2026-01-01T00:00:00Z",
        modified="2026-01-01T00:00:00Z",
        summary=summary or f"Summary of {title}.",
    )
    write_page(project / "wiki" / f"{page_id}.md", fm, body)
    registry = Registry(project / "_registry")
    entry = PageEntry(
        title=title,
        type=fm.type,
        summary=summary or f"Summary of {title}.",
        created=fm.created,
        modified=fm.modified,
    )
    registry.register_page(page_id, entry)
    registry.save()


def _rebuild_cache(project: Path) -> SQLiteCache:
    cache = SQLiteCache(project / "_registry" / "wiki.db")
    cache.full_rebuild(project)
    return cache


def _ok_query_response(**overrides: Any) -> dict[str, Any]:
    base = {
        "answer": "Flash Attention reduces GPU memory by tiling attention blocks.",
        "sources_consulted": [
            {"page_path": "concepts/flash-attention", "relevance": "primary"},
        ],
        "confidence": "high",
        "suggest_synthesis": False,
        "suggested_followups": ["How does it compare to standard attention?"],
    }
    base.update(overrides)
    return base


def _mock_llm(response: dict[str, Any] | Exception) -> LLMClient:
    client = MagicMock(spec=LLMClient)
    if isinstance(response, Exception):
        client.synthesize.side_effect = response
    else:
        client.synthesize.return_value = SynthesizeResult(
            result=response,
            metrics=LLMCallMetrics(
                tokens_in=200, tokens_out=100, cost_usd=0.0015, model="mock"
            ),
        )
    return client


# ----------------------------------------------------------------------
# load_query_prompt
# ----------------------------------------------------------------------


def test_load_query_prompt_returns_template(project: Path) -> None:
    prompt = load_query_prompt(project)
    assert "Query Prompt" in prompt or "knowledge base" in prompt.lower()


# ----------------------------------------------------------------------
# retrieve_context
# ----------------------------------------------------------------------


def test_retrieve_context_finds_matching_pages(project: Path) -> None:
    _add_page(
        project,
        "concepts/flash-attention",
        "Flash Attention",
        "An I/O-aware attention algorithm.",
        summary="I/O-aware attention.",
    )
    _add_page(
        project,
        "concepts/bert",
        "BERT",
        "Bidirectional encoder representations.",
        summary="Language model.",
    )
    cache = _rebuild_cache(project)

    contexts = retrieve_context("attention", cache, project / "wiki")
    page_ids = [c.page_id for c in contexts]
    assert "concepts/flash-attention" in page_ids


def test_retrieve_context_returns_empty_on_no_match(project: Path) -> None:
    cache = _rebuild_cache(project)
    contexts = retrieve_context("quantum computing", cache, project / "wiki")
    assert contexts == []


def test_retrieve_context_truncates_long_bodies(project: Path) -> None:
    _add_page(
        project,
        "concepts/verbose",
        "Verbose",
        "word " * 5000,
        summary="A very verbose page.",
    )
    cache = _rebuild_cache(project)

    contexts = retrieve_context("verbose", cache, project / "wiki", max_body_chars=100)
    assert len(contexts) == 1
    assert len(contexts[0].body) < 200
    assert "[... truncated]" in contexts[0].body


# ----------------------------------------------------------------------
# validate_query_response
# ----------------------------------------------------------------------


def test_validate_accepts_valid_response() -> None:
    assert validate_query_response(_ok_query_response()) == []


def test_validate_rejects_missing_answer() -> None:
    resp = _ok_query_response()
    del resp["answer"]
    assert any("answer" in e for e in validate_query_response(resp))


def test_validate_rejects_bad_relevance() -> None:
    resp = _ok_query_response(
        sources_consulted=[{"page_path": "x", "relevance": "critical"}]
    )
    assert any("relevance" in e for e in validate_query_response(resp))


def test_validate_rejects_bad_confidence() -> None:
    resp = _ok_query_response(confidence="absolute")
    assert any("confidence" in e for e in validate_query_response(resp))


# ----------------------------------------------------------------------
# run_query
# ----------------------------------------------------------------------


def test_run_query_returns_structured_answer(project: Path) -> None:
    _add_page(
        project,
        "concepts/flash-attention",
        "Flash Attention",
        "An I/O-aware attention algorithm.",
        summary="I/O-aware attention.",
    )
    _rebuild_cache(project)
    llm = _mock_llm(_ok_query_response())

    answer = run_query("What is flash attention?", project, llm)

    assert isinstance(answer, QueryAnswer)
    assert "Flash Attention" in answer.answer
    assert answer.confidence == "high"
    assert len(answer.sources_consulted) == 1
    assert answer.sources_consulted[0].page_path == "concepts/flash-attention"
    assert answer.metrics.tokens_in == 200


def test_run_query_works_with_empty_wiki(project: Path) -> None:
    _rebuild_cache(project)
    resp = _ok_query_response(
        answer="I don't have enough information to answer this.",
        sources_consulted=[],
        confidence="low",
    )
    llm = _mock_llm(resp)

    answer = run_query("What is quantum gravity?", project, llm)
    assert answer.confidence == "low"
    assert answer.sources_consulted == []


def test_run_query_raises_on_invalid_llm_response(project: Path) -> None:
    _rebuild_cache(project)
    llm = _mock_llm({"bad": "response"})

    with pytest.raises(ValueError, match="validation failed"):
        run_query("anything", project, llm)


def test_run_query_includes_followups(project: Path) -> None:
    _rebuild_cache(project)
    resp = _ok_query_response(
        suggested_followups=["Follow-up A", "Follow-up B"]
    )
    llm = _mock_llm(resp)

    answer = run_query("question", project, llm)
    assert answer.suggested_followups == ["Follow-up A", "Follow-up B"]


# ----------------------------------------------------------------------
# --save (synthesis page filing)
# ----------------------------------------------------------------------


def test_save_flag_writes_synthesis_page(project: Path) -> None:
    """Verifies the save path via the query module + manual write,
    since the CLI command isn't easily testable without Click's runner."""
    from wikiloom.utils import now_iso, slugify

    _add_page(
        project,
        "concepts/flash-attention",
        "Flash Attention",
        "An I/O-aware attention algorithm.",
        summary="I/O-aware attention.",
    )
    _rebuild_cache(project)
    llm = _mock_llm(_ok_query_response(suggest_synthesis=True))

    answer = run_query("What is flash attention?", project, llm)
    assert answer.suggest_synthesis is True

    # Simulate what the CLI --save flag does
    question = "What is flash attention?"
    slug = slugify(question)[:60]
    page_path = project / "wiki" / "syntheses" / f"{slug}.md"

    fm = Frontmatter(
        title=question,
        type="synthesis",
        status="active",
        created=now_iso(),
        modified=now_iso(),
        summary=answer.answer[:160],
        confidence=answer.confidence,
    )
    write_page(page_path, fm, answer.answer)

    assert page_path.exists()
    saved_fm, saved_body = read_page(page_path)
    assert saved_fm is not None
    assert saved_fm.type == "synthesis"
    assert "Flash Attention" in saved_body


def test_save_last_regenerates_indexes(project: Path) -> None:
    """`wikiloom query --save-last` must regenerate wiki/syntheses/index.md
    and the root wiki/index.md. Earlier versions only wrote the page
    file and updated the manifest, leaving indexes out of sync until the
    next ingest/relink."""
    from wikiloom.cli import _save_query_as_page

    synthesis_index = project / "wiki" / "syntheses" / "index.md"
    root_index = project / "wiki" / "index.md"

    # Capture the pre-save content so we can prove the save call modified it.
    before_synthesis_index = (
        synthesis_index.read_text() if synthesis_index.exists() else ""
    )

    _save_query_as_page(
        {
            "question": "What is flash attention?",
            "answer": "Flash attention is an I/O-aware attention algorithm.",
            "confidence": "high",
            "sources_consulted": [],
        },
        project,
    )

    # The synthesis page file exists…
    page_path = project / "wiki" / "syntheses" / "what-is-flash-attention.md"
    assert page_path.exists()

    # …and both indexes now reference it.
    assert synthesis_index.exists()
    synthesis_index_text = synthesis_index.read_text()
    assert "what-is-flash-attention" in synthesis_index_text
    assert synthesis_index_text != before_synthesis_index

    assert root_index.exists()
    assert "syntheses" in root_index.read_text()
