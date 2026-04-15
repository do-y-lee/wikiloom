"""End-to-end smoke tests for ``wikiloom.ingest.processor.ingest``.

These tests verify the write-side pipeline composes correctly:
extraction → raw copy → synthesis (mocked) → backlinks rebuild →
manifest sync → index regeneration → single atomic git commit. The
LLM client is monkeypatched in every test so no real provider is
hit; a dedicated test verifies the synthesis → page-writer path
with a non-empty mocked response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import git
import pytest

import wikiloom.ingest.processor as processor_module
from wikiloom.frontmatter import read_page
from wikiloom.ingest.processor import ingest
from wikiloom.llm import LLMCallMetrics, SynthesizeResult
from wikiloom.scaffold import init_project


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Real init_project output with a git identity configured."""
    project_dir = init_project(name="testproj", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    # Baseline commit so HEAD is valid before any ingest.
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


@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text(
        "# Sample\n\nA short document mentioning nothing in particular.\n",
        encoding="utf-8",
    )
    return path


def _empty_llm_response() -> SynthesizeResult:
    """Valid ingest_response.json shape with no pages or updates.

    Lets existing tests run through the synthesis stage without
    producing any new pages, so their assertions about commits and
    file counts stay accurate.
    """
    return SynthesizeResult(
        result={
            "source_summary": {
                "title": "Sample",
                "one_line": "A short sample document.",
                "content_markdown": "## About\n\nNothing in particular.",
            },
            "pages_to_create": [],
            "pages_to_update": [],
            "entities_mentioned": [],
            "concepts_mentioned": [],
        },
        metrics=LLMCallMetrics(
            tokens_in=20,
            tokens_out=15,
            cost_usd=0.0001,
            model="mock-model",
        ),
    )


def _install_mock_llm(
    monkeypatch: pytest.MonkeyPatch,
    response_builder=_empty_llm_response,
) -> MagicMock:
    """Patch LLMClient.synthesize so ingest tests never hit a real API.

    The processor imports ``LLMClient`` at module load time, so we
    monkeypatch the reference on the processor module rather than on
    the llm module. Returns the mock so tests can assert call counts.
    """
    mock = MagicMock()

    def _fake_synthesize(self: Any, system_prompt: str, user_prompt: str) -> Any:
        mock(system_prompt=system_prompt, user_prompt=user_prompt)
        return response_builder()

    monkeypatch.setattr(
        processor_module.LLMClient, "synthesize", _fake_synthesize
    )
    return mock


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Autouse-style mock for tests that don't care about synthesis output."""
    return _install_mock_llm(monkeypatch)


# ----------------------------------------------------------------------
# Smoke tests
# ----------------------------------------------------------------------


def test_ingest_writes_backlinks_file(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    ingest(sample_markdown, project_root=project)
    backlinks = project / "_registry" / "backlinks.json"
    assert backlinks.exists()
    data = json.loads(backlinks.read_text())
    assert data["version"] == 1
    assert "links" in data


def test_ingest_copies_source_to_raw(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    result = ingest(sample_markdown, project_root=project)
    assert result.raw_path is not None
    assert result.raw_path.exists()
    # Ends up under raw/articles/ for markdown (per RAW_DEST_BY_CONTENT_TYPE)
    assert "articles" in result.raw_path.parts


def test_ingest_regenerates_index_files(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    root_index = project / "wiki" / "index.md"
    concepts_index = project / "wiki" / "concepts" / "index.md"

    ingest(sample_markdown, project_root=project)

    # Root index was rewritten in the IndexUpdater format
    after_root = root_index.read_text()
    assert "# Wiki Index" in after_root
    assert "## Concepts" in after_root
    # Sub-indexes now follow the table format (or empty placeholder)
    assert "Concepts Index" in concepts_index.read_text()


def test_ingest_produces_single_ingest_commit(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    repo = git.Repo(project)
    commits_before = len(list(repo.iter_commits()))

    ingest(sample_markdown, project_root=project)

    commits_after = len(list(repo.iter_commits()))
    assert commits_after == commits_before + 1

    head = repo.head.commit
    assert head.message.startswith("ingest: sample.md")


def test_ingest_commit_includes_backlinks_and_indexes(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    """The ingest commit should be atomic: raw copy + backlinks + indexes
    all land in one commit so history reflects a coherent state."""
    repo = git.Repo(project)
    ingest(sample_markdown, project_root=project)

    head = repo.head.commit
    # Diff the ingest commit against its parent — every touched file in
    # this run should appear in the delta.
    changed = {item.a_path or item.b_path for item in head.diff(head.parents[0])}
    assert any("raw/articles/sample.md" in p for p in changed)
    assert any("_registry/backlinks.json" in p for p in changed)
    assert any("wiki/index.md" == p for p in changed)


def test_ingest_is_reentrant_on_identical_source(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    """Re-ingesting the same file should not pollute git history with
    empty commits. Second run's commit hash should equal HEAD from
    before (no-op) since the pipeline produces identical derived state."""
    repo = git.Repo(project)
    ingest(sample_markdown, project_root=project)
    head_after_first = repo.head.commit.hexsha

    ingest(sample_markdown, project_root=project)
    head_after_second = repo.head.commit.hexsha

    assert head_after_first == head_after_second


# ----------------------------------------------------------------------
# Session A: synthesis → page-writer end-to-end (mocked LLM)
# ----------------------------------------------------------------------


def _populated_llm_response() -> SynthesizeResult:
    """Mocked LLM response with real page proposals for end-to-end checks."""
    return SynthesizeResult(
        result={
            "source_summary": {
                "title": "Sample Doc",
                "one_line": "A test document about X and Y.",
                "content_markdown": "## About\n\nThis doc covers X and Y.",
            },
            "pages_to_create": [
                {
                    "type": "concept",
                    "suggested_slug": "topic-x",
                    "title": "Topic X",
                    "content_markdown": "# Topic X\n\nTopic X is a thing.",
                    "confidence": "high",
                },
                {
                    "type": "entity",
                    "suggested_slug": "org-y",
                    "title": "Org Y",
                    "content_markdown": "# Org Y\n\nOrg Y is an organization.",
                    "confidence": "medium",
                },
            ],
            "pages_to_update": [],
            "entities_mentioned": ["Org Y"],
            "concepts_mentioned": ["Topic X"],
        },
        metrics=LLMCallMetrics(
            tokens_in=500,
            tokens_out=300,
            cost_usd=0.0042,
            model="mock-model",
        ),
    )


def test_ingest_with_synthesis_writes_pages_end_to_end(
    project: Path,
    sample_markdown: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pipeline: extract → chunk → synthesize → page writer → commit.

    The single most important end-to-end assertion in the Session A
    scope. Proves that a real ingest of a file with a mocked LLM
    response actually produces pages on disk with chunk_ids in the
    frontmatter, all under one atomic git commit.
    """
    _install_mock_llm(monkeypatch, response_builder=_populated_llm_response)

    result = ingest(sample_markdown, project_root=project)

    # Pages were written
    topic_x = project / "wiki" / "concepts" / "topic-x.md"
    org_y = project / "wiki" / "entities" / "org-y.md"
    source_page = project / "wiki" / "sources" / "sample.md"
    assert topic_x.exists()
    assert org_y.exists()
    assert source_page.exists()

    # Each synthesized page carries the chunk_id that produced it
    fm, _ = read_page(topic_x)
    assert fm is not None
    assert len(fm.chunk_ids) == 1

    # Metrics populated on the IngestResult
    assert result.total_tokens_in == 500
    assert result.total_tokens_out == 300
    assert result.total_cost_usd == pytest.approx(0.0042)
    assert "concepts/topic-x" in result.pages_created
    assert "entities/org-y" in result.pages_created

    # Single atomic commit includes every synthesized page
    repo = git.Repo(project)
    head = repo.head.commit
    changed = {item.a_path or item.b_path for item in head.diff(head.parents[0])}
    assert "wiki/concepts/topic-x.md" in changed
    assert "wiki/entities/org-y.md" in changed
    assert "wiki/sources/sample.md" in changed


def test_ingest_populates_event_log_with_real_tokens(
    project: Path,
    sample_markdown: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WikiEvent for the ingest should carry real token + cost numbers."""
    _install_mock_llm(monkeypatch, response_builder=_populated_llm_response)

    ingest(sample_markdown, project_root=project)

    log_content = (project / "wiki" / "log.md").read_text()
    # Total tokens = 500 in + 300 out = 800. Event log uses the
    # "**Tokens used**: <n>" format emitted by events.to_log_entry.
    assert "**Tokens used**: 800" in log_content
    # Cost rounds to $0.00 at current pricing — the important thing
    # is that a cost line exists, not its exact value.
    assert "**Cost**" in log_content
