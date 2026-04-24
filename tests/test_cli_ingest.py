"""Tests for the CLI-level multi-file ingest flow.

Verifies the loop, per-file failure isolation, and grand summary
classification. Uses the same LLM mock pattern as
``test_ingest_pipeline.py`` so no real provider is hit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import git
import pytest
from click.testing import CliRunner

import wikiloom.ingest.processor as processor_module
from wikiloom.cli import main
from wikiloom.llm import LLMCallMetrics, SynthesizeResult
from wikiloom.scaffold import init_project


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = init_project(name="testproj", path=tmp_path, domain="test")
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


def _empty_llm_response() -> SynthesizeResult:
    return SynthesizeResult(
        result={
            "source_summary": {
                "title": "Sample",
                "one_line": "A sample document.",
                "content_markdown": "## About\n\nNothing.",
            },
            "pages_to_create": [],
            "pages_to_update": [],
            "entities_mentioned": [],
            "concepts_mentioned": [],
        },
        metrics=LLMCallMetrics(
            tokens_in=20, tokens_out=15, cost_usd=0.0001, model="mock-model"
        ),
    )


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()

    def _fake_synthesize(self: Any, system_prompt: str, user_prompt: str) -> Any:
        mock(system_prompt=system_prompt, user_prompt=user_prompt)
        return _empty_llm_response()

    monkeypatch.setattr(
        processor_module.LLMClient, "synthesize", _fake_synthesize
    )
    return mock


def _write_sample(project: Path, name: str, text: str | None = None) -> Path:
    path = project.parent / name
    path.write_text(
        text or f"# {name}\n\nA short document for testing.\n",
        encoding="utf-8",
    )
    return path


def test_single_file_has_no_grand_summary(
    project: Path, mock_llm: MagicMock
) -> None:
    """With one source, output is unchanged from pre-batch behavior."""
    sample = _write_sample(project, "one.md")

    result = CliRunner().invoke(
        main, ["ingest", str(sample), "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    # The grand-summary bucket labels only appear in multi-file mode.
    assert "complete" not in result.output
    assert "Partial:" not in result.output
    assert "Failed:" not in result.output


def test_multi_file_grand_summary_all_complete(
    project: Path, mock_llm: MagicMock
) -> None:
    """Two successful files → 2 complete in the grand summary."""
    a = _write_sample(project, "a.md")
    b = _write_sample(project, "b.md")

    result = CliRunner().invoke(
        main,
        ["ingest", str(a), str(b), "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert "[1/2]" in result.output
    assert "[2/2]" in result.output
    assert "2 complete" in result.output
    assert "0 partial" in result.output
    assert "0 failed" in result.output


def test_multi_file_missing_path_is_bucketed_as_failed(
    project: Path, mock_llm: MagicMock
) -> None:
    """A typoed path in a batch fails that file but keeps processing."""
    good = _write_sample(project, "good.md")
    missing = project.parent / "does-not-exist.md"

    result = CliRunner().invoke(
        main,
        ["ingest", str(good), str(missing), "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert "1 complete" in result.output
    assert "1 failed" in result.output
    assert "Failed:" in result.output
    assert "does-not-exist.md" in result.output


def test_single_missing_path_still_raises(
    project: Path, mock_llm: MagicMock
) -> None:
    """For a single-source invocation, a bad path is a hard error."""
    missing = project.parent / "nope.md"

    result = CliRunner().invoke(
        main,
        ["ingest", str(missing), "--project", str(project)],
    )

    assert result.exit_code != 0
    assert "No such file" in result.output
