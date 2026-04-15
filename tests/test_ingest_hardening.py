"""Tests for the ingest-boundary hardening pass.

Covers the two pre-C20 guards (file-size cap, empty-extraction check)
and the resume checkpoint lifecycle that C20's synthesis loop will
extend.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

import wikiloom.ingest.processor as processor_module
from wikiloom.ingest.errors import (
    BudgetExceededError,
    EmptyExtractionError,
    FileTooLargeError,
)
from wikiloom.ingest.processor import ingest
from wikiloom.ingest.state import STATE_FILENAME, IngestState
from wikiloom.llm import LLMCallMetrics, SynthesizeResult
from wikiloom.scaffold import init_project


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


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


def _write_toml_ingest_section(project: Path, body: str) -> None:
    """Append an ``[ingest]`` section to the project's wikiloom.toml."""
    toml_path = project / "wikiloom.toml"
    toml_path.write_text(
        toml_path.read_text() + "\n[ingest]\n" + body + "\n",
        encoding="utf-8",
    )


def _patch_empty_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock LLMClient.synthesize with an empty-but-valid response."""

    def fake_synthesize(self, system_prompt: str, user_prompt: str):
        return SynthesizeResult(
            result={
                "source_summary": {
                    "title": "T",
                    "one_line": "o",
                    "content_markdown": "m",
                },
                "pages_to_create": [],
                "pages_to_update": [],
                "entities_mentioned": [],
                "concepts_mentioned": [],
            },
            metrics=LLMCallMetrics(
                tokens_in=10, tokens_out=5, cost_usd=0.0, model="mock"
            ),
        )

    monkeypatch.setattr(
        processor_module.LLMClient, "synthesize", fake_synthesize
    )


# ----------------------------------------------------------------------
# File-size guard (item D)
# ----------------------------------------------------------------------


def test_file_size_guard_rejects_oversized_input(
    project: Path, tmp_path: Path
) -> None:
    _write_toml_ingest_section(project, "max_file_size_mb = 1")
    big = tmp_path / "big.md"
    big.write_bytes(b"# header\n" + b"x" * (2 * 1024 * 1024))  # ~2 MB

    with pytest.raises(FileTooLargeError) as exc_info:
        ingest(big, project_root=project)
    assert exc_info.value.limit_mb == 1
    assert exc_info.value.size_mb > 1


def test_file_size_guard_allows_under_limit(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_empty_llm(monkeypatch)
    _write_toml_ingest_section(project, "max_file_size_mb = 5")
    small = tmp_path / "small.md"
    small.write_text("# Small\n\nA few words only.\n")

    result = ingest(small, project_root=project)
    assert result.content.text  # did not raise


def test_file_size_guard_disabled_when_zero(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_empty_llm(monkeypatch)
    _write_toml_ingest_section(project, "max_file_size_mb = 0")
    big = tmp_path / "big.md"
    big.write_bytes(b"# header\n" + b"x" * (2 * 1024 * 1024))

    # Should not raise even though the file is large.
    result = ingest(big, project_root=project)
    assert result.content.text


# ----------------------------------------------------------------------
# Empty-extraction guard (item D)
# ----------------------------------------------------------------------


def test_empty_extraction_guard_rejects_blank_file(
    project: Path, tmp_path: Path
) -> None:
    blank = tmp_path / "blank.md"
    blank.write_text("")

    with pytest.raises(EmptyExtractionError) as exc_info:
        ingest(blank, project_root=project)
    assert exc_info.value.extracted_chars < 16


def test_empty_extraction_guard_rejects_whitespace_only(
    project: Path, tmp_path: Path
) -> None:
    spacer = tmp_path / "spacer.md"
    spacer.write_text("   \n\n\t\n")

    with pytest.raises(EmptyExtractionError):
        ingest(spacer, project_root=project)


def test_empty_extraction_guard_respects_min_chars_override(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_empty_llm(monkeypatch)
    # Lower the minimum so a very short doc is acceptable.
    _write_toml_ingest_section(project, "min_extracted_chars = 1")
    tiny = tmp_path / "tiny.md"
    tiny.write_text("hi")

    result = ingest(tiny, project_root=project)
    assert result.content.text.strip() == "hi"


# ----------------------------------------------------------------------
# Checkpoint lifecycle (item B)
# ----------------------------------------------------------------------


def test_ingest_clears_checkpoint_on_success(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_empty_llm(monkeypatch)
    sample = tmp_path / "doc.md"
    sample.write_text("# Doc\n\nA short document worth ingesting.\n")

    ingest(sample, project_root=project)

    state_path = project / "_registry" / STATE_FILENAME
    assert not state_path.exists()


def test_ingest_leaves_checkpoint_when_commit_fails(
    project: Path, tmp_path: Path, monkeypatch
) -> None:
    """If the commit stage blows up, the checkpoint file should remain.

    Simulates a mid-pipeline crash. The resume file has to survive so
    the next run can detect it.
    """
    from wikiloom.ingest import processor

    _patch_empty_llm(monkeypatch)

    sample = tmp_path / "doc.md"
    sample.write_text("# Doc\n\nA short document worth ingesting.\n")

    original = processor.GitOps

    class ExplodingGitOps:
        def __init__(self, *a, **kw):
            pass

        def commit_ingest(self, *a, **kw):
            raise RuntimeError("simulated commit failure")

    monkeypatch.setattr(processor, "GitOps", ExplodingGitOps)
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        ingest(sample, project_root=project)
    monkeypatch.setattr(processor, "GitOps", original)

    state_path = project / "_registry" / STATE_FILENAME
    assert state_path.exists()

    # The leftover state should be loadable and carry the plan.
    # Chunks can be marked done by synthesis (mocked to succeed here),
    # but the state file itself must survive the crash.
    state = IngestState.load(project / "_registry")
    assert state is not None
    assert state.source_name == "doc.md"
    assert len(state.chunks) >= 1


def test_checkpoint_from_failed_run_is_overwritten_by_next_ingest(
    project: Path, tmp_path: Path, monkeypatch
) -> None:
    """A prior crashed run's state file is replaced cleanly by the next begin."""
    from wikiloom.ingest import processor

    _patch_empty_llm(monkeypatch)

    doomed = tmp_path / "doomed.md"
    doomed.write_text("# Doomed\n\nThis ingest will fail at commit time.\n")

    original = processor.GitOps

    class ExplodingGitOps:
        def __init__(self, *a, **kw):
            pass

        def commit_ingest(self, *a, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(processor, "GitOps", ExplodingGitOps)
    with pytest.raises(RuntimeError):
        ingest(doomed, project_root=project)
    monkeypatch.setattr(processor, "GitOps", original)

    assert (project / "_registry" / STATE_FILENAME).exists()

    # A follow-up successful ingest should overwrite and then clear.
    ok = tmp_path / "ok.md"
    ok.write_text("# OK\n\nAnother short document to ingest cleanly.\n")
    ingest(ok, project_root=project)

    assert not (project / "_registry" / STATE_FILENAME).exists()


# ----------------------------------------------------------------------
# Pre-flight budget check
# ----------------------------------------------------------------------


def test_preflight_budget_check_refuses_over_budget(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ingest that would exceed the monthly budget should fail fast."""
    _patch_empty_llm(monkeypatch)

    # Rewrite the project's config with a vanishingly small budget.
    toml = project / "wikiloom.toml"
    toml.write_text(
        toml.read_text().replace(
            'monthly_budget_usd = 50.0', 'monthly_budget_usd = 0.00000001'
        ),
        encoding="utf-8",
    )

    sample = tmp_path / "doc.md"
    sample.write_text("# Doc\n\nA document that will exceed the tiny budget.\n")

    with pytest.raises(BudgetExceededError) as exc_info:
        ingest(sample, project_root=project)
    assert exc_info.value.budget_usd == 0.00000001
    assert exc_info.value.estimated_usd > 0


def test_preflight_budget_check_disabled_by_config(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting enable_budget_check = false bypasses the pre-flight."""
    _patch_empty_llm(monkeypatch)

    toml = project / "wikiloom.toml"
    toml.write_text(
        toml.read_text().replace(
            'monthly_budget_usd = 50.0', 'monthly_budget_usd = 0.00000001'
        )
        + "\n[ingest]\nenable_budget_check = false\n",
        encoding="utf-8",
    )

    sample = tmp_path / "doc.md"
    sample.write_text("# Doc\n\nNormal content.\n")

    # Should not raise even though the budget is effectively zero.
    ingest(sample, project_root=project)
