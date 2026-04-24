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


# ----------------------------------------------------------------------
# --batch-file / --batch-dir / mutual exclusivity
# ----------------------------------------------------------------------


def test_batch_file_reads_paths_and_skips_comments(
    project: Path, mock_llm: MagicMock
) -> None:
    """--batch-file reads one path per line, strips blanks and comments."""
    a = _write_sample(project, "a.md")
    b = _write_sample(project, "b.md")
    list_path = project.parent / "paths.txt"
    list_path.write_text(
        f"# a list of things to ingest\n\n{a}\n\n# section\n{b}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["ingest", "--batch-file", str(list_path), "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert "2 complete" in result.output


def test_batch_file_missing_raises(project: Path, mock_llm: MagicMock) -> None:
    """Missing --batch-file path is a hard error, not a silent empty run."""
    result = CliRunner().invoke(
        main,
        ["ingest", "--batch-file", "/tmp/does-not-exist.txt", "--project", str(project)],
    )
    assert result.exit_code != 0
    assert "--batch-file not found" in result.output


def test_batch_dir_resolves_sorted_files_and_skips_hidden(
    project: Path, mock_llm: MagicMock
) -> None:
    """--batch-dir returns a sorted list and ignores dotfiles + subdirs."""
    corpus = project.parent / "corpus"
    corpus.mkdir()
    body = "# {name}\n\nA longer document body so the extractor does not reject it as empty.\n"
    (corpus / "b.md").write_text(body.format(name="b"), encoding="utf-8")
    (corpus / "a.md").write_text(body.format(name="a"), encoding="utf-8")
    (corpus / ".hidden.md").write_text(body.format(name="x"), encoding="utf-8")
    (corpus / "sub").mkdir()

    result = CliRunner().invoke(
        main,
        ["ingest", "--batch-dir", str(corpus), "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert "2 complete" in result.output
    # Hidden file and the subdir were not ingested.
    assert ".hidden.md" not in result.output
    # Sorted: a appears before b in per-file headers.
    a_idx = result.output.find("a.md")
    b_idx = result.output.find("b.md")
    assert 0 <= a_idx < b_idx, result.output


def test_batch_dir_empty_raises(project: Path, mock_llm: MagicMock) -> None:
    """An empty --batch-dir is a hard error — no files to ingest."""
    empty = project.parent / "empty"
    empty.mkdir()
    result = CliRunner().invoke(
        main,
        ["ingest", "--batch-dir", str(empty), "--project", str(project)],
    )
    assert result.exit_code != 0
    assert "no files" in result.output


def test_mutual_exclusivity_positional_plus_batch_file_errors(
    project: Path, mock_llm: MagicMock
) -> None:
    """Passing both positional paths and --batch-file fails fast."""
    a = _write_sample(project, "a.md")
    list_path = project.parent / "paths.txt"
    list_path.write_text(f"{a}\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "ingest",
            str(a),
            "--batch-file",
            str(list_path),
            "--project",
            str(project),
        ],
    )

    assert result.exit_code != 0
    assert "Choose one input mode" in result.output


def test_mutual_exclusivity_batch_file_plus_batch_dir_errors(
    project: Path, mock_llm: MagicMock
) -> None:
    """--batch-file and --batch-dir can't be combined."""
    corpus = project.parent / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# a\n", encoding="utf-8")
    list_path = project.parent / "paths.txt"
    list_path.write_text("a.md\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "ingest",
            "--batch-file",
            str(list_path),
            "--batch-dir",
            str(corpus),
            "--project",
            str(project),
        ],
    )

    assert result.exit_code != 0
    assert "Choose one input mode" in result.output


def test_no_input_mode_errors(project: Path, mock_llm: MagicMock) -> None:
    """Calling `wikiloom ingest` with no paths and no flags errors."""
    result = CliRunner().invoke(
        main, ["ingest", "--project", str(project)]
    )
    assert result.exit_code != 0
    assert "Provide at least one source" in result.output


# ----------------------------------------------------------------------
# Soft cap + --yes
# ----------------------------------------------------------------------


def _fake_ingest_that_succeeds(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the processor-level ingest with a stub so large-batch
    tests don't actually run extraction + synthesis on every file.
    """
    from wikiloom.cli import _print_ingest_summary  # noqa: F401
    import wikiloom.cli as cli_module

    stub = MagicMock()

    class _StubResult:
        pages_created: list[str] = []
        pages_updated: list[str] = []
        notes: list[str] = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost_usd = 0.0
        chunks_total = 0
        chunks_processed = 0
        chunks_failed = 0

    def _fake(source, **kwargs):
        stub(source=source, **kwargs)
        return _StubResult()

    monkeypatch.setattr(
        "wikiloom.ingest.processor.ingest", _fake
    )
    monkeypatch.setattr(cli_module, "_require_clean_tree", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_warn_if_dirty", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_module, "_post_flight_budget_warning", lambda *a, **k: None
    )
    return stub


def _make_fake_files(project: Path, count: int) -> Path:
    """Write `count` small markdown files under project.parent and return
    a paths.txt listing them, one per line. Files exist (so pre-flight
    accepts them) but an ingest stub takes over so no real LLM runs.
    """
    list_path = project.parent / "paths.txt"
    lines = []
    for i in range(count):
        f = project.parent / f"fake-{i}.md"
        f.write_text(
            "# a\n\nA long-enough body to clear the empty-extraction guard.\n",
            encoding="utf-8",
        )
        lines.append(str(f))
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def test_small_batch_does_not_prompt(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 3-file batch stays below the soft threshold — no prompt."""
    _fake_ingest_that_succeeds(monkeypatch)
    list_path = _make_fake_files(project, 3)

    result = CliRunner().invoke(
        main,
        ["ingest", "--batch-file", str(list_path), "--project", str(project)],
    )

    assert result.exit_code == 0, result.output
    assert "rough estimate" not in result.output


def test_large_batch_prompts_and_aborts_on_no(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the threshold, answering 'n' aborts before any ingest runs."""
    stub = _fake_ingest_that_succeeds(monkeypatch)
    list_path = _make_fake_files(project, 25)

    result = CliRunner().invoke(
        main,
        ["ingest", "--batch-file", str(list_path), "--project", str(project)],
        input="n\n",
    )

    assert result.exit_code != 0
    assert "rough estimate" in result.output
    # Processor never called — we aborted at the prompt.
    assert stub.call_count == 0


def test_large_batch_proceeds_on_yes(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the threshold, answering 'y' proceeds with all files."""
    stub = _fake_ingest_that_succeeds(monkeypatch)
    list_path = _make_fake_files(project, 25)

    result = CliRunner().invoke(
        main,
        ["ingest", "--batch-file", str(list_path), "--project", str(project)],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "rough estimate" in result.output
    # All 25 files flowed through to the processor stub.
    assert stub.call_count == 25


def test_yes_flag_skips_prompt(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--yes bypasses the confirmation entirely for non-interactive runs."""
    stub = _fake_ingest_that_succeeds(monkeypatch)
    list_path = _make_fake_files(project, 25)

    result = CliRunner().invoke(
        main,
        [
            "ingest",
            "--batch-file",
            str(list_path),
            "--yes",
            "--project",
            str(project),
        ],
    )

    assert result.exit_code == 0, result.output
    # Prompt text never printed.
    assert "rough estimate" not in result.output
    assert stub.call_count == 25
