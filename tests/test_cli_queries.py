"""Tests for the `wikiloom queries` CLI command."""

from __future__ import annotations

from pathlib import Path

import git
import pytest
from click.testing import CliRunner

from wikiloom.cli import main
from wikiloom.query_history import (
    QueryHistory,
    QueryHistoryEntry,
    derive_query_id,
)
from wikiloom.scaffold import init_project


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = init_project(name="testproj", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    return project_dir


def _seed(project: Path, *questions: str) -> QueryHistory:
    history = QueryHistory.load(project / "_registry")
    for i, q in enumerate(questions):
        ts = f"2026-04-26T12:{i:02d}:00Z"
        history.append(
            QueryHistoryEntry(
                query_id=derive_query_id(q, ts),
                timestamp=ts,
                question=q,
                answer=f"Answer to {q}",
                confidence=["high", "medium", "low"][i % 3],
                sources=[{"page_path": f"concepts/p{i}", "relevance": "high"}],
                tokens_in=100,
                tokens_out=200,
                cost_usd=0.0015,
                latency_ms=850,
            ),
            max_entries=100,
        )
    history.save()
    return history


def test_queries_empty_history_shows_hint(project: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["queries", "--project", str(project)])
    assert result.exit_code == 0, result.output
    assert "No queries in history yet" in result.output


def test_queries_lists_entries_newest_first(project: Path) -> None:
    _seed(project, "first question", "second question", "third question")

    runner = CliRunner()
    result = runner.invoke(main, ["queries", "--project", str(project)])

    assert result.exit_code == 0, result.output
    assert "Query history" in result.output
    # Newest first: third → second → first
    third_pos = result.output.find("third question")
    second_pos = result.output.find("second question")
    first_pos = result.output.find("first question")
    assert 0 < third_pos < second_pos < first_pos


def test_queries_show_by_index_renders_full_answer(project: Path) -> None:
    _seed(project, "first question", "second question")

    runner = CliRunner()
    result = runner.invoke(
        main, ["queries", "--show", "1", "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    assert "second question" in result.output  # 1 = newest
    assert "Answer to second question" in result.output


def test_queries_show_by_query_id_prefix(project: Path) -> None:
    history = _seed(project, "the question")
    full_id = history.entries[0].query_id

    runner = CliRunner()
    result = runner.invoke(
        main, ["queries", "--show", full_id[:5], "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    assert "the question" in result.output


def test_queries_show_unknown_id_errors(project: Path) -> None:
    _seed(project, "the question")
    runner = CliRunner()
    result = runner.invoke(
        main, ["queries", "--show", "nonexistent", "--project", str(project)]
    )
    assert result.exit_code != 0
    assert "No entry matches" in result.output


def test_queries_save_writes_synthesis_page(project: Path) -> None:
    _seed(project, "what is x")

    runner = CliRunner()
    result = runner.invoke(
        main, ["queries", "--save", "1", "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    syntheses_dir = project / "wiki" / "syntheses"
    pages = list(syntheses_dir.glob("*.md"))
    pages = [p for p in pages if p.name != "index.md"]
    assert len(pages) == 1
    assert "Answer to what is x" in pages[0].read_text(encoding="utf-8")


def test_queries_show_and_save_are_mutually_exclusive(project: Path) -> None:
    _seed(project, "x")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["queries", "--show", "1", "--save", "1", "--project", str(project)],
    )
    assert result.exit_code != 0
    assert "not both" in result.output.lower()


def test_queries_default_caps_at_20(project: Path) -> None:
    _seed(project, *(f"q{i}" for i in range(25)))

    runner = CliRunner()
    result = runner.invoke(main, ["queries", "--project", str(project)])

    assert result.exit_code == 0, result.output
    # Newest 20 (q24..q5) shown; q4..q0 hidden behind the "more" hint.
    assert "q24" in result.output
    assert "q5" in result.output
    assert "5 more" in result.output
    assert "q4" not in result.output


def test_queries_all_shows_every_retained_entry(project: Path) -> None:
    _seed(project, *(f"q{i}" for i in range(25)))

    runner = CliRunner()
    result = runner.invoke(
        main, ["queries", "--all", "--project", str(project)]
    )

    assert result.exit_code == 0, result.output
    assert "q24" in result.output
    assert "q0" in result.output
    assert "more" not in result.output
