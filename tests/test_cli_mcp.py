"""Tests for the ``wikiloom mcp`` CLI subcommand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from wikiloom.cli import main


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A directory that exists — enough for Click's ``exists=True`` check."""
    p = tmp_path / "proj"
    p.mkdir()
    return p


def test_mcp_help_lists_print_config_and_project_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "--print-config" in result.output
    assert "--project" in result.output
    assert "stdio" in result.output  # docstring teaches transport


def test_mcp_print_config_emits_valid_json(project_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["mcp", "--project", str(project_dir), "--print-config"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)  # must round-trip cleanly
    assert "mcpServers" in data
    assert "wikiloom" in data["mcpServers"]


def test_mcp_print_config_uses_running_python_and_module_invocation(
    project_dir: Path,
) -> None:
    # The config must reference the running interpreter so Claude Desktop
    # spawns the server in the same env where wikiloom was installed.
    runner = CliRunner()
    result = runner.invoke(
        main, ["mcp", "--project", str(project_dir), "--print-config"]
    )
    data = json.loads(result.output)
    entry = data["mcpServers"]["wikiloom"]
    assert entry["command"] == sys.executable
    assert entry["args"][:3] == ["-m", "wikiloom", "mcp"]
    assert "--project" in entry["args"]


def test_mcp_print_config_resolves_project_to_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Relative ``--project .`` from a tmp cwd should print an absolute
    # path — that's the whole point of the helper (Claude Desktop's
    # working dir at launch is unpredictable).
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    runner = CliRunner()
    result = runner.invoke(main, ["mcp", "--project", ".", "--print-config"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    project_arg_idx = data["mcpServers"]["wikiloom"]["args"].index("--project") + 1
    project_arg = data["mcpServers"]["wikiloom"]["args"][project_arg_idx]
    assert Path(project_arg).is_absolute()
    assert Path(project_arg).resolve() == project.resolve()


def test_mcp_rejects_nonexistent_project_path(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["mcp", "--project", str(tmp_path / "does-not-exist"), "--print-config"]
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower() or "invalid value" in result.output.lower()
