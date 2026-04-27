"""Tests for spaCy model detection during `wikiloom init`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wikiloom import cli as cli_mod
from wikiloom.cli import _check_and_install_spacy_model


@pytest.fixture
def model_present(monkeypatch):
    """Pretend spaCy + the model are installed."""
    fake_spacy = MagicMock()
    fake_spacy.load.return_value = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "spacy", fake_spacy)
    return fake_spacy


@pytest.fixture
def model_missing(monkeypatch):
    """Pretend spaCy is installed but the model isn't."""
    fake_spacy = MagicMock()
    fake_spacy.load.side_effect = OSError("model not found")
    monkeypatch.setitem(__import__("sys").modules, "spacy", fake_spacy)
    return fake_spacy


def test_silent_when_model_already_installed(
    model_present, capsys
) -> None:
    """No prompt, no message — already installed is the silent path."""
    _check_and_install_spacy_model(no_interactive=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_no_interactive_prints_manual_command(
    model_missing, capsys
) -> None:
    """In CI mode the manual command is shown; nothing is auto-installed."""
    with patch("subprocess.run") as mock_run:
        _check_and_install_spacy_model(no_interactive=True)
    captured = capsys.readouterr()
    assert "en_core_web_sm" in captured.out
    assert "python -m spacy download en_core_web_sm" in captured.out
    mock_run.assert_not_called()


def test_decline_prints_install_later_command(
    model_missing, capsys, monkeypatch
) -> None:
    """User says 'n' → manual command shown, no subprocess."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with patch("click.confirm", return_value=False), patch(
        "subprocess.run"
    ) as mock_run:
        _check_and_install_spacy_model(no_interactive=False)
    captured = capsys.readouterr()
    assert "Install later with" in captured.out
    assert "python -m spacy download en_core_web_sm" in captured.out
    mock_run.assert_not_called()


def test_accept_runs_subprocess_with_sys_executable(
    model_missing, capsys, monkeypatch
) -> None:
    """User says 'Y' → subprocess fires with sys.executable + spacy download."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    fake_result = MagicMock(returncode=0)
    with patch("click.confirm", return_value=True), patch(
        "subprocess.run", return_value=fake_result
    ) as mock_run:
        _check_and_install_spacy_model(no_interactive=False)

    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    # sys.executable + ['-m', 'spacy', 'download', 'en_core_web_sm']
    assert cmd[1:] == ["-m", "spacy", "download", "en_core_web_sm"]


def test_accept_with_failed_install_does_not_crash(
    model_missing, capsys, monkeypatch
) -> None:
    """Subprocess returncode != 0 prints a fallback message but doesn't raise."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    fake_result = MagicMock(returncode=1)
    with patch("click.confirm", return_value=True), patch(
        "subprocess.run", return_value=fake_result
    ):
        # Must not raise — init should complete even if the download fails.
        _check_and_install_spacy_model(no_interactive=False)

    captured = capsys.readouterr()
    assert "Download failed" in captured.out
    assert "Install manually" in captured.out


def test_subprocess_filenotfound_does_not_crash(
    model_missing, capsys, monkeypatch
) -> None:
    """If spawning the subprocess itself fails, fall back to manual command."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with patch("click.confirm", return_value=True), patch(
        "subprocess.run", side_effect=FileNotFoundError("no python")
    ):
        _check_and_install_spacy_model(no_interactive=False)

    captured = capsys.readouterr()
    assert "Couldn" in captured.out  # "Couldn't run spaCy"
    assert "Install manually" in captured.out


def test_non_tty_stdin_skips_prompt_like_no_interactive(
    model_missing, capsys, monkeypatch
) -> None:
    """A piped (non-TTY) stdin behaves like --no-interactive: no prompt."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with patch("subprocess.run") as mock_run:
        _check_and_install_spacy_model(no_interactive=False)
    captured = capsys.readouterr()
    assert "Install with" in captured.out
    mock_run.assert_not_called()
