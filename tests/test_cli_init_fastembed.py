"""Tests for fastembed model prefetch during `wikiloom init`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wikiloom.cli import _maybe_prefetch_fastembed_model


@pytest.fixture
def model_cached(monkeypatch, tmp_path: Path) -> Path:
    """Pretend the model is already on disk under the durable cache."""
    cache = tmp_path / "fastembed"
    snapshot = cache / "models--qdrant--bge-small-en-v1.5-onnx-q" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.onnx").write_bytes(b"\x00")
    monkeypatch.setattr(
        "wikiloom.embeddings.fastembed_cache_dir", lambda: cache
    )
    monkeypatch.setattr(
        "wikiloom.cli.fastembed_cache_dir",
        lambda: cache,
        raising=False,
    )
    return cache


@pytest.fixture
def model_missing(monkeypatch, tmp_path: Path) -> Path:
    """Pretend the cache is empty (no snapshots, no .onnx files)."""
    cache = tmp_path / "fastembed"
    monkeypatch.setattr(
        "wikiloom.embeddings.fastembed_cache_dir", lambda: cache
    )
    return cache


def test_silent_when_model_already_cached(
    model_cached, capsys, monkeypatch
) -> None:
    """A populated cache means no prompt and no message."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    _maybe_prefetch_fastembed_model(no_interactive=False)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_no_interactive_skips_download(
    model_missing, capsys
) -> None:
    """In CI mode the model is not downloaded; user is told it'll happen later."""
    with patch("wikiloom.embeddings.FastEmbedBackend") as mock_backend:
        _maybe_prefetch_fastembed_model(no_interactive=True)
    captured = capsys.readouterr()
    assert "fastembed" in captured.out
    assert "first ingest" in captured.out
    mock_backend.assert_not_called()


def test_decline_prints_config_hint(
    model_missing, capsys, monkeypatch
) -> None:
    """User says 'n' → hint points at [embeddings] in wikiloom.toml; no download."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with patch("click.confirm", return_value=False), patch(
        "wikiloom.embeddings.FastEmbedBackend"
    ) as mock_backend:
        _maybe_prefetch_fastembed_model(no_interactive=False)
    captured = capsys.readouterr()
    assert "Skipped" in captured.out
    assert "[embeddings]" in captured.out
    assert "wikiloom.toml" in captured.out
    mock_backend.assert_not_called()


def test_accept_constructs_backend_to_trigger_download(
    model_missing, capsys, monkeypatch
) -> None:
    """User says 'Y' → FastEmbedBackend() is constructed (which downloads)."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with patch("click.confirm", return_value=True), patch(
        "wikiloom.embeddings.FastEmbedBackend"
    ) as mock_backend:
        _maybe_prefetch_fastembed_model(no_interactive=False)
    mock_backend.assert_called_once_with()
    captured = capsys.readouterr()
    assert "Downloading" in captured.out


def test_accept_with_failed_download_does_not_crash(
    model_missing, capsys, monkeypatch
) -> None:
    """Download error prints a fallback message but doesn't raise."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with patch("click.confirm", return_value=True), patch(
        "wikiloom.embeddings.FastEmbedBackend",
        side_effect=RuntimeError("network down"),
    ):
        _maybe_prefetch_fastembed_model(no_interactive=False)
    captured = capsys.readouterr()
    assert "Download failed" in captured.out
    assert "first ingest" in captured.out


def test_non_tty_stdin_skips_prompt_like_no_interactive(
    model_missing, capsys, monkeypatch
) -> None:
    """A piped (non-TTY) stdin behaves like --no-interactive: no prompt."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with patch("wikiloom.embeddings.FastEmbedBackend") as mock_backend:
        _maybe_prefetch_fastembed_model(no_interactive=False)
    captured = capsys.readouterr()
    assert "first ingest" in captured.out
    mock_backend.assert_not_called()
