"""Tests for wikiloom.config — TOML parsing edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikiloom.config import Config, ConfigError


def test_load_raises_file_not_found_when_missing(tmp_path: Path) -> None:
    """Missing wikiloom.toml is FileNotFoundError, distinct from ConfigError."""
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path)


def test_load_raises_config_error_on_malformed_toml(tmp_path: Path) -> None:
    """Malformed TOML surfaces as ConfigError with a helpful message.

    The original tomllib.TOMLDecodeError is chained as __cause__ so
    debugging tools can dig in, but the user-facing message points at
    the file path and the parse error.
    """
    bad = tmp_path / "wikiloom.toml"
    bad.write_text("[llm\nmodel = oops\n", encoding="utf-8")  # missing ]

    with pytest.raises(ConfigError) as exc_info:
        Config.load(tmp_path)

    msg = str(exc_info.value)
    assert "wikiloom.toml" in msg
    assert exc_info.value.__cause__ is not None  # original tomllib error chained


def test_load_succeeds_with_valid_minimal_toml(tmp_path: Path) -> None:
    """Sanity check: a minimal valid file loads with defaults."""
    (tmp_path / "wikiloom.toml").write_text(
        '[project]\nname = "test"\n', encoding="utf-8"
    )
    cfg = Config.load(tmp_path)
    assert cfg.project.name == "test"
    # Sections not present in the file fall back to defaults.
    assert cfg.llm.monthly_budget_usd == 50.0
    assert cfg.dormant.default_window_days == 90
