"""Tests for the ingest-boundary hardening pass.

Covers the two pre-C20 guards (file-size cap, empty-extraction check)
and the resume checkpoint lifecycle that C20's synthesis loop will
extend.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from wikiloom.ingest.errors import EmptyExtractionError, FileTooLargeError
from wikiloom.ingest.processor import ingest
from wikiloom.ingest.state import STATE_FILENAME, IngestState
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
    project: Path, tmp_path: Path
) -> None:
    _write_toml_ingest_section(project, "max_file_size_mb = 5")
    small = tmp_path / "small.md"
    small.write_text("# Small\n\nA few words only.\n")

    result = ingest(small, project_root=project)
    assert result.content.text  # did not raise


def test_file_size_guard_disabled_when_zero(
    project: Path, tmp_path: Path
) -> None:
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
    project: Path, tmp_path: Path
) -> None:
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
    project: Path, tmp_path: Path
) -> None:
    sample = tmp_path / "doc.md"
    sample.write_text("# Doc\n\nA short document worth ingesting.\n")

    ingest(sample, project_root=project)

    state_path = project / "_registry" / STATE_FILENAME
    assert not state_path.exists()


def test_ingest_leaves_checkpoint_when_commit_fails(
    project: Path, tmp_path: Path, monkeypatch
) -> None:
    """If the commit stage blows up, the checkpoint file should remain.

    Simulates a mid-pipeline crash (what C20 synthesis failures will
    look like). The resume file has to survive so the next run can
    detect it.
    """
    from wikiloom.ingest import processor

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
    state = IngestState.load(project / "_registry")
    assert state is not None
    assert state.source_name == "doc.md"
    assert len(state.chunks) >= 1
    assert state.pending_indices()  # nothing marked done yet


def test_checkpoint_from_failed_run_is_overwritten_by_next_ingest(
    project: Path, tmp_path: Path, monkeypatch
) -> None:
    """A prior crashed run's state file is replaced cleanly by the next begin."""
    from wikiloom.ingest import processor

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
