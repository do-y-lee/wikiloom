"""Tests for wikiloom.source_catalog and ingest-level dedup."""

from __future__ import annotations

import json
from pathlib import Path

import git
import pytest

from wikiloom.ingest import router
from wikiloom.ingest.extractors.web_ext import WebExtractor
from wikiloom.ingest.processor import ingest
from wikiloom.scaffold import init_project
from wikiloom.source_catalog import SourceCatalog, SourceEntry, hash_file


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def test_hash_file_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello")
    assert hash_file(p) == hash_file(p)


def test_hash_file_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert hash_file(a) != hash_file(b)


# ----------------------------------------------------------------------
# SourceCatalog
# ----------------------------------------------------------------------


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    d = tmp_path / "_registry"
    d.mkdir()
    return d


def test_catalog_empty_on_first_load(registry_dir: Path) -> None:
    cat = SourceCatalog(registry_dir)
    assert cat.has("abc") is False
    assert cat.get("abc") is None


def test_catalog_record_and_roundtrip(registry_dir: Path) -> None:
    cat = SourceCatalog(registry_dir)
    cat.record(
        SourceEntry(
            content_hash="deadbeef",
            name="paper.pdf",
            content_type="pdf",
            size_bytes=1234,
            raw_path="raw/papers/paper.pdf",
            first_ingested_at="2026-04-13T00:00:00Z",
            last_ingested_at="2026-04-13T00:00:00Z",
        )
    )
    cat.save()

    reloaded = SourceCatalog(registry_dir)
    assert reloaded.has("deadbeef")
    entry = reloaded.get("deadbeef")
    assert entry is not None
    assert entry.name == "paper.pdf"
    assert entry.content_type == "pdf"


def test_catalog_touch_bumps_counter(registry_dir: Path) -> None:
    cat = SourceCatalog(registry_dir)
    cat.record(
        SourceEntry(
            content_hash="h",
            name="a",
            content_type="markdown",
            size_bytes=0,
            raw_path=None,
            first_ingested_at="t",
            last_ingested_at="t",
        )
    )
    cat.touch("h")
    assert cat.get("h").ingest_count == 2


def test_catalog_touch_unknown_returns_none(registry_dir: Path) -> None:
    cat = SourceCatalog(registry_dir)
    assert cat.touch("nope") is None


def test_catalog_save_is_deterministic(registry_dir: Path) -> None:
    cat = SourceCatalog(registry_dir)
    for h in ("b", "a", "c"):
        cat.record(
            SourceEntry(
                content_hash=h,
                name=h,
                content_type="markdown",
                size_bytes=0,
                raw_path=None,
                first_ingested_at="t",
                last_ingested_at="t",
            )
        )
    cat.save()
    first = (registry_dir / "sources.json").read_text()

    cat2 = SourceCatalog(registry_dir)
    cat2.save()
    second = (registry_dir / "sources.json").read_text()

    def strip_ts(text: str) -> str:
        return "\n".join(line for line in text.splitlines() if "updated_at" not in line)

    assert strip_ts(first) == strip_ts(second)
    # Sorted keys — "a" must come before "b" in the serialized output
    data = json.loads(first)
    assert list(data["sources"]) == ["a", "b", "c"]


# ----------------------------------------------------------------------
# Ingest-level dedup (integration)
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


@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text("# Sample\n\nsome body\n", encoding="utf-8")
    return path


def test_first_ingest_records_source_in_catalog(
    project: Path, sample_markdown: Path
) -> None:
    ingest(sample_markdown, project_root=project)
    catalog_path = project / "_registry" / "sources.json"
    assert catalog_path.exists()

    data = json.loads(catalog_path.read_text())
    assert len(data["sources"]) == 1
    entry = next(iter(data["sources"].values()))
    assert entry["name"] == "sample.md"
    assert entry["ingest_count"] == 1
    assert entry["raw_path"] is not None


def test_repeat_ingest_skips_pipeline_and_bumps_counter(
    project: Path, sample_markdown: Path
) -> None:
    ingest(sample_markdown, project_root=project)
    repo = git.Repo(project)
    head_after_first = repo.head.commit.hexsha

    result = ingest(sample_markdown, project_root=project)

    # No new commit (dedup skipped the pipeline entirely)
    assert repo.head.commit.hexsha == head_after_first
    # Result carries a dedup note and has no chunks
    assert any("already in catalog" in note for note in result.notes)
    assert result.chunks == []

    # Counter bumped
    data = json.loads((project / "_registry" / "sources.json").read_text())
    entry = next(iter(data["sources"].values()))
    assert entry["ingest_count"] == 2


def test_force_flag_reruns_pipeline_on_duplicate(
    project: Path, sample_markdown: Path
) -> None:
    ingest(sample_markdown, project_root=project)

    result = ingest(sample_markdown, project_root=project, force=True)

    # Pipeline ran (no dedup note) and chunks were produced
    assert not any("already in catalog" in note for note in result.notes)
    assert len(result.chunks) >= 1


# ----------------------------------------------------------------------
# URL dispatch (no network — just routing)
# ----------------------------------------------------------------------


def test_router_dispatches_https_url_to_web_extractor() -> None:
    extractor = router.route("https://example.com/paper.pdf")
    assert isinstance(extractor, WebExtractor)


def test_router_dispatches_http_url_to_web_extractor() -> None:
    extractor = router.route("http://example.com/paper")
    assert isinstance(extractor, WebExtractor)


def test_router_can_handle_reports_url() -> None:
    assert router.can_handle("https://example.com/paper.pdf") is True
