"""End-to-end smoke tests for ``wikiloom.ingest.processor.ingest``.

These tests verify the write-side pipeline composes correctly:
extraction → raw copy → synthesis (mocked) → backlinks rebuild →
manifest sync → index regeneration → single atomic git commit. The
LLM client is monkeypatched in every test so no real provider is
hit; a dedicated test verifies the synthesis → page-writer path
with a non-empty mocked response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import git
import pytest

import wikiloom.ingest.processor as processor_module
from wikiloom.frontmatter import read_page
from wikiloom.ingest.processor import ingest
from wikiloom.llm import LLMCallMetrics, SynthesizeResult
from wikiloom.scaffold import init_project


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Real init_project output with a git identity configured."""
    project_dir = init_project(name="testproj", path=tmp_path, domain="test")
    repo = git.Repo(project_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    # Baseline commit so HEAD is valid before any ingest.
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
    path.write_text(
        "# Sample\n\nA short document mentioning nothing in particular.\n",
        encoding="utf-8",
    )
    return path


def _empty_llm_response() -> SynthesizeResult:
    """Valid ingest_response.json shape with no pages or updates.

    Lets existing tests run through the synthesis stage without
    producing any new pages, so their assertions about commits and
    file counts stay accurate.
    """
    return SynthesizeResult(
        result={
            "source_summary": {
                "title": "Sample",
                "one_line": "A short sample document.",
                "content_markdown": "## About\n\nNothing in particular.",
            },
            "pages_to_create": [],
            "pages_to_update": [],
            "entities_mentioned": [],
            "concepts_mentioned": [],
        },
        metrics=LLMCallMetrics(
            tokens_in=20,
            tokens_out=15,
            cost_usd=0.0001,
            model="mock-model",
        ),
    )


def _install_mock_llm(
    monkeypatch: pytest.MonkeyPatch,
    response_builder=_empty_llm_response,
) -> MagicMock:
    """Patch LLMClient.synthesize so ingest tests never hit a real API.

    The processor imports ``LLMClient`` at module load time, so we
    monkeypatch the reference on the processor module rather than on
    the llm module. Returns the mock so tests can assert call counts.
    """
    mock = MagicMock()

    def _fake_synthesize(self: Any, system_prompt: str, user_prompt: str) -> Any:
        mock(system_prompt=system_prompt, user_prompt=user_prompt)
        return response_builder()

    monkeypatch.setattr(
        processor_module.LLMClient, "synthesize", _fake_synthesize
    )
    return mock


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Autouse-style mock for tests that don't care about synthesis output."""
    return _install_mock_llm(monkeypatch)


# ----------------------------------------------------------------------
# Smoke tests
# ----------------------------------------------------------------------


def test_ingest_writes_backlinks_file(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    ingest(sample_markdown, project_root=project)
    backlinks = project / "_registry" / "backlinks.json"
    assert backlinks.exists()
    data = json.loads(backlinks.read_text())
    assert data["version"] == 1
    assert "links" in data


def test_ingest_copies_source_to_raw(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    result = ingest(sample_markdown, project_root=project)
    assert result.raw_path is not None
    assert result.raw_path.exists()
    # Ends up under raw/articles/ for markdown (per RAW_DEST_BY_CONTENT_TYPE)
    assert "articles" in result.raw_path.parts


def test_ingest_result_carries_chunk_counts_on_success(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    """A successful synthesis exposes its per-chunk counts on IngestResult.

    The batch-ingest layer reads these to classify a file as complete /
    partial / failed, so they must be populated on the happy path.
    """
    result = ingest(sample_markdown, project_root=project)
    assert result.chunks_total > 0
    assert result.chunks_processed == result.chunks_total
    assert result.chunks_failed == 0


def test_ingest_result_chunk_counts_are_zero_on_dedup_skip(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    """Re-ingesting an already-catalogued source short-circuits before
    synthesis, so the counts stay at their zero defaults — the "synthesis
    didn't run" signal the batch layer relies on.
    """
    ingest(sample_markdown, project_root=project)
    second = ingest(sample_markdown, project_root=project)
    assert second.chunks_total == 0
    assert second.chunks_processed == 0
    assert second.chunks_failed == 0


def test_ingest_regenerates_index_files(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    root_index = project / "wiki" / "index.md"
    concepts_index = project / "wiki" / "concepts" / "index.md"

    ingest(sample_markdown, project_root=project)

    # Root index was rewritten in the IndexUpdater format
    after_root = root_index.read_text()
    assert "# Wiki Index" in after_root
    assert "## Concepts" in after_root
    # Sub-indexes now follow the table format (or empty placeholder)
    assert "Concepts Index" in concepts_index.read_text()


def test_ingest_produces_two_ingest_commits(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    """An ingest lands as a pair of commits: the data commit
    (pages/backlinks/indexes/raw) and a small tail commit that
    folds in ``wiki/log.md`` and ``_registry/sources.json`` so the
    working tree is never left dirty.
    """
    repo = git.Repo(project)
    commits_before = len(list(repo.iter_commits()))

    ingest(sample_markdown, project_root=project)

    commits_after = len(list(repo.iter_commits()))
    assert commits_after == commits_before + 2

    head = repo.head.commit
    prev = head.parents[0]
    assert head.message.startswith("ingest: log + catalog for sample.md")
    assert prev.message.startswith("ingest: sample.md")


def test_ingest_commit_includes_backlinks_and_indexes(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    """The ingest data commit should be atomic: raw copy + backlinks +
    indexes all land in HEAD~1 (the data commit); HEAD carries the
    log + catalog tail so the working tree stays clean.
    """
    repo = git.Repo(project)
    ingest(sample_markdown, project_root=project)

    data_commit = repo.head.commit.parents[0]
    changed = {
        item.a_path or item.b_path
        for item in data_commit.diff(data_commit.parents[0])
    }
    assert any("raw/articles/sample.md" in p for p in changed)
    assert any("_registry/backlinks.json" in p for p in changed)
    assert any("wiki/index.md" == p for p in changed)

    # Working tree is clean immediately after ingest — no residual dirt.
    assert repo.is_dirty(untracked_files=True) is False


def test_ingest_is_reentrant_on_identical_source(
    project: Path, sample_markdown: Path, mock_llm: MagicMock
) -> None:
    """Re-ingesting the same file should not pollute git history with
    empty commits. Second run's commit hash should equal HEAD from
    before (no-op) since the pipeline produces identical derived state."""
    repo = git.Repo(project)
    ingest(sample_markdown, project_root=project)
    head_after_first = repo.head.commit.hexsha

    ingest(sample_markdown, project_root=project)
    head_after_second = repo.head.commit.hexsha

    assert head_after_first == head_after_second


# ----------------------------------------------------------------------
# Session A: synthesis → page-writer end-to-end (mocked LLM)
# ----------------------------------------------------------------------


def _populated_llm_response() -> SynthesizeResult:
    """Mocked LLM response with real page proposals for end-to-end checks."""
    return SynthesizeResult(
        result={
            "source_summary": {
                "title": "Sample Doc",
                "one_line": "A test document about X and Y.",
                "content_markdown": "## About\n\nThis doc covers X and Y.",
            },
            "pages_to_create": [
                {
                    "type": "concept",
                    "suggested_slug": "topic-x",
                    "title": "Topic X",
                    "content_markdown": "# Topic X\n\nTopic X is a thing.",
                    "confidence": "high",
                },
                {
                    "type": "entity",
                    "suggested_slug": "org-y",
                    "title": "Org Y",
                    "content_markdown": "# Org Y\n\nOrg Y is an organization.",
                    "confidence": "medium",
                },
            ],
            "pages_to_update": [],
            "entities_mentioned": ["Org Y"],
            "concepts_mentioned": ["Topic X"],
        },
        metrics=LLMCallMetrics(
            tokens_in=500,
            tokens_out=300,
            cost_usd=0.0042,
            model="mock-model",
        ),
    )


def test_ingest_with_synthesis_writes_pages_end_to_end(
    project: Path,
    sample_markdown: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pipeline: extract → chunk → synthesize → page writer → commit.

    The single most important end-to-end assertion in the Session A
    scope. Proves that a real ingest of a file with a mocked LLM
    response actually produces pages on disk with chunk_ids in the
    frontmatter, all under one atomic git commit.
    """
    _install_mock_llm(monkeypatch, response_builder=_populated_llm_response)

    result = ingest(sample_markdown, project_root=project)

    # Pages were written
    topic_x = project / "wiki" / "concepts" / "topic-x.md"
    org_y = project / "wiki" / "entities" / "org-y.md"
    source_page = project / "wiki" / "sources" / "sample.md"
    assert topic_x.exists()
    assert org_y.exists()
    assert source_page.exists()

    # Each synthesized page carries the chunk_id that produced it,
    # nested under its source entry in `sources`.
    fm, _ = read_page(topic_x)
    assert fm is not None
    assert len(fm.all_chunk_ids()) == 1

    # Metrics populated on the IngestResult
    assert result.total_tokens_in == 500
    assert result.total_tokens_out == 300
    assert result.total_cost_usd == pytest.approx(0.0042)
    assert "concepts/topic-x" in result.pages_created
    assert "entities/org-y" in result.pages_created

    # Data commit (HEAD~1) includes every synthesized page; HEAD is the
    # log + catalog tail commit.
    repo = git.Repo(project)
    data_commit = repo.head.commit.parents[0]
    changed = {
        item.a_path or item.b_path
        for item in data_commit.diff(data_commit.parents[0])
    }
    assert "wiki/concepts/topic-x.md" in changed
    assert "wiki/entities/org-y.md" in changed
    assert "wiki/sources/sample.md" in changed


def test_ingest_links_pages_after_synthesis(
    project: Path,
    sample_markdown: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After synthesis, the linker should insert wikilinks between pages
    that reference each other's topics."""

    def _linking_response() -> SynthesizeResult:
        return SynthesizeResult(
            result={
                "source_summary": {
                    "title": "Doc",
                    "one_line": "D",
                    "content_markdown": "## About\n\nSummary.",
                },
                "pages_to_create": [
                    {
                        "type": "concept",
                        "suggested_slug": "flash-attention",
                        "title": "Flash Attention",
                        "content_markdown": (
                            "# Flash Attention\n\n"
                            "Flash Attention was introduced by Tri Dao."
                        ),
                        "confidence": "high",
                    },
                    {
                        "type": "entity",
                        "suggested_slug": "tri-dao",
                        "title": "Tri Dao",
                        "content_markdown": (
                            "# Tri Dao\n\n"
                            "Tri Dao created Flash Attention."
                        ),
                        "confidence": "high",
                    },
                ],
                "pages_to_update": [],
                "entities_mentioned": ["Tri Dao"],
                "concepts_mentioned": ["Flash Attention"],
            },
            metrics=LLMCallMetrics(
                tokens_in=300, tokens_out=200, cost_usd=0.002, model="mock"
            ),
        )

    _install_mock_llm(monkeypatch, response_builder=_linking_response)
    result = ingest(sample_markdown, project_root=project)

    # Both pages should exist
    fa_page = project / "wiki" / "concepts" / "flash-attention.md"
    td_page = project / "wiki" / "entities" / "tri-dao.md"
    assert fa_page.exists()
    assert td_page.exists()

    # The linker should have inserted wikilinks — Flash Attention page
    # mentions "Tri Dao" which should now be [[entities/tri-dao|Tri Dao]]
    fa_body = fa_page.read_text(encoding="utf-8")
    td_body = td_page.read_text(encoding="utf-8")

    assert "[[entities/tri-dao" in fa_body or "Tri Dao" in fa_body
    assert "[[concepts/flash-attention" in td_body or "Flash Attention" in td_body

    # Backlinks should reflect the cross-references
    import json

    bl_data = json.loads(
        (project / "_registry" / "backlinks.json").read_text(encoding="utf-8")
    )
    links = bl_data.get("links", {})
    # At least one page should have inbound or outbound links
    has_links = any(
        entry.get("inbound") or entry.get("outbound")
        for entry in links.values()
    )
    assert has_links, "Expected at least one backlink edge after linking"


def test_ingest_populates_event_log_with_real_tokens(
    project: Path,
    sample_markdown: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WikiEvent for the ingest should carry real token + cost numbers."""
    _install_mock_llm(monkeypatch, response_builder=_populated_llm_response)

    ingest(sample_markdown, project_root=project)

    log_content = (project / "wiki" / "log.md").read_text()
    # Total tokens = 500 in + 300 out = 800. Event log uses the
    # "**Tokens used**: <n>" format emitted by events.to_log_entry.
    assert "**Tokens used**: 800" in log_content
    # Cost rounds to $0.00 at current pricing — the important thing
    # is that a cost line exists, not its exact value.
    assert "**Cost**" in log_content


def test_ingest_url_writes_pages_and_records_catalog_entry(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL ingests go through the same extract→synthesize→write path as files.

    Regression guard: the synthesis block used to be gated on
    ``not is_url and source_path.is_file()``, so URL ingests silently
    produced zero pages and an empty commit. Lifting the gate — plus
    hashing the extracted text at step 1b — should make URLs behave
    like any other source, including writing synthesized pages,
    recording a catalog entry keyed by content hash, and carrying the
    ``url`` field on that entry.
    """
    _install_mock_llm(monkeypatch, response_builder=_populated_llm_response)

    # Stub WebExtractor so the test doesn't hit the network. Returns a
    # fixed ExtractedContent — enough tokens to exercise the chunker
    # without triggering it.
    from wikiloom.ingest.extractors.base import ExtractedContent

    def _fake_extract(self: Any, path: Any) -> ExtractedContent:
        return ExtractedContent(
            text="Some text about Topic X and Org Y on a web page.",
            metadata={"url": str(path)},
            source_path=Path(str(path)),
            content_type="web",
            extraction_method="stub",
            token_estimate=20,
        )

    monkeypatch.setattr(
        "wikiloom.ingest.extractors.web_ext.WebExtractor.extract",
        _fake_extract,
    )

    url = "https://example.com/test-page"
    result = ingest(url, project_root=project)

    # Synthesis produced pages on disk
    topic_x = project / "wiki" / "concepts" / "topic-x.md"
    org_y = project / "wiki" / "entities" / "org-y.md"
    assert topic_x.exists()
    assert org_y.exists()
    assert "concepts/topic-x" in result.pages_created
    assert "entities/org-y" in result.pages_created

    # Catalog entry exists, keyed by the hash of the extracted text,
    # and carries the URL (no raw_path since nothing was copied).
    from wikiloom.source_catalog import SourceCatalog, hash_text

    catalog = SourceCatalog(project / "_registry")
    expected_hash = hash_text(
        "Some text about Topic X and Org Y on a web page."
    )
    entry = catalog.get(expected_hash)
    assert entry is not None
    assert entry.url == url
    assert entry.raw_path is None
    assert entry.content_type == "web"
    assert set(entry.pages_produced) >= {"concepts/topic-x", "entities/org-y"}


def test_ingest_url_is_reentrant_on_identical_content(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-ingesting the same URL with unchanged content is a cheap no-op.

    Dedup key is the hash of the extracted text, so if trafilatura
    returns the same text on the second fetch, the second ingest
    bumps ingest_count without re-running synthesis.
    """
    from wikiloom.ingest.extractors.base import ExtractedContent

    calls = {"extract": 0}

    def _fake_extract(self: Any, path: Any) -> ExtractedContent:
        calls["extract"] += 1
        return ExtractedContent(
            text="Stable content for dedup test.",
            metadata={"url": str(path)},
            source_path=Path(str(path)),
            content_type="web",
            extraction_method="stub",
            token_estimate=8,
        )

    monkeypatch.setattr(
        "wikiloom.ingest.extractors.web_ext.WebExtractor.extract",
        _fake_extract,
    )
    llm_mock = _install_mock_llm(
        monkeypatch, response_builder=_populated_llm_response
    )

    url = "https://example.com/stable"
    ingest(url, project_root=project)
    synth_calls_first = llm_mock.call_count
    ingest(url, project_root=project)

    # Second ingest still hits the extractor (needed to hash), but
    # should skip synthesis entirely once the catalog dedup fires.
    assert llm_mock.call_count == synth_calls_first
    assert calls["extract"] == 2


def test_post_merge_relink_scopes_cache_sync_to_linked_pages(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run_post_merge_relink`` must NOT trigger a blind full_rebuild.

    Regression guard: the previous version called
    ``sync_from_files(project_root, embedder=...)`` with no
    ``changed_files``, which silently fell through to ``full_rebuild``.
    Under any embedder hiccup at that moment, the whole wiki's
    embeddings ended up NULL because full_rebuild DROPs the pages
    table before re-embedding.

    Now the sync must pass ``changed_files=<linked>`` so at worst
    only the handful of pages the linker actually modified are
    re-embedded, and unaffected rows keep their existing embeddings.
    """
    from wikiloom.ingest.processor import _run_post_merge_relink
    from wikiloom.config import Config

    # Seed a page so rglob finds at least one .md file.
    wiki_dir = project / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (wiki_dir / "concepts" / "alpha.md").write_text(
        "---\ntitle: Alpha\ntype: concept\nstatus: active\n---\nBody.\n",
        encoding="utf-8",
    )

    # Stub LinkingEngine.link_all to return a fixed "linked" list of
    # one page — the thing we want to verify gets passed through.
    linked_paths = [wiki_dir / "concepts" / "alpha.md"]

    import wikiloom.ingest.processor as processor_module

    class _StubLinker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def link_all(self, pages: list[Path], progress: Any = None) -> list[Path]:
            return linked_paths

    monkeypatch.setattr(
        "wikiloom.linker.LinkingEngine", _StubLinker, raising=True
    )

    # Capture the sync_from_files arguments to prove changed_files is
    # passed (not None → no full_rebuild).
    from wikiloom.cache import SQLiteCache

    sync_calls: list[dict[str, Any]] = []
    real_sync = SQLiteCache.sync_from_files

    def spy_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
        # args[0] is project_root; kwargs carry changed_files/embedder
        sync_calls.append({
            "project_root": args[0] if args else None,
            "changed_files": kwargs.get("changed_files"),
            "has_embedder": kwargs.get("embedder") is not None,
        })
        return real_sync(self, *args, **kwargs)

    monkeypatch.setattr(SQLiteCache, "sync_from_files", spy_sync)

    # Stub git commit — no repo state needed for this test.
    class _StubRepoGit:
        def add(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _StubRepo:
        git = _StubRepoGit()

    class _StubGitOps:
        repo = _StubRepo()

        def commit(self, *args: Any, **kwargs: Any) -> None:
            pass

    cfg = Config(project_root=project)
    _run_post_merge_relink(project, cfg, _StubGitOps())

    # Exactly one sync should have happened, with changed_files set
    # to the linker's output — NOT None.
    assert len(sync_calls) == 1
    call = sync_calls[0]
    assert call["changed_files"] == linked_paths
    assert call["changed_files"] is not None
