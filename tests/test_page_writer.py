"""Tests for the synthesis → disk page writer.

The most important assertion in this file is
``test_force_reingest_preserves_human_region`` — it's the invariant
check that keeps hand-authored content from being clobbered by LLM
rewrites.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from wikiloom.frontmatter import read_page
from wikiloom.ingest.page_writer import PageWriter
from wikiloom.protection import AUTO_MARKER, HumanEditProtection
from wikiloom.registry import Registry
from wikiloom.scaffold import init_project
from wikiloom.source_catalog import SourceEntry
from wikiloom.synthesis import (
    PageProposal,
    SourceSummary,
    SynthesisResult,
)
from wikiloom.utils import now_iso


# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = init_project(name="pw-test", path=tmp_path, domain="test")
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
def registry(project: Path) -> Registry:
    return Registry(project / "_registry")


def _source_entry(hash_: str = "abc123", name: str = "paper.pdf") -> SourceEntry:
    now = now_iso()
    return SourceEntry(
        content_hash=hash_,
        name=name,
        content_type="pdf",
        size_bytes=1024,
        raw_path=f"raw/papers/{name}",
        first_ingested_at=now,
        last_ingested_at=now,
    )


def _create_proposal(
    slug: str,
    title: str | None = None,
    type_: str = "concept",
    body: str | None = None,
    chunk_id: str = "chunk-aaa",
    confidence: str = "high",
) -> PageProposal:
    return PageProposal(
        intent="create",
        chunk_id=chunk_id,
        title=title or slug.replace("-", " ").title(),
        type=type_,
        suggested_slug=slug,
        content_markdown=body or f"# {slug}\n\nSynthesized body about {slug}.",
        confidence=confidence,
    )


def _update_proposal(
    existing_path: str,
    additions: str,
    chunk_id: str = "chunk-zzz",
) -> PageProposal:
    return PageProposal(
        intent="update",
        chunk_id=chunk_id,
        existing_path=existing_path,
        additions_markdown=additions,
    )


def _synth(
    *,
    creates: list[PageProposal] | None = None,
    updates: list[PageProposal] | None = None,
    source_title: str | None = "Paper Title",
) -> SynthesisResult:
    return SynthesisResult(
        source_summary=(
            SourceSummary(
                title=source_title,
                one_line=f"A one-line description of {source_title}.",
                content_markdown=f"## {source_title}\n\nOverall summary.",
            )
            if source_title
            else None
        ),
        pages_to_create=creates or [],
        pages_to_update=updates or [],
        entities_mentioned=[],
        concepts_mentioned=[],
        total_tokens_in=100,
        total_tokens_out=200,
        total_cost_usd=0.001,
        chunks_processed=1,
        chunks_failed=0,
        notes=[],
    )


# ----------------------------------------------------------------------
# Fresh create
# ----------------------------------------------------------------------


def test_fresh_create_writes_page(project: Path, registry: Registry) -> None:
    writer = PageWriter(project, registry)
    result = writer.write(
        _synth(creates=[_create_proposal("transformer", "Transformer")]),
        source_entry=_source_entry(),
    )

    target = project / "wiki" / "concepts" / "transformer.md"
    assert target.exists()
    assert any(p.name == "transformer.md" for p in result.created_paths)
    assert registry.get_page("concepts/transformer") is not None


def test_fresh_create_prepends_auto_marker(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo")]),
        source_entry=_source_entry(),
    )

    page = project / "wiki" / "concepts" / "foo.md"
    _, body = read_page(page)
    assert AUTO_MARKER in body
    assert body.split(AUTO_MARKER, 1)[0].strip() == ""  # nothing above marker


def test_fresh_create_writes_chunk_ids_in_frontmatter(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo", chunk_id="c-alpha")]),
        source_entry=_source_entry(),
    )

    fm, _ = read_page(project / "wiki" / "concepts" / "foo.md")
    assert fm is not None
    assert fm.chunk_ids == ["c-alpha"]


def test_fresh_create_writes_all_type_dirs(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(
            creates=[
                _create_proposal("alpha", type_="entity"),
                _create_proposal("beta", type_="concept"),
                _create_proposal("gamma", type_="synthesis"),
                _create_proposal("delta", type_="decision"),
            ]
        ),
        source_entry=_source_entry(),
    )

    assert (project / "wiki" / "entities" / "alpha.md").exists()
    assert (project / "wiki" / "concepts" / "beta.md").exists()
    assert (project / "wiki" / "syntheses" / "gamma.md").exists()
    assert (project / "wiki" / "decisions" / "delta.md").exists()


def test_fresh_create_rejects_unknown_type(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    with pytest.raises(ValueError, match="unknown page type"):
        writer.write(
            _synth(creates=[_create_proposal("foo", type_="bogus")]),
            source_entry=_source_entry(),
        )


# ----------------------------------------------------------------------
# Collision without --force: skip
# ----------------------------------------------------------------------


def test_collision_without_force_skips(project: Path, registry: Registry) -> None:
    writer = PageWriter(project, registry)
    # First ingest creates the page
    writer.write(
        _synth(creates=[_create_proposal("foo", body="first ingest body")]),
        source_entry=_source_entry(),
    )

    # Second ingest tries to create the same page without force
    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2)
    result = writer2.write(
        _synth(creates=[_create_proposal("foo", body="second ingest body")]),
        source_entry=_source_entry("def456", "other.pdf"),
    )

    assert "concepts/foo" in result.skipped_collisions
    _, body = read_page(project / "wiki" / "concepts" / "foo.md")
    assert "first ingest body" in body
    assert "second ingest body" not in body


# ----------------------------------------------------------------------
# Collision with --force: preserve human region
# ----------------------------------------------------------------------


def test_force_reingest_preserves_human_region(
    project: Path, registry: Registry
) -> None:
    """The invariant check. Hand-edited region above the marker must
    survive a forced re-ingest that rewrites the auto region."""
    writer = PageWriter(project, registry)
    # First ingest
    writer.write(
        _synth(
            creates=[_create_proposal("foo", body="Original LLM content about foo.")]
        ),
        source_entry=_source_entry(),
    )

    # Simulate a human edit: prepend their own content above the marker
    target = project / "wiki" / "concepts" / "foo.md"
    human_edit = "# Foo\n\n**My own take: this is actually about something else.**\n"
    original_text = target.read_text(encoding="utf-8")
    # Rewrite the file: human content above the marker, original auto content below
    fm_header, body = original_text.split("---\n", 2)[1], original_text.split("---\n", 2)[2]
    assert AUTO_MARKER in body
    new_body = human_edit + "\n" + body
    target.write_text(
        "---\n" + fm_header + "---\n" + new_body,
        encoding="utf-8",
    )

    # Forced re-ingest with different LLM content
    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2, force=True)
    writer2.write(
        _synth(
            creates=[
                _create_proposal(
                    "foo",
                    body="Totally different LLM content on re-ingest.",
                    chunk_id="c-new",
                )
            ]
        ),
        source_entry=_source_entry(),
    )

    _, final_body = read_page(target)
    # Human region survives
    assert "**My own take: this is actually about something else.**" in final_body
    # Auto region was replaced — old content gone, new content present
    assert "Totally different LLM content on re-ingest." in final_body
    assert "Original LLM content about foo." not in final_body
    # Marker still in place
    assert AUTO_MARKER in final_body


def test_force_reingest_unions_chunk_ids(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo", chunk_id="c-first")]),
        source_entry=_source_entry(),
    )

    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2, force=True)
    writer2.write(
        _synth(creates=[_create_proposal("foo", chunk_id="c-second")]),
        source_entry=_source_entry(),
    )

    fm, _ = read_page(project / "wiki" / "concepts" / "foo.md")
    assert fm is not None
    assert "c-first" in fm.chunk_ids
    assert "c-second" in fm.chunk_ids


# ----------------------------------------------------------------------
# pages_to_update (append)
# ----------------------------------------------------------------------


def test_update_appends_to_existing_auto_region(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    # Create the page first
    writer.write(
        _synth(
            creates=[_create_proposal("foo", body="First source content.")]
        ),
        source_entry=_source_entry("hash-A", "paperA.pdf"),
    )

    # Different source produces an update
    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2)
    writer2.write(
        _synth(
            source_title=None,  # skip source page to focus on the update path
            updates=[
                _update_proposal(
                    "concepts/foo",
                    "Additional content from paper B.",
                    chunk_id="c-from-b",
                )
            ],
        ),
        source_entry=_source_entry("hash-B", "paperB.pdf"),
    )

    _, body = read_page(project / "wiki" / "concepts" / "foo.md")
    assert "First source content." in body
    assert "Additional content from paper B." in body


def test_update_preserves_human_region(
    project: Path, registry: Registry
) -> None:
    """Human edits above the marker must survive pages_to_update too."""
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo", body="Original body")]),
        source_entry=_source_entry("hash-A"),
    )

    target = project / "wiki" / "concepts" / "foo.md"
    text = target.read_text(encoding="utf-8")
    fm_header, body = text.split("---\n", 2)[1], text.split("---\n", 2)[2]
    human_edit = "## My hand-written notes\n\nI think this page needs more context.\n\n"
    target.write_text(
        "---\n" + fm_header + "---\n" + human_edit + body,
        encoding="utf-8",
    )

    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2)
    writer2.write(
        _synth(
            source_title=None,
            updates=[_update_proposal("concepts/foo", "More auto content.")],
        ),
        source_entry=_source_entry("hash-B", "other.pdf"),
    )

    _, body = read_page(target)
    assert "My hand-written notes" in body
    assert "More auto content." in body


def test_update_unions_chunk_ids(project: Path, registry: Registry) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo", chunk_id="c-a")]),
        source_entry=_source_entry("hash-A"),
    )

    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2)
    writer2.write(
        _synth(
            source_title=None,
            updates=[
                _update_proposal("concepts/foo", "more", chunk_id="c-b"),
            ],
        ),
        source_entry=_source_entry("hash-B"),
    )

    fm, _ = read_page(project / "wiki" / "concepts" / "foo.md")
    assert fm is not None
    assert "c-a" in fm.chunk_ids
    assert "c-b" in fm.chunk_ids


def test_update_to_nonexistent_page_is_skipped(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    result = writer.write(
        _synth(
            source_title=None,
            updates=[_update_proposal("concepts/nonexistent", "new content")],
        ),
        source_entry=_source_entry(),
    )
    assert any("concepts/nonexistent" in n for n in result.notes)


def test_update_accepts_varied_path_formats(
    project: Path, registry: Registry
) -> None:
    """LLM can produce existing_path in any of several formats."""
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo")]),
        source_entry=_source_entry("hash-A"),
    )

    for raw_path in [
        "concepts/foo",
        "concepts/foo.md",
        "wiki/concepts/foo",
        "wiki/concepts/foo.md",
        "/wiki/concepts/foo.md",
    ]:
        registry2 = Registry(project / "_registry")
        writer2 = PageWriter(project, registry2)
        result = writer2.write(
            _synth(
                source_title=None,
                updates=[_update_proposal(raw_path, f"note for {raw_path}")],
            ),
            source_entry=_source_entry("hash-X"),
        )
        assert not any("target not found" in n for n in result.notes), (
            f"failed to resolve path: {raw_path}"
        )


def test_update_merges_contradictions(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo")]),
        source_entry=_source_entry("hash-A"),
    )

    proposal = _update_proposal("concepts/foo", "more")
    proposal.contradictions = [
        {
            "existing_claim": "X is fast",
            "new_claim": "X is slow",
            "source_location": "section 2",
        }
    ]
    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2)
    writer2.write(
        _synth(source_title=None, updates=[proposal]),
        source_entry=_source_entry("hash-B"),
    )

    fm, _ = read_page(project / "wiki" / "concepts" / "foo.md")
    assert fm is not None
    assert len(fm.contradictions) == 1
    assert fm.contradictions[0]["new_claim"] == "X is slow"


# ----------------------------------------------------------------------
# Source pages
# ----------------------------------------------------------------------


def test_source_page_is_written(project: Path, registry: Registry) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(
            creates=[_create_proposal("foo")],
            source_title="Flash Attention Paper",
        ),
        source_entry=_source_entry("hash-A", "flash-attention.pdf"),
    )

    src_page = project / "wiki" / "sources" / "flash-attention.md"
    assert src_page.exists()
    fm, _ = read_page(src_page)
    assert fm is not None
    assert fm.type == "source"
    assert any(s.get("hash") == "hash-A" for s in fm.sources)


def test_source_page_collision_with_different_hash_gets_counter(
    project: Path, registry: Registry
) -> None:
    """Two different files with the same stem get unique source slugs."""
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo")], source_title="Paper"),
        source_entry=_source_entry("hash-A", "paper.pdf"),
    )

    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2)
    writer2.write(
        _synth(creates=[_create_proposal("bar")], source_title="Different Paper"),
        source_entry=_source_entry("hash-B", "paper.pdf"),
    )

    assert (project / "wiki" / "sources" / "paper.md").exists()
    assert (project / "wiki" / "sources" / "paper-2.md").exists()


def test_source_page_same_hash_overwrites(
    project: Path, registry: Registry
) -> None:
    """Re-ingesting the same file overwrites its source page, not creates a new one."""
    writer = PageWriter(project, registry)
    writer.write(
        _synth(creates=[_create_proposal("foo")], source_title="Paper v1"),
        source_entry=_source_entry("hash-A", "paper.pdf"),
    )

    registry2 = Registry(project / "_registry")
    writer2 = PageWriter(project, registry2)
    writer2.write(
        _synth(creates=[_create_proposal("foo2")], source_title="Paper v2"),
        source_entry=_source_entry("hash-A", "paper.pdf"),
    )

    assert (project / "wiki" / "sources" / "paper.md").exists()
    assert not (project / "wiki" / "sources" / "paper-2.md").exists()
    fm, _ = read_page(project / "wiki" / "sources" / "paper.md")
    assert fm is not None
    assert fm.title == "Paper v2"  # latest wins


# ----------------------------------------------------------------------
# Registry integration
# ----------------------------------------------------------------------


def test_created_pages_are_registered(
    project: Path, registry: Registry
) -> None:
    writer = PageWriter(project, registry)
    writer.write(
        _synth(
            creates=[
                _create_proposal("alpha"),
                _create_proposal("beta"),
            ]
        ),
        source_entry=_source_entry(),
    )

    assert registry.get_page("concepts/alpha") is not None
    assert registry.get_page("concepts/beta") is not None
