"""Tests for wikiloom.backlinks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wikiloom.backlinks import (
    BacklinkRegistry,
    _extract_context,
    _extract_edges_from_body,
    _strip_code,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "_registry").mkdir()
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    (tmp_path / "wiki" / "entities").mkdir(parents=True)
    return tmp_path


def _write_page(project: Path, rel: str, body: str) -> Path:
    path = project / "wiki" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def test_strip_code_blanks_fenced_blocks_preserving_offsets() -> None:
    body = "before\n```\n[[fake-link]]\n```\nafter [[real]]"
    cleaned = _strip_code(body)
    assert len(cleaned) == len(body)
    assert "[[fake-link]]" not in cleaned
    assert "[[real]]" in cleaned


def test_strip_code_blanks_inline_code() -> None:
    body = "use `[[not-a-link]]` and [[real]]"
    cleaned = _strip_code(body)
    assert "[[not-a-link]]" not in cleaned
    assert "[[real]]" in cleaned


def test_extract_context_takes_preceding_line_snippet() -> None:
    body = "The transformer was developed at [[entities/google-brain]]."
    start = body.index("[[")
    ctx = _extract_context(body, start)
    assert "developed at" in ctx
    assert "[[" not in ctx


def test_extract_context_stops_at_newline() -> None:
    body = "line1 stuff\nrelated to [[x]]"
    start = body.index("[[")
    ctx = _extract_context(body, start)
    assert ctx == "related to"


def test_extract_context_collapses_whitespace() -> None:
    body = "foo   bar\t\tbaz [[x]]"
    start = body.index("[[")
    ctx = _extract_context(body, start)
    assert ctx == "foo bar baz"


def test_extract_edges_from_body_parses_wikilinks() -> None:
    body = "refers to [[concepts/transformer]] and [[entities/brain|the team]]."
    edges = _extract_edges_from_body(body, "sources/paper", "2026-04-13T00:00:00Z")
    targets = {e.target for e in edges}
    assert targets == {"concepts/transformer", "entities/brain"}
    assert all(e.source == "sources/paper" for e in edges)
    assert all(e.confidence == "high" for e in edges)


def test_extract_edges_skips_self_links() -> None:
    body = "see also [[concepts/transformer]]"
    edges = _extract_edges_from_body(body, "concepts/transformer", "t")
    assert edges == []


def test_extract_edges_ignores_links_in_code() -> None:
    body = "```python\n# [[concepts/not-real]]\n```\nreal: [[concepts/real]]"
    edges = _extract_edges_from_body(body, "sources/paper", "t")
    targets = {e.target for e in edges}
    assert targets == {"concepts/real"}


# ----------------------------------------------------------------------
# rebuild
# ----------------------------------------------------------------------


def test_rebuild_populates_edges_from_wiki(project: Path) -> None:
    _write_page(
        project,
        "concepts/transformer.md",
        "Developed at [[entities/google-brain]] and related to [[concepts/attention]].",
    )
    _write_page(
        project,
        "concepts/attention.md",
        "Core component of [[concepts/transformer]].",
    )

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    pairs = {(e.source, e.target) for e in reg._edges}
    assert ("concepts/transformer", "entities/google-brain") in pairs
    assert ("concepts/transformer", "concepts/attention") in pairs
    assert ("concepts/attention", "concepts/transformer") in pairs


def test_rebuild_skips_index_and_log_files(project: Path) -> None:
    _write_page(project, "index.md", "[[concepts/x]]")
    _write_page(project, "log.md", "[[concepts/y]]")
    _write_page(project, "concepts/real.md", "[[entities/z]]")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    sources = {e.source for e in reg._edges}
    assert sources == {"concepts/real"}


def test_rebuild_skips_archived_pages(project: Path) -> None:
    """Archive files would yield archive-prefixed source page_ids that
    don't exist in the manifest, producing spurious lint reports."""
    _write_page(project, "concepts/active.md", "[[concepts/target]]")
    _write_page(project, "concepts/target.md", "body")
    _write_page(project, "archive/concepts__old.md", "[[concepts/target]]")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    sources = {e.source for e in reg._edges}
    assert sources == {"concepts/active"}


def test_rebuild_preserves_linked_at_for_existing_edges(project: Path) -> None:
    _write_page(
        project,
        "concepts/a.md",
        "links to [[concepts/b]]",
    )
    _write_page(project, "concepts/b.md", "body")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()
    reg.save()

    original_ts = next(
        e.linked_at for e in reg._edges if (e.source, e.target) == ("concepts/a", "concepts/b")
    )

    # New registry instance reloads from disk; rebuild should keep
    # the original linked_at even with a later clock.
    reg2 = BacklinkRegistry(project / "_registry", project / "wiki")
    reg2.rebuild()
    preserved_ts = next(
        e.linked_at for e in reg2._edges if (e.source, e.target) == ("concepts/a", "concepts/b")
    )
    assert preserved_ts == original_ts


def test_rebuild_stamps_new_edges_with_current_time(project: Path) -> None:
    _write_page(project, "concepts/a.md", "links to [[concepts/b]]")
    _write_page(project, "concepts/b.md", "body")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()
    reg.save()

    _write_page(project, "concepts/a.md", "links to [[concepts/b]] and [[concepts/c]]")
    _write_page(project, "concepts/c.md", "body")

    reg2 = BacklinkRegistry(project / "_registry", project / "wiki")
    reg2.rebuild()
    ab_ts = next(
        e.linked_at for e in reg2._edges if (e.source, e.target) == ("concepts/a", "concepts/b")
    )
    ac_ts = next(
        e.linked_at for e in reg2._edges if (e.source, e.target) == ("concepts/a", "concepts/c")
    )
    # New edge has a ts >= old one (clock monotonic in the single test)
    assert ac_ts >= ab_ts


def test_rebuild_invalidates_cached_graph(project: Path) -> None:
    _write_page(project, "concepts/a.md", "[[concepts/b]]")
    _write_page(project, "concepts/b.md", "body")
    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    _ = reg.graph
    assert reg._graph is not None

    reg.rebuild()
    assert reg._graph is None


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_save_writes_version_and_last_linked_at(project: Path) -> None:
    _write_page(project, "concepts/a.md", "refers to [[concepts/b]]")
    _write_page(project, "concepts/b.md", "body")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()
    reg.save()

    data = json.loads((project / "_registry" / "backlinks.json").read_text())
    assert data["version"] == 1
    assert "updated_at" in data
    # concepts/b has an inbound from concepts/a, so last_linked_at is set
    assert data["links"]["concepts/b"]["last_linked_at"]
    assert data["links"]["concepts/b"]["inbound"][0]["from"] == "concepts/a"
    assert data["links"]["concepts/a"]["outbound"][0]["to"] == "concepts/b"


def test_save_is_deterministic_across_runs(project: Path) -> None:
    _write_page(project, "concepts/a.md", "[[concepts/b]] and [[entities/x]]")
    _write_page(project, "concepts/b.md", "body")
    _write_page(project, "entities/x.md", "body")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()
    reg.save()
    first = (project / "_registry" / "backlinks.json").read_text()

    reg2 = BacklinkRegistry(project / "_registry", project / "wiki")
    reg2.rebuild()
    reg2.save()
    second = (project / "_registry" / "backlinks.json").read_text()

    # Only the updated_at line differs; strip it and compare.
    def strip_ts(s: str) -> str:
        return "\n".join(line for line in s.splitlines() if "updated_at" not in line)

    assert strip_ts(first) == strip_ts(second)


def test_load_tolerates_missing_version_field(project: Path) -> None:
    path = project / "_registry" / "backlinks.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-04-13T00:00:00Z",
                "links": {
                    "concepts/b": {
                        "inbound": [
                            {
                                "from": "concepts/a",
                                "context": "refers to",
                                "confidence": "high",
                                "linked_at": "2026-04-13T00:00:00Z",
                            }
                        ],
                        "outbound": [],
                        "last_linked_at": "2026-04-13T00:00:00Z",
                    },
                    "concepts/a": {
                        "inbound": [],
                        "outbound": [
                            {
                                "to": "concepts/b",
                                "context": "refers to",
                                "confidence": "high",
                                "linked_at": "2026-04-13T00:00:00Z",
                            }
                        ],
                        "last_linked_at": "2026-04-13T00:00:00Z",
                    },
                },
            }
        )
    )

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    pairs = {(e.source, e.target) for e in reg._edges}
    assert ("concepts/a", "concepts/b") in pairs


# ----------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------


def test_get_most_connected_ranks_by_total_degree(project: Path) -> None:
    _write_page(project, "concepts/hub.md", "[[concepts/a]] [[concepts/b]] [[concepts/c]]")
    _write_page(project, "concepts/a.md", "[[concepts/hub]]")
    _write_page(project, "concepts/b.md", "[[concepts/hub]]")
    _write_page(project, "concepts/c.md", "body")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    top = reg.get_most_connected(n=3)
    assert top[0][0] == "concepts/hub"
    assert top[0][1] >= 5  # 3 outbound + 2 inbound


def test_get_link_path_is_undirected(project: Path) -> None:
    # flash-attention links to attention, but attention does NOT link back.
    _write_page(project, "concepts/flash-attention.md", "variant of [[concepts/attention]]")
    _write_page(project, "concepts/attention.md", "mechanism")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    path = reg.get_link_path("concepts/attention", "concepts/flash-attention")
    assert path == ["concepts/attention", "concepts/flash-attention"]


def test_get_link_path_returns_none_for_unknown_nodes(project: Path) -> None:
    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()
    assert reg.get_link_path("a", "b") is None


def test_get_link_path_returns_none_for_disconnected(project: Path) -> None:
    _write_page(project, "concepts/a.md", "[[concepts/b]]")
    _write_page(project, "concepts/b.md", "body")
    _write_page(project, "concepts/c.md", "[[concepts/d]]")
    _write_page(project, "concepts/d.md", "body")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    assert reg.get_link_path("concepts/a", "concepts/c") is None


# ----------------------------------------------------------------------
# link_counts
# ----------------------------------------------------------------------


def test_link_counts_tracks_in_and_out_degrees(project: Path) -> None:
    _write_page(project, "concepts/a.md", "[[concepts/b]] [[concepts/c]]")
    _write_page(project, "concepts/b.md", "[[concepts/c]]")
    _write_page(project, "concepts/c.md", "body")

    reg = BacklinkRegistry(project / "_registry", project / "wiki")
    reg.rebuild()

    counts = reg.link_counts()
    assert counts["concepts/a"] == (0, 2)  # 0 in, 2 out
    assert counts["concepts/b"] == (1, 1)
    assert counts["concepts/c"] == (2, 0)
