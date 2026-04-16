"""Backlink Registry & Graph.

Turns ``[[target|display]]`` wikilinks embedded in wiki page bodies into
a queryable bidirectional graph. Persists to
``_registry/backlinks.json``; builds a NetworkX ``DiGraph`` lazily for
path-finding and connectivity queries.

Design notes
------------
- ``rebuild`` is a full scan today. The internals are structured around
  a per-file extractor (``_extract_edges_from_page``) so incremental
  updates can be added later without reshaping the public API.
- ``linked_at`` is preserved across rebuilds: if an edge already existed
  in ``backlinks.json``, its original timestamp is kept; only brand-new
  edges stamp ``now_iso()``.
- Every extracted edge defaults to ``confidence="high"``. Low-confidence
  candidates live in ``pending.json`` and never reach this file.
- ``context`` is the ~60 chars of plain text preceding the wikilink on
  its line, normalized to a single-line snippet. Good enough for the
  spec's "developed at" style; the linker can enrich later if needed.
- The NetworkX graph is lazy (``self._graph is None`` until first use)
  and ``rebuild`` invalidates it so callers that mix ``rebuild`` with
  graph queries can't get stale results.
- ``get_link_path`` traverses the graph **undirected**: a wiki user
  thinks of ``A`` and ``B`` as "connected" whether the arrow points
  forward or back. We keep the directed graph internally for
  ``get_most_connected`` (inbound vs outbound matter there).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wikiloom.utils import now_iso, page_id_from_path


# Wikilinks: [[target]] or [[target|display]]. Target captures 1 or
# optional display captures 2; we only persist the target.
_WIKILINK_RE = re.compile(r"\[\[([^\]|\n]+)(?:\|([^\]\n]+))?\]\]")

# Strip code so we don't pick up wikilink-looking text inside ``` fences
# or inline `code`. Backlink extraction is best-effort — we don't need
# the linker's full safe-zone machinery here.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_TILDE_FENCE_RE = re.compile(r"~~~.*?~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

_CONTEXT_CHARS = 60


@dataclass(frozen=True)
class BacklinkEdge:
    """A single directed edge in the wikilink graph."""

    source: str
    target: str
    context: str
    confidence: str
    linked_at: str


def _strip_code(body: str) -> str:
    """Remove fenced and inline code so they don't pollute link extraction.

    Replaces each code region with spaces of the same length so the rest
    of the body keeps its original character offsets — important because
    we use those offsets to slice out ``context``.
    """
    def blank(match: re.Match[str]) -> str:
        return " " * (match.end() - match.start())

    body = _FENCED_CODE_RE.sub(blank, body)
    body = _TILDE_FENCE_RE.sub(blank, body)
    body = _INLINE_CODE_RE.sub(blank, body)
    return body


def _extract_context(body: str, link_start: int) -> str:
    """Grab the snippet of body text preceding ``link_start``.

    Walks back at most ``_CONTEXT_CHARS`` characters, stops at a newline
    so we stay on the same line, and collapses internal whitespace.
    """
    line_start = body.rfind("\n", 0, link_start) + 1
    window_start = max(line_start, link_start - _CONTEXT_CHARS)
    snippet = body[window_start:link_start].strip()
    return re.sub(r"\s+", " ", snippet)


def _extract_edges_from_body(
    body: str,
    source_page_id: str,
    timestamp: str,
) -> list[BacklinkEdge]:
    """Extract wikilink edges from one page body."""
    cleaned = _strip_code(body)
    edges: list[BacklinkEdge] = []
    for match in _WIKILINK_RE.finditer(cleaned):
        target = match.group(1).strip()
        if not target or target == source_page_id:
            continue
        edges.append(
            BacklinkEdge(
                source=source_page_id,
                target=target,
                context=_extract_context(cleaned, match.start()),
                confidence="high",
                linked_at=timestamp,
            )
        )
    return edges


class BacklinkRegistry:
    """Reads/writes ``_registry/backlinks.json`` and serves graph queries."""

    def __init__(self, registry_dir: Path, wiki_dir: Path | None = None):
        self.registry_dir = Path(registry_dir)
        self.wiki_dir = (
            Path(wiki_dir) if wiki_dir else self.registry_dir.parent / "wiki"
        )
        self.backlinks_path = self.registry_dir / "backlinks.json"

        self._version: int = 1
        self._updated_at: str = now_iso()
        self._edges: list[BacklinkEdge] = []
        self._graph = None  # lazy NetworkX DiGraph
        self._undirected = None  # lazy cache for undirected traversal

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing ``backlinks.json`` if present.

        Tolerates missing ``version`` (treats as 1) so a vanilla file
        written by ``wikiloom init`` loads without fuss.
        """
        if not self.backlinks_path.exists():
            return
        try:
            data = json.loads(self.backlinks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return

        self._version = int(data.get("version", 1))
        self._updated_at = data.get("updated_at", now_iso())

        links = data.get("links", {}) or {}
        seen: set[tuple[str, str]] = set()
        for page_id, entry in links.items():
            for item in entry.get("outbound", []) or []:
                target = item.get("to")
                if not target:
                    continue
                key = (page_id, target)
                if key in seen:
                    continue
                seen.add(key)
                self._edges.append(
                    BacklinkEdge(
                        source=page_id,
                        target=target,
                        context=item.get("context", ""),
                        confidence=item.get("confidence", "high"),
                        linked_at=item.get("linked_at", self._updated_at),
                    )
                )

    def save(self) -> None:
        """Serialize edges to ``backlinks.json`` in deterministic order.

        Sorted keys + sorted edge lists keep git diffs minimal across
        rebuilds that don't change the underlying graph.
        """
        self._updated_at = now_iso()

        by_page: dict[str, dict[str, list[dict[str, str]]]] = {}

        def bucket(page_id: str) -> dict[str, list[dict[str, str]]]:
            return by_page.setdefault(page_id, {"inbound": [], "outbound": []})

        for edge in self._edges:
            bucket(edge.source)["outbound"].append(
                {
                    "to": edge.target,
                    "context": edge.context,
                    "confidence": edge.confidence,
                    "linked_at": edge.linked_at,
                }
            )
            bucket(edge.target)["inbound"].append(
                {
                    "from": edge.source,
                    "context": edge.context,
                    "confidence": edge.confidence,
                    "linked_at": edge.linked_at,
                }
            )

        links_out: dict[str, Any] = {}
        for page_id in sorted(by_page):
            entry = by_page[page_id]
            inbound = sorted(entry["inbound"], key=lambda e: (e["from"], e["linked_at"]))
            outbound = sorted(entry["outbound"], key=lambda e: (e["to"], e["linked_at"]))
            last = max(
                (e["linked_at"] for e in inbound + outbound if e.get("linked_at")),
                default="",
            )
            links_out[page_id] = {
                "inbound": inbound,
                "outbound": outbound,
                "last_linked_at": last,
            }

        data = {
            "version": self._version,
            "updated_at": self._updated_at,
            "links": links_out,
        }
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.backlinks_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild(self, wiki_dir: Path | None = None) -> None:
        """Scan every markdown page and regenerate the edge set.

        Preserves ``linked_at`` for edges that already existed; only
        brand-new edges get a fresh timestamp. Invalidates the cached
        NetworkX graph.
        """
        target_dir = Path(wiki_dir) if wiki_dir else self.wiki_dir
        previous: dict[tuple[str, str], BacklinkEdge] = {
            (e.source, e.target): e for e in self._edges
        }

        now = now_iso()
        new_edges: list[BacklinkEdge] = []
        seen: set[tuple[str, str]] = set()

        for md_path in sorted(target_dir.rglob("*.md")):
            if md_path.name == "index.md":
                continue
            if md_path.name == "log.md":
                continue
            page_id = page_id_from_path(target_dir, md_path)
            body = md_path.read_text(encoding="utf-8")
            for edge in self._extract_edges_from_page(body, page_id, now):
                key = (edge.source, edge.target)
                if key in seen:
                    continue
                seen.add(key)
                prior = previous.get(key)
                if prior is not None:
                    # Preserve the original linked_at; refresh context
                    # and confidence from the current body.
                    new_edges.append(
                        BacklinkEdge(
                            source=edge.source,
                            target=edge.target,
                            context=edge.context,
                            confidence=edge.confidence,
                            linked_at=prior.linked_at,
                        )
                    )
                else:
                    new_edges.append(edge)

        self._edges = new_edges
        self._graph = None
        self._undirected = None

    def _extract_edges_from_page(
        self,
        body: str,
        page_id: str,
        timestamp: str,
    ) -> list[BacklinkEdge]:
        """Wrapper so incremental update can call the same extractor."""
        return _extract_edges_from_body(body, page_id, timestamp)

    # ------------------------------------------------------------------
    # Edge access
    # ------------------------------------------------------------------

    @property
    def edges(self) -> list[BacklinkEdge]:
        """Read-only snapshot of all edges. Used by the SQLite cache sync."""
        return list(self._edges)

    # ------------------------------------------------------------------
    # Graph (lazy)
    # ------------------------------------------------------------------

    @property
    def graph(self):
        """Directed wikilink graph. Built on first access."""
        if self._graph is None:
            import networkx as nx

            g = nx.DiGraph()
            for edge in self._edges:
                g.add_edge(edge.source, edge.target)
            self._graph = g
        return self._graph

    def _undirected_graph(self):
        if self._undirected is None:
            self._undirected = self.graph.to_undirected()
        return self._undirected

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_orphans(self) -> list[str]:
        """Pages with zero inbound edges.

        Only considers pages that appear as an edge source or target in
        ``backlinks.json``. Truly-isolated pages (no edges at all in
        either direction) aren't tracked here; the linter's
        ``check_orphans`` should also consult the manifest.
        """
        inbound_counts: dict[str, int] = {}
        all_pages: set[str] = set()
        for edge in self._edges:
            all_pages.add(edge.source)
            all_pages.add(edge.target)
            inbound_counts[edge.target] = inbound_counts.get(edge.target, 0) + 1
        return sorted(p for p in all_pages if inbound_counts.get(p, 0) == 0)

    def get_most_connected(self, n: int = 10) -> list[tuple[str, int]]:
        """Top-N pages by total degree (inbound + outbound).

        Ties broken alphabetically so output is deterministic.
        """
        g = self.graph
        scored = [(page, g.in_degree(page) + g.out_degree(page)) for page in g.nodes]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:n]

    def get_link_path(self, source: str, target: str) -> list[str] | None:
        """Shortest undirected path between two pages, or None.

        Undirected because wiki users think of link paths as symmetric:
        ``A → B`` is just as much a connection as ``B → A``.
        """
        import networkx as nx

        g = self._undirected_graph()
        if source not in g or target not in g:
            return None
        try:
            return nx.shortest_path(g, source=source, target=target)
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------
    # Counts (for manifest sync)
    # ------------------------------------------------------------------

    def link_counts(self) -> dict[str, tuple[int, int]]:
        """Return ``{page_id: (inbound_count, outbound_count)}`` for all pages."""
        counts: dict[str, list[int]] = {}
        for edge in self._edges:
            counts.setdefault(edge.source, [0, 0])[1] += 1
            counts.setdefault(edge.target, [0, 0])[0] += 1
        return {page: (c[0], c[1]) for page, c in counts.items()}
