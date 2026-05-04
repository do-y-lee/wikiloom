"""YAML frontmatter parsing and writing for wiki pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from yaml import CSafeLoader as _SafeLoader, CSafeDumper as _SafeDumper
except ImportError:
    from yaml import SafeLoader as _SafeLoader, SafeDumper as _SafeDumper  # type: ignore[assignment]


@dataclass
class Frontmatter:
    """Represents YAML frontmatter for a wiki page.

    Provenance lives inside ``sources`` — each source dict carries
    its own ``chunk_ids`` list. Use ``all_chunk_ids()`` to get a
    flat list across all sources.
    """

    title: str
    type: str  # entity, concept, source, synthesis, decision
    status: str = "active"
    created: str = ""
    modified: str = ""
    summary: str = ""
    aliases: list[str] = field(default_factory=list)
    # Each source dict: {"hash": str, "name": str, "raw_path": str,
    # "chunk_ids": list[str]}
    sources: list[dict[str, Any]] = field(default_factory=list)
    source_count: int = 0
    confidence: str = "medium"
    dormant_window_days: int = 90
    human_edited: bool = False
    human_edited_at: str | None = None
    superseded_by: str | None = None
    contradictions: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related_pages: list[str] = field(default_factory=list)
    # Set True when this page was created because an LLM ``pages_to_update``
    # proposal referenced a page_id that didn't exist — the writer promoted
    # it to a create to preserve the content. Surfaced by ``wikiloom lint``
    # so users can review whether the content really belongs in the wiki.
    # The user clears it via an edit + ``wikiloom save`` after review.
    promoted_from_update: bool = False

    def all_chunk_ids(self) -> list[str]:
        """Flatten chunk_ids across all sources, preserving order."""
        out: list[str] = []
        seen: set[str] = set()
        for src in self.sources:
            if not isinstance(src, dict):
                continue
            for cid in src.get("chunk_ids") or []:
                if cid and cid not in seen:
                    seen.add(cid)
                    out.append(cid)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict suitable for YAML serialization."""
        return {
            "title": self.title,
            "type": self.type,
            "status": self.status,
            "created": self.created,
            "modified": self.modified,
            "summary": self.summary,
            "aliases": self.aliases,
            "sources": self.sources,
            "source_count": self.source_count,
            "confidence": self.confidence,
            "dormant_window_days": self.dormant_window_days,
            "human_edited": self.human_edited,
            "human_edited_at": self.human_edited_at,
            "superseded_by": self.superseded_by,
            "contradictions": self.contradictions,
            "tags": self.tags,
            "related_pages": self.related_pages,
            "promoted_from_update": self.promoted_from_update,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Frontmatter:
        """Create a Frontmatter instance from a dict."""
        return cls(
            title=data.get("title", ""),
            type=data.get("type", "entity"),
            status=data.get("status", "active"),
            created=data.get("created", ""),
            modified=data.get("modified", ""),
            summary=data.get("summary", ""),
            aliases=data.get("aliases", []),
            sources=data.get("sources", []),
            source_count=data.get("source_count", 0),
            confidence=data.get("confidence", "medium"),
            dormant_window_days=data.get(
                "dormant_window_days",
                data.get("staleness_window_days", 90),  # legacy fallback
            ),
            human_edited=data.get("human_edited", False),
            human_edited_at=data.get("human_edited_at"),
            superseded_by=data.get("superseded_by"),
            contradictions=data.get("contradictions", []),
            tags=data.get("tags", []),
            related_pages=data.get("related_pages", []),
            promoted_from_update=bool(
                data.get("promoted_from_update", False)
            ),
        )


def parse_frontmatter(text: str) -> tuple[Frontmatter | None, str]:
    """Parse YAML frontmatter from markdown text.

    Returns (frontmatter, body) where body is the content after the
    frontmatter block. If no frontmatter is found, returns (None, text).
    """
    if not text.startswith("---"):
        return None, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text

    try:
        data = yaml.load(parts[1], Loader=_SafeLoader)
    except yaml.YAMLError:
        return None, text

    if not isinstance(data, dict):
        return None, text

    fm = Frontmatter.from_dict(data)
    body = parts[2].lstrip("\n")
    return fm, body


def render_frontmatter(fm: Frontmatter) -> str:
    """Render a Frontmatter object to a YAML frontmatter string."""
    data = fm.to_dict()
    yaml_str = yaml.dump(
        data,
        Dumper=_SafeDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    return f"---\n{yaml_str}---\n"


def read_page(path: Path) -> tuple[Frontmatter | None, str]:
    """Read a wiki page file and return its frontmatter and body."""
    text = path.read_text(encoding="utf-8")
    return parse_frontmatter(text)


def write_page(path: Path, fm: Frontmatter, body: str) -> None:
    """Write a wiki page file with frontmatter and body content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_frontmatter(fm) + "\n" + body
    path.write_text(content, encoding="utf-8")
