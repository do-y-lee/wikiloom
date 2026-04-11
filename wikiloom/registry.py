"""Manifest registry for WikiLoom page tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wikiloom.utils import now_iso, parse_iso


@dataclass
class PageEntry:
    """A single page entry in the manifest."""

    title: str
    type: str  # entity, concept, source, synthesis, decision
    status: str = "active"
    aliases: list[str] = field(default_factory=list)
    created: str = ""
    modified: str = ""
    summary: str = ""
    source_count: int = 0
    inbound_link_count: int = 0
    outbound_link_count: int = 0
    confidence: str = "medium"
    staleness_window_days: int = 90
    human_edited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "type": self.type,
            "status": self.status,
            "aliases": self.aliases,
            "created": self.created,
            "modified": self.modified,
            "summary": self.summary,
            "source_count": self.source_count,
            "inbound_link_count": self.inbound_link_count,
            "outbound_link_count": self.outbound_link_count,
            "confidence": self.confidence,
            "staleness_window_days": self.staleness_window_days,
            "human_edited": self.human_edited,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageEntry:
        return cls(
            title=data.get("title", ""),
            type=data.get("type", "entity"),
            status=data.get("status", "active"),
            aliases=data.get("aliases", []),
            created=data.get("created", ""),
            modified=data.get("modified", ""),
            summary=data.get("summary", ""),
            source_count=data.get("source_count", 0),
            inbound_link_count=data.get("inbound_link_count", 0),
            outbound_link_count=data.get("outbound_link_count", 0),
            confidence=data.get("confidence", "medium"),
            staleness_window_days=data.get("staleness_window_days", 90),
            human_edited=data.get("human_edited", False),
        )


@dataclass
class PageListItem:
    """Lightweight page info for LLM context injection."""

    page_id: str
    title: str
    type: str
    summary: str
    aliases: list[str]


class Registry:
    """Central manifest registry for all wiki pages.

    Loaded once per operation, modified in memory, saved once at end.
    """

    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.manifest_path = registry_dir / "manifest.json"
        self._version: int = 1
        self._updated_at: str = now_iso()
        self._pages: dict[str, PageEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load manifest from disk."""
        if not self.manifest_path.exists():
            return
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._version = data.get("version", 1)
        self._updated_at = data.get("updated_at", now_iso())
        for page_id, page_data in data.get("pages", {}).items():
            self._pages[page_id] = PageEntry.from_dict(page_data)

    def get_page(self, page_id: str) -> PageEntry | None:
        """Get a page entry by its ID."""
        return self._pages.get(page_id)

    def get_page_list(self) -> list[PageListItem]:
        """Return lightweight page list for LLM context injection."""
        items = []
        for page_id, entry in self._pages.items():
            if entry.status == "active":
                items.append(PageListItem(
                    page_id=page_id,
                    title=entry.title,
                    type=entry.type,
                    summary=entry.summary,
                    aliases=entry.aliases,
                ))
        return items

    def find_by_alias(self, alias: str) -> PageEntry | None:
        """Find a page by one of its aliases (case-insensitive)."""
        alias_lower = alias.lower()
        for entry in self._pages.values():
            if alias_lower in (a.lower() for a in entry.aliases):
                return entry
            if entry.title.lower() == alias_lower:
                return entry
        return None

    def register_page(self, page_id: str, entry: PageEntry) -> None:
        """Register a new page or update an existing one."""
        self._pages[page_id] = entry

    def deprecate_page(self, page_id: str, superseded_by: str | None = None) -> None:
        """Mark a page as deprecated."""
        entry = self._pages.get(page_id)
        if entry:
            entry.status = "deprecated"
            entry.modified = now_iso()

    def get_all_aliases(self) -> dict[str, str]:
        """Return a mapping of alias -> page_id for all active pages."""
        aliases: dict[str, str] = {}
        for page_id, entry in self._pages.items():
            if entry.status != "active":
                continue
            aliases[entry.title.lower()] = page_id
            for alias in entry.aliases:
                aliases[alias.lower()] = page_id
        return aliases

    def get_stale_pages(self) -> list[tuple[str, PageEntry]]:
        """Return pages that exceed their staleness window."""
        stale = []
        now = parse_iso(now_iso())
        for page_id, entry in self._pages.items():
            if entry.status != "active" or not entry.modified:
                continue
            modified = parse_iso(entry.modified)
            age_days = (now - modified).days
            if age_days > entry.staleness_window_days:
                stale.append((page_id, entry))
        return stale

    def save(self) -> None:
        """Save the manifest to disk."""
        self._updated_at = now_iso()
        data = {
            "version": self._version,
            "updated_at": self._updated_at,
            "pages": {
                page_id: entry.to_dict()
                for page_id, entry in self._pages.items()
            },
        }
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
