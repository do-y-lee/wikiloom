"""Manifest registry for WikiLoom page tracking."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from wikiloom.utils import now_iso, parse_iso


@dataclass
class PageEntry:
    """A single page entry in the manifest."""

    title: str
    type: str  # entity, concept, source, synthesis, decision
    page_id: str = ""  # populated by Registry; not persisted in to_dict
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
    superseded_by: str | None = None

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
            "superseded_by": self.superseded_by,
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
            superseded_by=data.get("superseded_by"),
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

    def __init__(self, registry_dir: Path, wiki_dir: Path | None = None):
        self.registry_dir = Path(registry_dir)
        self.manifest_path = self.registry_dir / "manifest.json"
        # Convention: wiki/ sits next to _registry/. Caller can override.
        self.wiki_dir = Path(wiki_dir) if wiki_dir else self.registry_dir.parent / "wiki"
        self._version: int = 1
        self._updated_at: str = now_iso()
        self._pages: dict[str, PageEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load manifest from disk."""
        if not self.manifest_path.exists():
            return
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._version = data.get("version", 1)
        self._updated_at = data.get("updated_at", now_iso())
        for page_id, page_data in data.get("pages", {}).items():
            entry = PageEntry.from_dict(page_data)
            entry.page_id = page_id
            self._pages[page_id] = entry

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

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @property
    def pages(self) -> dict[str, PageEntry]:
        """Read-only view of all pages by ID. Used by the linking engine."""
        return self._pages

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
        """Find a page by one of its aliases or title (case-insensitive)."""
        alias_lower = alias.lower()
        for entry in self._pages.values():
            if entry.title.lower() == alias_lower:
                return entry
            if alias_lower in (a.lower() for a in entry.aliases):
                return entry
        return None

    def get_all_aliases(self) -> dict[str, str]:
        """Return a mapping of alias -> page_id for all active pages.

        Includes each page's title and every alias, all lowercased.
        """
        aliases: dict[str, str] = {}
        for page_id, entry in self._pages.items():
            if entry.status != "active":
                continue
            aliases[entry.title.lower()] = page_id
            for alias in entry.aliases:
                aliases[alias.lower()] = page_id
        return aliases

    def get_stale_pages(self) -> list[PageEntry]:
        """Return active pages that exceed their staleness window."""
        stale: list[PageEntry] = []
        now = parse_iso(now_iso())
        for entry in self._pages.values():
            if entry.status != "active" or not entry.modified:
                continue
            try:
                modified = parse_iso(entry.modified)
            except ValueError:
                continue
            age_days = (now - modified).days
            if age_days > entry.staleness_window_days:
                stale.append(entry)
        return stale

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def register_page(self, page_id: str, entry: PageEntry) -> None:
        """Register a new page or update an existing one."""
        entry.page_id = page_id
        if not entry.created:
            entry.created = now_iso()
        entry.modified = now_iso()
        self._pages[page_id] = entry

    def update(self, entries: Iterable[tuple[str, PageEntry] | PageEntry]) -> None:
        """Bulk-register or refresh a collection of pages.

        Accepts either ``(page_id, PageEntry)`` tuples or ``PageEntry``
        objects whose ``page_id`` field is already populated. This is the
        method the ingest processor calls after writing pages to disk.
        """
        for item in entries:
            if isinstance(item, tuple):
                page_id, entry = item
            else:
                entry = item
                page_id = entry.page_id
                if not page_id:
                    raise ValueError(
                        "PageEntry passed to update() must have page_id set"
                    )
            self.register_page(page_id, entry)

    def add_alias(self, page_id: str, alias: str) -> None:
        """Add a new alias to an existing page (used by the review workflow)."""
        entry = self._pages.get(page_id)
        if entry is None:
            raise KeyError(f"Unknown page_id: {page_id}")
        normalized = alias.lower().strip()
        if normalized and normalized not in (a.lower() for a in entry.aliases):
            entry.aliases.append(normalized)
            entry.modified = now_iso()

    def deprecate_page(
        self,
        page_id: str,
        superseded_by: str | None = None,
        move_to_archive: bool = True,
        emit_event: bool = True,
    ) -> Path | None:
        """Mark a page as deprecated.

        Sets ``status="deprecated"``, records ``superseded_by`` if given,
        and (by default) moves the underlying ``.md`` file to
        ``wiki/archive/`` with a slug derived from the original page_id.
        Emits a DEPRECATE event to ``wiki/log.md`` unless ``emit_event``
        is False (set to False inside batch operations that emit their
        own aggregate events).

        Returns the new archive path if the file was moved, else None.
        """
        entry = self._pages.get(page_id)
        if entry is None:
            return None

        entry.status = "deprecated"
        entry.superseded_by = superseded_by
        entry.modified = now_iso()

        archive_path: Path | None = None
        if move_to_archive:
            source_path = self.wiki_dir / f"{page_id}.md"
            if source_path.exists():
                archive_dir = self.wiki_dir / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                # Preserve the original category in the archived filename so
                # different categories with the same slug don't collide.
                archive_name = page_id.replace("/", "__") + ".md"
                archive_path = archive_dir / archive_name
                shutil.move(str(source_path), str(archive_path))

        if emit_event:
            self._emit_deprecate_event(page_id, superseded_by)

        return archive_path

    def _emit_deprecate_event(
        self, page_id: str, superseded_by: str | None
    ) -> None:
        """Append a DEPRECATE entry to wiki/log.md.

        Imported lazily to avoid a circular dependency with events.py.
        """
        from wikiloom.events import EventType, append_event, create_event

        log_path = self.wiki_dir / "log.md"
        if not log_path.parent.exists():
            return  # no wiki/ directory — nothing to log to

        description = (
            f"{page_id} → {superseded_by}" if superseded_by else page_id
        )
        event = create_event(
            EventType.DEPRECATE,
            description=description,
            pages_deprecated=[page_id],
        )
        append_event(log_path, event)
