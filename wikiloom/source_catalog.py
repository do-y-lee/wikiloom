"""Content-addressed catalog of ingested sources.

Persists to _registry/sources.json, keyed by SHA-256 of file bytes.
Re-ingesting an identical file is a cheap no-op via hash lookup.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wikiloom.utils import now_iso


def hash_file(path: Path) -> str:
    """Compute SHA-256 of a file's bytes as a lowercase hex string."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SourceEntry:
    """One row in the source catalog."""

    content_hash: str
    name: str
    content_type: str
    size_bytes: int
    raw_path: str | None               # relative to project root; None for URL/derived
    first_ingested_at: str
    last_ingested_at: str
    ingest_count: int = 1
    url: str | None = None
    pages_produced: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "raw_path": self.raw_path,
            "first_ingested_at": self.first_ingested_at,
            "last_ingested_at": self.last_ingested_at,
            "ingest_count": self.ingest_count,
            "url": self.url,
            "pages_produced": sorted(self.pages_produced),
        }

    @classmethod
    def from_dict(cls, content_hash: str, data: dict[str, Any]) -> SourceEntry:
        return cls(
            content_hash=content_hash,
            name=data.get("name", ""),
            content_type=data.get("content_type", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            raw_path=data.get("raw_path"),
            first_ingested_at=data.get("first_ingested_at", ""),
            last_ingested_at=data.get("last_ingested_at", ""),
            ingest_count=int(data.get("ingest_count", 1)),
            url=data.get("url"),
            pages_produced=list(data.get("pages_produced", [])),
        )


class SourceCatalog:
    """Read/write layer over ``_registry/sources.json``."""

    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)
        self.catalog_path = self.registry_dir / "sources.json"
        self._version = 1
        self._updated_at = now_iso()
        self._entries: dict[str, SourceEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.catalog_path.exists():
            return
        try:
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        self._version = int(data.get("version", 1))
        self._updated_at = data.get("updated_at", now_iso())
        for content_hash, entry in (data.get("sources") or {}).items():
            self._entries[content_hash] = SourceEntry.from_dict(content_hash, entry)

    def save(self) -> None:
        self._updated_at = now_iso()
        data = {
            "version": self._version,
            "updated_at": self._updated_at,
            "sources": {
                h: self._entries[h].to_dict() for h in sorted(self._entries)
            },
        }
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Lookup / mutation
    # ------------------------------------------------------------------

    def has(self, content_hash: str) -> bool:
        return content_hash in self._entries

    def get(self, content_hash: str) -> SourceEntry | None:
        return self._entries.get(content_hash)

    def record(self, entry: SourceEntry) -> None:
        """Insert or overwrite an entry. Does not call ``save``."""
        self._entries[entry.content_hash] = entry

    def touch(self, content_hash: str) -> SourceEntry | None:
        """Bump ``ingest_count`` and ``last_ingested_at`` for a known hash.

        Used when ``--force`` re-ingests an already-seen source. Returns
        the updated entry or None if the hash is unknown.
        """
        entry = self._entries.get(content_hash)
        if entry is None:
            return None
        entry.ingest_count += 1
        entry.last_ingested_at = now_iso()
        return entry
