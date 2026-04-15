"""Per-ingest resume checkpoint.

The ingest pipeline chunks a source, then (once Component 20 lands)
runs an LLM synthesis call per chunk. That loop can fail partway
through — rate limit, network blip, LLM refusal, parse error — and
without a checkpoint every already-synthesized chunk is thrown away
and re-paid for on the next run.

``IngestState`` persists the progress of a single in-flight ingest
to ``_registry/ingest_state.json``. It records the chunk plan up
front, marks each chunk done as the synthesis loop advances, and is
cleared by the processor on successful completion. A subsequent
``wikiloom ingest`` run for the same source can then detect the
leftover state and skip chunks already marked done.

Design notes
------------
- One active ingest at a time. The processor holds ``FileLock`` for
  the duration of the run, so we don't need concurrent-state support.
  If ``ingest_state.json`` exists on startup it means the *previous*
  run crashed.
- The checkpoint is keyed by source content hash (or URL) so we can
  match a resumed run to its state file. If the next ingest is for a
  different source, the stale state is discarded — we don't
  cross-contaminate.
- Serialization is plain JSON via ``write_text``. That matches the
  existing ``SourceCatalog`` / ``BacklinkRegistry`` pattern. Atomic
  rename is overkill: a torn write is detected as a JSON parse error
  and treated as "no state," which falls back to starting fresh.
- This module provides the *infrastructure*. The synthesis loop that
  calls ``mark_chunk_done`` lands with Component 20; for today the
  processor only writes the initial plan and clears the state on
  success, exercising the save/load/clear paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wikiloom.utils import now_iso

STATE_FILENAME = "ingest_state.json"


@dataclass
class ChunkState:
    """Per-chunk progress record."""

    index: int
    total: int
    token_estimate: int
    done: bool = False
    page_id: str | None = None  # set by the synthesis loop when a chunk produces a page
    error: str | None = None    # last-seen failure reason, if any

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "total": self.total,
            "token_estimate": self.token_estimate,
            "done": self.done,
            "page_id": self.page_id,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkState:
        return cls(
            index=int(data.get("index", 0)),
            total=int(data.get("total", 0)),
            token_estimate=int(data.get("token_estimate", 0)),
            done=bool(data.get("done", False)),
            page_id=data.get("page_id"),
            error=data.get("error"),
        )


@dataclass
class IngestState:
    """Durable progress record for a single in-flight ingest.

    The file lives at ``_registry/ingest_state.json`` while an ingest
    is running and is deleted on successful completion.
    """

    registry_dir: Path
    source_key: str = ""        # content_hash for files, URL for web sources
    source_name: str = ""       # display name for logs / error messages
    content_type: str = ""
    started_at: str = field(default_factory=now_iso)
    chunks: list[ChunkState] = field(default_factory=list)
    version: int = 1

    # ------------------------------------------------------------------
    # Path
    # ------------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return Path(self.registry_dir) / STATE_FILENAME

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def begin(
        cls,
        registry_dir: Path,
        source_key: str,
        source_name: str,
        content_type: str,
        chunks: list[ChunkState],
    ) -> IngestState:
        """Create a fresh state record for a new ingest run.

        Overwrites any prior state file. The processor calls this once
        the chunk plan is known; the returned instance is saved
        immediately so a crash before the first chunk completes still
        leaves a resume point.
        """
        state = cls(
            registry_dir=Path(registry_dir),
            source_key=source_key,
            source_name=source_name,
            content_type=content_type,
            chunks=list(chunks),
        )
        state.save()
        return state

    @classmethod
    def load(cls, registry_dir: Path) -> IngestState | None:
        """Return the existing state file, or None if absent / unreadable.

        A torn / invalid file is treated as "no state" so a corrupt
        checkpoint never blocks the next ingest.
        """
        path = Path(registry_dir) / STATE_FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None

        chunks_data = data.get("chunks") or []
        chunks = [ChunkState.from_dict(c) for c in chunks_data if isinstance(c, dict)]
        return cls(
            registry_dir=Path(registry_dir),
            source_key=data.get("source_key", ""),
            source_name=data.get("source_name", ""),
            content_type=data.get("content_type", ""),
            started_at=data.get("started_at", now_iso()),
            chunks=chunks,
            version=int(data.get("version", 1)),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Serialize to ``_registry/ingest_state.json``."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "source_key": self.source_key,
            "source_name": self.source_name,
            "content_type": self.content_type,
            "started_at": self.started_at,
            "updated_at": now_iso(),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        self.state_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def clear(self) -> None:
        """Delete the state file. Called on successful ingest completion."""
        if self.state_path.exists():
            self.state_path.unlink()

    # ------------------------------------------------------------------
    # Chunk progress
    # ------------------------------------------------------------------

    def mark_chunk_done(self, index: int, page_id: str | None = None) -> None:
        """Mark chunk ``index`` as completed. Persists to disk."""
        for chunk in self.chunks:
            if chunk.index == index:
                chunk.done = True
                chunk.page_id = page_id
                chunk.error = None
                break
        self.save()

    def mark_chunk_failed(self, index: int, error: str) -> None:
        """Record the last failure reason for a chunk. Persists to disk."""
        for chunk in self.chunks:
            if chunk.index == index:
                chunk.error = error
                break
        self.save()

    def pending_indices(self) -> list[int]:
        """Chunk indices that are not yet done, in order."""
        return [c.index for c in self.chunks if not c.done]

    def is_complete(self) -> bool:
        """True when every chunk is marked done (or there are no chunks)."""
        return all(c.done for c in self.chunks) if self.chunks else True

    def matches(self, source_key: str) -> bool:
        """Whether this state file belongs to a given source key.

        Used by the processor to decide whether a leftover state file
        is a resume candidate or cross-contamination from a prior run
        of a different source.
        """
        return bool(source_key) and self.source_key == source_key
