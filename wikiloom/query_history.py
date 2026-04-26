"""Rolling cache of ``wikiloom query`` results.

Each successful query appends a structured record to
``_registry/query_history.json`` so users can browse, replay, or
promote older answers to synthesis pages without re-running the (paid)
LLM call. The file is gitignored — it's per-machine state, not project
state — and may contain sensitive prompts; users can opt out via
``[query] history_enabled = false`` in ``wikiloom.toml``.

Newest entries are stored first (index 0 = most recent), so
``history[0]`` collapses what ``last_query.json`` used to be.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from wikiloom.utils import now_iso


HISTORY_FILENAME = "query_history.json"
LEGACY_LAST_QUERY_FILENAME = "last_query.json"
SCHEMA_VERSION = 1


@dataclass
class QueryHistoryEntry:
    """One row in ``query_history.json``.

    Captures everything ``--detail`` shows plus the structural metadata
    needed to reconstruct what the user asked, what the wiki said, how
    confident, and what it cost.
    """

    query_id: str
    timestamp: str
    question: str
    answer: str
    relevance: str = ""
    confidence: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    pages_consulted: int = 0
    followups: list[str] = field(default_factory=list)
    suggest_synthesis: bool = False
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryHistoryEntry:
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


def derive_query_id(question: str, timestamp: str) -> str:
    """Stable short hash for addressing entries from the CLI."""
    digest = hashlib.sha256(
        (timestamp + "|" + question).encode("utf-8")
    ).hexdigest()
    return digest[:10]


@dataclass
class QueryHistory:
    """Read / write helper for ``query_history.json``.

    All writes go through ``_atomic_write_text`` (write-temp-then-
    rename) so concurrent ``wikiloom query`` runs can't clobber the
    file mid-write. Reads tolerate a missing file (empty history).
    """

    path: Path
    entries: list[QueryHistoryEntry] = field(default_factory=list)

    @classmethod
    def load(cls, registry_dir: Path) -> QueryHistory:
        path = registry_dir / HISTORY_FILENAME
        if not path.exists():
            return cls(path=path, entries=[])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls(path=path, entries=[])
        raw_entries = data.get("entries", []) if isinstance(data, dict) else []
        entries = [QueryHistoryEntry.from_dict(e) for e in raw_entries]
        return cls(path=path, entries=entries)

    def append(
        self, entry: QueryHistoryEntry, *, max_entries: int
    ) -> None:
        """Prepend ``entry`` (newest first) and trim to ``max_entries``."""
        self.entries.insert(0, entry)
        if max_entries > 0 and len(self.entries) > max_entries:
            del self.entries[max_entries:]

    def save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "entries": [e.to_dict() for e in self.entries],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, text)

    def latest(self) -> QueryHistoryEntry | None:
        return self.entries[0] if self.entries else None

    def get(self, query_id: str) -> QueryHistoryEntry | None:
        """Lookup by query_id or by 1-based index string ('1' = most recent)."""
        if query_id.isdigit():
            idx = int(query_id) - 1
            if 0 <= idx < len(self.entries):
                return self.entries[idx]
            return None
        # query_id is a 10-char prefix; allow shorter matches if unambiguous
        matches = [e for e in self.entries if e.query_id.startswith(query_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    def migrate_legacy(self, registry_dir: Path) -> bool:
        """Seed an empty history from ``last_query.json`` if present.

        Returns True if a migration happened. Removes the legacy file
        afterward so the next write doesn't re-import a stale snapshot.
        """
        if self.entries:
            return False
        legacy = registry_dir / LEGACY_LAST_QUERY_FILENAME
        if not legacy.exists():
            return False
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            legacy.unlink(missing_ok=True)
            return False
        question = data.get("question", "")
        timestamp = data.get("timestamp", now_iso())
        entry = QueryHistoryEntry(
            query_id=derive_query_id(question, timestamp),
            timestamp=timestamp,
            question=question,
            answer=data.get("answer", ""),
            confidence=data.get("confidence", ""),
            sources=data.get("sources_consulted", []) or [],
            followups=data.get("suggested_followups", []) or [],
            suggest_synthesis=bool(data.get("suggest_synthesis", False)),
            tokens_in=int(data.get("tokens_in", 0)),
            tokens_out=int(data.get("tokens_out", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
        )
        self.entries.append(entry)
        legacy.unlink(missing_ok=True)
        return True


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Writes to a temporary file in the same directory, then ``rename``s
    on top of the target — this is atomic on POSIX and avoids leaving
    a half-written file if the process is killed mid-write.
    """
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup — don't mask the original error.
        Path(tmp).unlink(missing_ok=True)
        raise
