"""Event sourcing for WikiLoom operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from wikiloom.utils import now_iso


class EventType(Enum):
    INGEST = "ingest"
    QUERY = "query"
    LINT = "lint"
    MERGE = "merge"
    DEPRECATE = "deprecate"
    PURGE = "purge"
    REINDEX = "reindex"
    RELINK = "relink"
    RELATED = "related"
    REBUILD_CACHE = "rebuild-cache"
    HUMAN_EDIT = "human-edit"
    SCHEMA_MIGRATION = "schema-migration"
    STUB_CREATED = "stub-created"


@dataclass
class WikiEvent:
    timestamp: str
    event_type: EventType
    description: str
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    pages_deprecated: list[str] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    links_inserted: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    git_commit_hash: str | None = None

    def to_log_entry(self) -> str:
        """Format this event as a markdown log entry for wiki/log.md."""
        lines = [
            f"## [{self.timestamp}] {self.event_type.value} | {self.description}",
        ]
        if self.pages_created:
            lines.append(f"- **Created**: {', '.join(self.pages_created)}")
        if self.pages_updated:
            lines.append(f"- **Updated**: {', '.join(self.pages_updated)}")
        if self.pages_deprecated:
            lines.append(f"- **Deprecated**: {', '.join(self.pages_deprecated)}")
        if self.contradictions:
            lines.append(f"- **Contradictions**: {len(self.contradictions)}")
        if self.links_inserted:
            lines.append(f"- **Links inserted**: {self.links_inserted}")
        if self.tokens_used:
            lines.append(f"- **Tokens used**: {self.tokens_used:,}")
        if self.cost_usd:
            lines.append(f"- **Cost**: ${self.cost_usd:.2f}")
        if self.git_commit_hash:
            lines.append(f"- **Commit**: {self.git_commit_hash[:8]}")
        lines.append("")
        return "\n".join(lines)


def append_event(log_path: Path, event: WikiEvent) -> None:
    """Append an event entry to the wiki log file.

    Locking convention
    ------------------
    This function does NOT acquire ``FileLock`` itself. Callers are
    responsible for holding the project lock if there's any chance of
    concurrent writers. In practice:

    - During ``wikiloom ingest``, the ingest processor holds ``FileLock``
      for the whole pipeline, so any events emitted by inner steps
      (linker stub creation, registry mutations) are already protected.
    - For standalone callers (e.g. a future ``wikiloom deprecate``
      command), the caller MUST acquire ``FileLock`` before calling this
      function or risk corrupting ``wiki/log.md``.

    The file is created with a header on first write if it doesn't exist.
    """
    entry = event.to_log_entry()
    if log_path.exists():
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n" + entry)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("# WikiLoom Event Log\n\n" + entry)


def create_event(
    event_type: EventType,
    description: str,
    **kwargs: object,
) -> WikiEvent:
    """Create a new WikiEvent with the current timestamp."""
    return WikiEvent(
        timestamp=now_iso(),
        event_type=event_type,
        description=description,
        **kwargs,  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------
# Log parsing
# ----------------------------------------------------------------------

import re

_LOG_HEADER_RE = re.compile(
    r"^## \[([^\]]+)\] (\S+) \| (.+)$"
)
_LOG_FIELD_RE = re.compile(
    r"^- \*\*([^*]+)\*\*: (.+)$"
)


def parse_log(log_path: Path) -> list[dict[str, str | int | float]]:
    """Parse ``wiki/log.md`` back into a list of event dicts.

    Returns newest-first. Each dict has at minimum ``timestamp``,
    ``event_type``, ``description``. Optional keys match the
    ``to_log_entry`` output: ``created``, ``updated``, ``tokens_used``,
    ``cost_usd``, ``commit``, etc.
    """
    if not log_path.exists():
        return []

    text = log_path.read_text(encoding="utf-8")
    entries: list[dict[str, str | int | float]] = []
    current: dict[str, str | int | float] | None = None

    for line in text.splitlines():
        header = _LOG_HEADER_RE.match(line)
        if header:
            if current is not None:
                entries.append(current)
            current = {
                "timestamp": header.group(1),
                "event_type": header.group(2),
                "description": header.group(3).strip(),
            }
            continue

        if current is None:
            continue

        field_match = _LOG_FIELD_RE.match(line)
        if field_match:
            key = field_match.group(1).strip().lower().replace(" ", "_")
            value: str | int | float = field_match.group(2).strip()
            if key == "tokens_used":
                value = int(str(value).replace(",", ""))
            elif key == "cost":
                value = float(str(value).lstrip("$").replace(",", ""))
                key = "cost_usd"
            current[key] = value

    if current is not None:
        entries.append(current)

    entries.reverse()  # newest first
    return entries
