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
    """Append an event entry to the wiki log file."""
    entry = event.to_log_entry()
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        log_path.write_text(existing + "\n" + entry, encoding="utf-8")
    else:
        header = "# WikiLoom Event Log\n\n"
        log_path.write_text(header + entry, encoding="utf-8")


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
