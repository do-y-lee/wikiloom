"""Utility functions for WikiLoom."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    return max(1, len(text) // 4)


def slugify(text: str) -> str:
    """Convert text to a lowercase hyphenated slug.

    No abbreviations, no special characters.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(timestamp: str) -> datetime:
    """Parse an ISO 8601 timestamp string."""
    # Handle both Z suffix and +00:00
    timestamp = timestamp.replace("Z", "+00:00")
    return datetime.fromisoformat(timestamp)


def page_id_from_path(wiki_dir: Path, md_path: Path) -> str:
    """Canonical ``category/slug`` page_id for a markdown file.

    The one-true conversion used by linker, backlinks, lint, and the
    IndexUpdater. Keeping it here stops each component from re-inventing
    the same relative-path dance.

    Example: ``/proj/wiki/concepts/attention.md`` → ``concepts/attention``.
    """
    rel = Path(md_path).resolve().relative_to(Path(wiki_dir).resolve())
    return rel.with_suffix("").as_posix()
