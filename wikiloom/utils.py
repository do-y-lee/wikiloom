"""Utility functions for WikiLoom."""

from __future__ import annotations

import re
from datetime import datetime, timezone


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


def page_id_from_path(path: str) -> str:
    """Convert a relative wiki path to a page ID.

    e.g. 'entities/google-brain.md' -> 'entities/google-brain'
    """
    if path.endswith(".md"):
        path = path[:-3]
    return path
