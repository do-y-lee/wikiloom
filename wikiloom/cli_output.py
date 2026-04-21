"""Shared CLI output helpers.

Small styling primitives — dim text, colored check marks, elapsed
time and token formatters, and a ``done_summary`` composer — used
by writer commands that want the "Done. A • B • C • 1.2s" look
pioneered in the ingest + rebuild-cache output polish. Keeping them
in one module means writer commands don't drift visually.
"""

from __future__ import annotations

import click


def dim(text: str) -> str:
    """Gray accent for separators and secondary info."""
    return click.style(text, fg="bright_black")


def check() -> str:
    """Green ✓ used to mark per-item success lines."""
    return click.style("✓", fg="green")


def cross() -> str:
    """Red ✗ used to mark skipped / failed items."""
    return click.style("✗", fg="red")


def skip_mark() -> str:
    """Yellow ↷ used to mark user-skipped items (distinct from failures)."""
    return click.style("↷", fg="yellow")


def separator() -> str:
    """Dim bullet separator for the Done-summary line."""
    return dim("•")


def format_elapsed(seconds: float) -> str:
    """Compact wall-clock for the end-of-command summary.

    >= 60s: ``Xm Ys``
    < 60s: ``X.Xs``
    """
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    return f"{seconds:.1f}s"


def format_tokens(n: int) -> str:
    """Human-readable token count: 1234 -> '1,234', 33800 -> '33.8k'."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


def done_summary(parts: list[str], elapsed: float | None = None) -> str:
    """Compose the final ``Done. A  •  B  •  C  •  1.2s`` line.

    ``parts`` is the ordered list of human-readable facts
    (``["8 merged", "0 skipped"]`` etc.). ``elapsed`` is appended as
    a formatted time when provided. Returns the whole line including
    the green bold ``Done.`` prefix — the caller just ``click.echo``s it.
    """
    done = click.style("Done.", fg="green", bold=True)
    sep = separator()
    segments = list(parts)
    if elapsed is not None:
        segments.append(format_elapsed(elapsed))
    body = f"  {sep}  ".join(segments)
    return f"  {done} {body}"
