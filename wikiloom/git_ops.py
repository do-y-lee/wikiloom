"""Git integration.

Thin wrapper over GitPython that enforces WikiLoom's commit-message
convention so downstream tools (linter, ``get_file_history``,
``is_human_edited``) can classify commits by parsing their subject line.

Commit subject grammar::

    <type>: <description> [<stats>]

Where ``<type>`` is one of: ``ingest``, ``query``, ``lint``, ``merge``,
``deprecate``, ``human-edit``, ``migration``. The ``[<stats>]`` suffix is
optional and human-readable only.

Locking: ``GitOps`` does NOT acquire ``FileLock``. Callers that may race
with other writers must already hold the project lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError


# Prefixes that identify a wikiloom-generated commit. Any commit whose
# subject prefix is NOT in this set is treated as human-authored — this
# catches both explicit ``human-edit:`` commits and plain-message
# commits made outside the wikiloom CLI (e.g. a quick edit in the user's
# editor followed by ``git commit -m "fix typo"``).
AUTO_COMMIT_TYPES: frozenset[str] = frozenset(
    {"ingest", "query", "lint", "merge", "deprecate", "migration"}
)


@dataclass(frozen=True)
class CommitInfo:
    """Summary of a single commit affecting a file."""

    hash: str
    message: str
    author: str
    timestamp: str
    commit_type: str  # parsed from subject prefix; "unknown" if non-conforming


def _parse_commit_type(message: str) -> str:
    subject = message.splitlines()[0] if message else ""
    prefix, sep, _ = subject.partition(":")
    if not sep:
        return "unknown"
    return prefix.strip()


def _count(value: object) -> int:
    """Normalize a stats field that may be a list or int into a count."""
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _format_ingest_stats(stats: dict) -> str:
    """Render an ingest stats dict as ``[+N pages, ~M pages, K links]``.

    Accepts either int counts or lists (e.g. ``pages_created=["a", "b"]``)
    so callers can pass a ``WikiEvent`` dict directly. Missing/zero keys
    are omitted; returns an empty string when nothing to show.
    """
    created = _count(stats.get("pages_created"))
    updated = _count(stats.get("pages_updated"))
    links = _count(stats.get("links_inserted"))

    parts: list[str] = []
    if created:
        parts.append(f"+{created} page{'s' if created != 1 else ''}")
    if updated:
        parts.append(f"~{updated} page{'s' if updated != 1 else ''}")
    if links:
        parts.append(f"{links} link{'s' if links != 1 else ''}")
    return f" [{', '.join(parts)}]" if parts else ""


class GitOps:
    """Commit orchestration for a WikiLoom project repository."""

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).resolve()
        try:
            self.repo = git.Repo(self.repo_path)
        except (InvalidGitRepositoryError, NoSuchPathError) as e:
            raise ValueError(f"Not a git repository: {self.repo_path}") from e

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rel(self, p: Path | str) -> str:
        path = Path(p)
        if path.is_absolute():
            path = path.resolve().relative_to(self.repo_path)
        return path.as_posix()

    def _stage_and_commit(self, files: Iterable[Path], message: str) -> str:
        """Stage ``files`` and create one commit.

        If nothing is staged (no files, or staged content is identical to
        HEAD), returns the current HEAD hash instead of creating an empty
        commit. Returns an empty string only for a fresh repo with no HEAD
        and nothing to commit.
        """
        rel_paths = [self._rel(f) for f in files]

        if rel_paths:
            self.repo.index.add(rel_paths)

        head_valid = self.repo.head.is_valid()
        if head_valid:
            if not self.repo.index.diff(self.repo.head.commit):
                return self.repo.head.commit.hexsha
        elif not rel_paths:
            return ""

        commit = self.repo.index.commit(message)
        return commit.hexsha

    # ------------------------------------------------------------------
    # Commit helpers
    # ------------------------------------------------------------------

    def commit_ingest(
        self,
        source_name: str,
        files: list[Path],
        stats: dict,
    ) -> str:
        """Commit the result of an ingest operation.

        Subject: ``ingest: <source_name> [+N pages, ~M pages, K links]``.
        ``stats`` may hold either int counts or lists for
        ``pages_created`` / ``pages_updated`` / ``links_inserted``.
        """
        message = f"ingest: {source_name}{_format_ingest_stats(stats)}"
        return self._stage_and_commit(files, message)

    def commit_human_edit(self, file_path: Path) -> str:
        """Commit a single human edit.

        Subject: ``human-edit: <relative-path> [protected]``. The
        ``human-edit:`` prefix is the signal ``is_human_edited`` looks for.
        """
        rel = self._rel(file_path)
        message = f"human-edit: {rel} [protected]"
        return self._stage_and_commit([Path(rel)], message)

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def get_changed_files_since(self, commit_hash: str) -> list[Path]:
        """Return absolute paths changed between ``commit_hash`` and HEAD."""
        try:
            base = self.repo.commit(commit_hash)
        except Exception as e:
            raise ValueError(f"Unknown commit: {commit_hash}") from e

        if not self.repo.head.is_valid():
            return []

        diff = base.diff(self.repo.head.commit)
        changed: set[Path] = set()
        for d in diff:
            for p in (d.a_path, d.b_path):
                if p:
                    changed.add(self.repo_path / p)
        return sorted(changed)

    def get_file_history(self, file_path: Path) -> list[CommitInfo]:
        """Return commits touching ``file_path``, newest first."""
        rel = self._rel(file_path)
        if not self.repo.head.is_valid():
            return []

        history: list[CommitInfo] = []
        for c in self.repo.iter_commits(paths=rel):
            msg = c.message if isinstance(c.message, str) else c.message.decode("utf-8")
            history.append(
                CommitInfo(
                    hash=c.hexsha,
                    message=msg.strip(),
                    author=c.author.name or "",
                    timestamp=c.committed_datetime.isoformat(),
                    commit_type=_parse_commit_type(msg),
                )
            )
        return history

    def is_human_edited(self, file_path: Path) -> bool:
        """True if the most recent commit touching ``file_path`` is human-authored.

        Git is the source of truth: we classify by the subject prefix
        rather than a frontmatter flag or author name, because LLM commits
        share the local git identity with human commits. Any commit whose
        prefix is not in ``AUTO_COMMIT_TYPES`` — including explicit
        ``human-edit:`` commits *and* plain-message commits made outside
        the wikiloom CLI — is treated as human.
        """
        info = self.latest_commit_type(file_path)
        if info is None:
            return False
        return info not in AUTO_COMMIT_TYPES

    def latest_commit_type(self, file_path: Path) -> str | None:
        """Return the parsed commit-type prefix for the most recent commit
        touching ``file_path``, or ``None`` if the file has no history.
        """
        rel = self._rel(file_path)
        if not self.repo.head.is_valid():
            return None
        for c in self.repo.iter_commits(paths=rel, max_count=1):
            msg = c.message if isinstance(c.message, str) else c.message.decode("utf-8")
            return _parse_commit_type(msg)
        return None
