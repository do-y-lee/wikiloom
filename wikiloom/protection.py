"""Human Edit Protection.

Keeps human-edit flags in sync with git history and manages the
wikiloom:auto marker that separates human writing from LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wikiloom.events import EventType, append_event, create_event
from wikiloom.frontmatter import (
    Frontmatter,
    parse_frontmatter,
    render_frontmatter,
)
from wikiloom.git_ops import AUTO_COMMIT_TYPES, GitOps
from wikiloom.registry import Registry
from wikiloom.utils import page_id_from_path


AUTO_MARKER = "<!-- wikiloom:auto -->"


@dataclass(frozen=True)
class DriftedPage:
    """A page whose human-edit flag disagrees with git."""

    page_id: str
    manifest_says: bool      # what the manifest currently claims
    git_says: bool           # what git log says (authoritative)
    last_commit_type: str    # parsed prefix from the most recent commit


class HumanEditProtection:
    """Keeps human-edit flags in sync with git and manages the auto-marker."""

    def __init__(
        self,
        project_root: Path,
        git: GitOps | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.wiki_dir = self.project_root / "wiki"
        self.registry_dir = self.project_root / "_registry"
        self.registry = registry or Registry(self.registry_dir, self.wiki_dir)
        self.git = git or GitOps(self.project_root)

    # ------------------------------------------------------------------
    # Flag sync
    # ------------------------------------------------------------------

    def scan(self) -> list[DriftedPage]:
        """Return pages whose manifest flag disagrees with git.

        A page is drifted if git_ops says it's human-edited but the
        manifest says it isn't (or vice versa). Pages with no git
        history are treated as un-edited.
        """
        drifted: list[DriftedPage] = []
        for page_id, entry in self.registry.pages.items():
            page_path = self._page_path(page_id)
            if page_path is None:
                continue
            last_type = self.git.latest_commit_type(page_path)
            if last_type is None:
                continue
            git_says = last_type not in AUTO_COMMIT_TYPES
            if entry.human_edited != git_says:
                drifted.append(
                    DriftedPage(
                        page_id=page_id,
                        manifest_says=entry.human_edited,
                        git_says=git_says,
                        last_commit_type=last_type,
                    )
                )
        return drifted

    def sync(self) -> list[DriftedPage]:
        """Apply git-truth to manifest + frontmatter for all drifted pages.

        Emits a single ``HUMAN_EDIT`` event summarizing the reclassification
        if any pages changed. Returns the list of pages whose flags were
        updated so callers can render a report.
        """
        drifted = self.scan()
        if not drifted:
            return drifted

        manifest_changed = False
        reclassified_ids: list[str] = []
        for page in drifted:
            page_path = self._page_path(page.page_id)
            if page_path is None:
                continue

            entry = self.registry.get_page(page.page_id)
            if entry is None:
                continue

            entry.human_edited = page.git_says
            manifest_changed = True
            self._update_frontmatter_flag(page_path, page.git_says)
            reclassified_ids.append(page.page_id)

        if manifest_changed:
            self.registry.save()

        if reclassified_ids:
            self._emit_sync_event(reclassified_ids)

        return drifted

    def is_protected(self, page_id: str) -> bool:
        """True if the page should be treated as human-authored.

        Delegates to ``git_ops.is_human_edited`` — git is the source of
        truth. Safe to call during an ingest run without syncing first.
        """
        page_path = self._page_path(page_id)
        if page_path is None:
            return False
        return self.git.is_human_edited(page_path)

    # ------------------------------------------------------------------
    # Content mechanism
    # ------------------------------------------------------------------

    @staticmethod
    def split(body: str) -> tuple[str, str]:
        """Split a page body into ``(human, auto)`` regions by the marker.

        If the marker is absent, the entire body is considered human and
        the auto region is empty. This is the conservative default: the
        LLM writer will append its output *after* a new marker, so an
        un-markered page naturally becomes "all human, no auto yet".
        """
        if AUTO_MARKER not in body:
            return body, ""
        human, _, auto = body.partition(AUTO_MARKER)
        return human, auto

    @staticmethod
    def merge(human: str, auto: str) -> str:
        """Join a human region and an auto region with the marker between.

        Ensures exactly one blank line separates each region from the
        marker so the rendered markdown stays readable.
        """
        human = human.rstrip() + "\n"
        auto = auto.lstrip("\n")
        return f"{human}\n{AUTO_MARKER}\n\n{auto}" if auto else f"{human}\n{AUTO_MARKER}\n"

    def preserve_human(self, page_path: Path, new_auto_body: str) -> str:
        """Return the body the writer should persist for a page update.

        Three cases:

        1. The page doesn't exist → return ``new_auto_body`` unchanged;
           the writer is creating the page fresh.
        2. The page exists but has no marker → treat the whole body as
           human and append the new auto content below a fresh marker.
        3. The page exists with a marker → keep the human region, swap
           the auto region for ``new_auto_body``.
        """
        if not page_path.exists():
            return new_auto_body

        text = page_path.read_text(encoding="utf-8")
        _, existing_body = parse_frontmatter(text)
        human, _ = self.split(existing_body)
        return self.merge(human, new_auto_body)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _page_path(self, page_id: str) -> Path | None:
        candidate = self.wiki_dir / f"{page_id}.md"
        return candidate if candidate.exists() else None

    def _update_frontmatter_flag(self, page_path: Path, human_edited: bool) -> None:
        """Write ``human_edited`` + ``human_edited_at`` into the page's frontmatter."""
        text = page_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if fm is None:
            return  # no frontmatter to update; lint's check_frontmatter will catch it

        fm.human_edited = human_edited
        if human_edited and not fm.human_edited_at:
            # Use the actual commit timestamp so the flag reflects when
            # the edit landed in git, not when sync happened to run.
            history = self.git.get_file_history(page_path)
            if history:
                fm.human_edited_at = history[0].timestamp
        if not human_edited:
            fm.human_edited_at = None
        page_path.write_text(
            render_frontmatter(fm) + "\n" + body, encoding="utf-8"
        )

    def _emit_sync_event(self, page_ids: list[str]) -> None:
        log_path = self.wiki_dir / "log.md"
        if not log_path.parent.exists():
            return
        event = create_event(
            EventType.HUMAN_EDIT,
            description=f"{len(page_ids)} page(s) reclassified from git history",
            pages_updated=page_ids,
        )
        append_event(log_path, event)
