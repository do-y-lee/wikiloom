"""Deterministic linking engine.

Runs after the LLM writes prose. Takes plain markdown and inserts
validated ``[[wikilinks]]`` using spaCy NER + rapidfuzz fuzzy matching
against the manifest. No LLM in the loop.

The engine uses **tiered confidence**:

    high   (>= 95)   auto-link, no review
    medium (>= 85)   auto-link, flagged in backlinks as confidence: medium
    low    (>= 70)   defer to _registry/pending.json for batch review
    < 70             ignored

Domain-specific recognition is bootstrapped from the manifest itself:
every existing page's title and aliases become spaCy ``EntityRuler``
patterns, so "flash attention" or "LoRA" are caught even though
``en_core_web_sm`` doesn't know about them out of the box.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz, process

from wikiloom.config import LinkingConfig
from wikiloom.frontmatter import Frontmatter, parse_frontmatter, render_frontmatter
from wikiloom.registry import PageEntry, Registry
from wikiloom.utils import now_iso, page_id_from_path, slugify

if TYPE_CHECKING:
    import spacy.language


# Common English words that should never be auto-linked even if a page
# happens to share their title. Conservative — extend as needed.
STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "data", "model", "system", "method", "approach", "result", "results",
    "paper", "study", "work", "use", "used", "using", "way", "case",
    "type", "value", "values", "time", "thing", "things", "part", "parts",
    "example", "note", "section", "figure", "table", "chapter",
    "this", "that", "these", "those", "some", "any", "all",
    "also", "such", "more", "less", "most", "least",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
})


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass
class MatchResult:
    page_id: str
    score: int
    method: str  # "exact" | "fuzzy"


@dataclass
class ResolvedLink:
    original_text: str
    page_id: str
    score: int
    confidence: str  # "high" | "medium"
    start: int
    end: int


@dataclass
class PendingLink:
    source_page: str
    matched_text: str
    candidate_page_id: str
    score: int
    label: str


@dataclass
class UnresolvedEntity:
    text: str
    label: str


@dataclass
class LinkingResult:
    page: Path
    high_confidence_links: int = 0
    medium_confidence_links: int = 0
    pending_review: int = 0
    stubs_created: int = 0
    unresolved: list[UnresolvedEntity] = field(default_factory=list)


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


class LinkingEngine:
    """spaCy + rapidfuzz wikilink inserter."""

    def __init__(
        self,
        registry: Registry,
        config: LinkingConfig | None = None,
        nlp: "spacy.language.Language | None" = None,
    ) -> None:
        import spacy  # imported lazily so tests that don't need NER stay fast

        self.registry = registry
        self.config = config or LinkingConfig()
        self.high_threshold = self.config.high_confidence_threshold
        self.medium_threshold = self.config.medium_confidence_threshold
        self.low_threshold = self.config.low_confidence_threshold

        if nlp is None:
            self.nlp = spacy.load(self.config.ner_model)
        else:
            self.nlp = nlp

        self.alias_map: dict[str, str] = registry.get_all_aliases()
        self._build_entity_ruler()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_entity_ruler(self) -> None:
        """Auto-generate spaCy entity patterns from manifest pages and aliases."""
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns: list[dict[str, Any]] = []
        for entry in self.registry.pages.values():
            if entry.status != "active":
                continue
            label = "ORG" if entry.type == "entity" else "CONCEPT"
            patterns.append({"label": label, "pattern": entry.title})
            for alias in entry.aliases:
                patterns.append({"label": label, "pattern": alias})
        if patterns:
            ruler.add_patterns(patterns)

    def refresh(self) -> None:
        """Rebuild the alias map and entity ruler from the registry.

        Call this after registering new pages mid-pipeline (e.g. after
        creating stubs) so subsequent ``link_page`` calls see them.
        """
        self.alias_map = self.registry.get_all_aliases()
        self._build_entity_ruler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link_page(self, page_path: Path) -> LinkingResult:
        """Run linking on a single wiki page and write back the result."""
        page_path = Path(page_path)
        text = page_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)

        result = self._link_text(
            body,
            source_page_id=page_id_from_path(self.registry.wiki_dir, page_path),
        )

        if fm is not None:
            # Preserve frontmatter when writing back
            page_path.write_text(render_frontmatter(fm) + "\n" + result.body, encoding="utf-8")
        else:
            page_path.write_text(result.body, encoding="utf-8")

        # Persist pending links
        if result.pending:
            self._save_pending(result.pending)

        # Optional stub creation
        stubs_created = 0
        if result.unresolved and self.config.auto_create_stubs:
            stubs_created = self._create_stubs(result.unresolved)
            if stubs_created:
                self.refresh()

        return LinkingResult(
            page=page_path,
            high_confidence_links=len(result.high_links),
            medium_confidence_links=len(result.medium_links),
            pending_review=len(result.pending),
            stubs_created=stubs_created,
            unresolved=result.unresolved,
        )

    def link_all(self, pages: list[Path]) -> list[Path]:
        """Run linking across multiple pages.

        Returns the list of pages that actually got at least one link.
        """
        modified: list[Path] = []
        for page in pages:
            r = self.link_page(page)
            if r.high_confidence_links + r.medium_confidence_links > 0:
                modified.append(r.page)
        return modified

    # ------------------------------------------------------------------
    # Core link-text pipeline (tested in isolation)
    # ------------------------------------------------------------------

    @dataclass
    class _LinkTextResult:
        body: str
        high_links: list[ResolvedLink]
        medium_links: list[ResolvedLink]
        pending: list[PendingLink]
        unresolved: list[UnresolvedEntity]

    def _link_text(self, body: str, source_page_id: str) -> _LinkTextResult:
        """Apply linking to a body string. Returns the modified body and stats.

        Pulled out so it can be tested without writing to disk.
        """
        safe_zones = self._compute_safe_zones(body)
        doc = self.nlp(body)

        candidates: list[tuple[str, str, int, int]] = [
            (ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
        ]
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:
                candidates.append(
                    (chunk.text, "CONCEPT", chunk.start_char, chunk.end_char)
                )

        candidates = self._dedupe_candidates(candidates)
        candidates = [c for c in candidates if self._in_safe_zone(c, safe_zones)]
        candidates = [c for c in candidates if c[0].lower().strip() not in STOP_WORDS]

        high_links: list[ResolvedLink] = []
        medium_links: list[ResolvedLink] = []
        pending: list[PendingLink] = []
        linked_targets: set[str] = set()
        resolved_or_pending_texts: set[str] = set()

        for text, label, start, end in candidates:
            match = self._resolve(text)
            if not match:
                continue
            if match.page_id == source_page_id:
                # Don't link, but mark as handled so a later surface-form
                # variant (e.g. plural) doesn't get a stub created for it.
                resolved_or_pending_texts.add(text.lower())
                continue
            if match.page_id in linked_targets:
                # Already linked once on this page — skip subsequent
                # mentions, but mark them handled so variant surface forms
                # ("Flash Attention" then "Flash Attentions") don't end up
                # in `unresolved` and trigger duplicate stub creation.
                resolved_or_pending_texts.add(text.lower())
                continue

            if match.score >= self.high_threshold:
                high_links.append(
                    ResolvedLink(
                        original_text=text,
                        page_id=match.page_id,
                        score=match.score,
                        confidence="high",
                        start=start,
                        end=end,
                    )
                )
                linked_targets.add(match.page_id)
                resolved_or_pending_texts.add(text.lower())
            elif match.score >= self.medium_threshold:
                medium_links.append(
                    ResolvedLink(
                        original_text=text,
                        page_id=match.page_id,
                        score=match.score,
                        confidence="medium",
                        start=start,
                        end=end,
                    )
                )
                linked_targets.add(match.page_id)
                resolved_or_pending_texts.add(text.lower())
            elif match.score >= self.low_threshold:
                pending.append(
                    PendingLink(
                        source_page=source_page_id,
                        matched_text=text,
                        candidate_page_id=match.page_id,
                        score=match.score,
                        label=label,
                    )
                )
                resolved_or_pending_texts.add(text.lower())

        unresolved = [
            UnresolvedEntity(text=text, label=label)
            for text, label, _, _ in candidates
            if text.lower() not in resolved_or_pending_texts
        ]

        new_body = self._insert_links(body, high_links + medium_links)

        return self._LinkTextResult(
            body=new_body,
            high_links=high_links,
            medium_links=medium_links,
            pending=pending,
            unresolved=unresolved,
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve(self, text: str) -> MatchResult | None:
        """Resolve entity text to a manifest page.

        Step 1: exact match against alias map.
        Step 2: rapidfuzz token_sort_ratio against all aliases.
        """
        normalized = text.lower().strip()
        if not normalized:
            return None
        if normalized in self.alias_map:
            return MatchResult(
                page_id=self.alias_map[normalized], score=100, method="exact"
            )
        if not self.alias_map:
            return None
        best = process.extractOne(
            normalized,
            list(self.alias_map.keys()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.low_threshold,
        )
        if best is None:
            return None
        alias, score, _ = best
        return MatchResult(
            page_id=self.alias_map[alias], score=int(score), method="fuzzy"
        )

    # ------------------------------------------------------------------
    # Safe zones
    # ------------------------------------------------------------------

    _FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
    _TILDE_FENCE_RE = re.compile(r"~~~.*?~~~", re.DOTALL)
    _INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
    _WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
    _MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]+\)")
    _MD_REF_LINK_RE = re.compile(r"\[[^\]]*\]\[[^\]]*\]")
    _MD_REF_DEF_RE = re.compile(r"^\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
    _URL_RE = re.compile(r"https?://\S+")
    _HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
    _HTML_PRE_CODE_RE = re.compile(
        r"<(pre|code|script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
    )
    _HEADING_RE = re.compile(r"^#{1,6}.*$", re.MULTILINE)

    def _compute_safe_zones(self, body: str) -> list[tuple[int, int]]:
        """Compute ranges of `body` where wikilinks can be inserted.

        Returns a list of (start, end) character ranges. Excludes:
            - fenced code blocks
            - inline code spans
            - existing wikilinks
            - markdown link syntax
            - bare URLs
            - HTML comments (provenance markers)
            - headings (avoid linking inside titles)
        """
        unsafe: list[tuple[int, int]] = []
        for pattern in (
            self._FENCED_CODE_RE,
            self._TILDE_FENCE_RE,
            self._INLINE_CODE_RE,
            self._WIKILINK_RE,
            self._MD_LINK_RE,
            self._MD_REF_LINK_RE,
            self._MD_REF_DEF_RE,
            self._URL_RE,
            self._HTML_COMMENT_RE,
            self._HTML_PRE_CODE_RE,
            self._HEADING_RE,
        ):
            for m in pattern.finditer(body):
                unsafe.append((m.start(), m.end()))

        if not unsafe:
            return [(0, len(body))]

        unsafe.sort()
        merged: list[tuple[int, int]] = []
        for start, end in unsafe:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        safe: list[tuple[int, int]] = []
        cursor = 0
        for start, end in merged:
            if cursor < start:
                safe.append((cursor, start))
            cursor = end
        if cursor < len(body):
            safe.append((cursor, len(body)))
        return safe

    def _in_safe_zone(
        self, candidate: tuple[str, str, int, int], safe_zones: list[tuple[int, int]]
    ) -> bool:
        _, _, start, end = candidate
        return any(sz_start <= start and end <= sz_end for sz_start, sz_end in safe_zones)

    # ------------------------------------------------------------------
    # Link insertion
    # ------------------------------------------------------------------

    def _insert_links(self, body: str, links: list[ResolvedLink]) -> str:
        """Insert ``[[page_id|original_text]]`` from end to start.

        Working from the back avoids invalidating earlier offsets.
        """
        if not links:
            return body
        for link in sorted(links, key=lambda l: l.start, reverse=True):
            wikilink = f"[[{link.page_id}|{link.original_text}]]"
            body = body[: link.start] + wikilink + body[link.end :]
        return body

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_candidates(
        candidates: list[tuple[str, str, int, int]],
    ) -> list[tuple[str, str, int, int]]:
        """Deduplicate candidates by character span, keeping NER over chunks."""
        seen: dict[tuple[int, int], tuple[str, str, int, int]] = {}
        for text, label, start, end in candidates:
            key = (start, end)
            existing = seen.get(key)
            if existing is None:
                seen[key] = (text, label, start, end)
            elif existing[1] == "CONCEPT" and label != "CONCEPT":
                # Prefer non-CONCEPT (i.e. a real NER hit) over a chunk overlap
                seen[key] = (text, label, start, end)
        return sorted(seen.values(), key=lambda c: c[2])

    # ------------------------------------------------------------------
    # Pending link persistence
    # ------------------------------------------------------------------

    def _save_pending(self, pending: list[PendingLink]) -> None:
        """Append pending links to ``_registry/pending.json``."""
        path = self.registry.registry_dir / "pending.json"

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {"version": 1, "pending": []}
        else:
            data = {"version": 1, "pending": []}

        # Tolerate both a bare list and a {"pending": [...]} object so the
        # file produced by `wikiloom init` round-trips cleanly.
        if isinstance(data, list):
            existing = data
            data = {"version": 1, "pending": existing}
        existing_items = data.setdefault("pending", [])

        timestamp = now_iso()
        for p in pending:
            existing_items.append(
                {
                    "source_page": p.source_page,
                    "matched_text": p.matched_text,
                    "candidate_page_id": p.candidate_page_id,
                    "score": p.score,
                    "label": p.label,
                    "added_at": timestamp,
                }
            )

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Stub creation
    # ------------------------------------------------------------------

    def _create_stubs(self, unresolved: list[UnresolvedEntity]) -> int:
        """Create minimal stub pages for unresolved entities.

        Emits a single STUB_CREATED event listing every stub created in
        this batch (one event per call, not per stub).
        """
        wiki_dir = self.registry.wiki_dir
        if not wiki_dir.exists():
            return 0

        created = 0
        created_page_ids: list[str] = []
        seen_slugs: set[str] = set()
        for entity in unresolved:
            slug = slugify(entity.text)
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            page_type = (
                "entity"
                if entity.label in ("PERSON", "ORG", "GPE", "PRODUCT")
                else "concept"
            )
            category = "entities" if page_type == "entity" else "concepts"
            page_id = f"{category}/{slug}"

            if self.registry.get_page(page_id):
                continue

            page_path = wiki_dir / category / f"{slug}.md"
            page_path.parent.mkdir(parents=True, exist_ok=True)

            fm = Frontmatter(
                title=entity.text,
                type=page_type,
                status="stub",
                created=now_iso(),
                modified=now_iso(),
                summary=f"Stub page for {entity.text}.",
                aliases=[entity.text.lower()],
                confidence="low",
            )
            body = f"# {entity.text}\n\n*Stub — awaiting content.*\n"
            page_path.write_text(
                render_frontmatter(fm) + "\n" + body, encoding="utf-8"
            )

            entry = PageEntry(
                title=entity.text,
                type=page_type,
                status="stub",
                aliases=[entity.text.lower()],
                summary=f"Stub page for {entity.text}.",
                confidence="low",
            )
            self.registry.register_page(page_id, entry)
            created_page_ids.append(page_id)
            created += 1

        if created_page_ids:
            # Persist the manifest so lint + subsequent runs see the new
            # stubs on disk. The linker is the only mutator between ingest
            # and lint, so saving here keeps the invariant without
            # requiring every caller to remember.
            self.registry.save()
            self._emit_stub_created_event(created_page_ids)

        return created

    def _emit_stub_created_event(self, page_ids: list[str]) -> None:
        """Append a STUB_CREATED entry to wiki/log.md.

        Imported lazily to avoid pulling events.py into the linker's
        public import surface.
        """
        from wikiloom.events import EventType, append_event, create_event

        log_path = self.registry.wiki_dir / "log.md"
        if not log_path.parent.exists():
            return

        event = create_event(
            EventType.STUB_CREATED,
            description=f"{len(page_ids)} stub(s)",
            pages_created=page_ids,
        )
        append_event(log_path, event)
