"""Hybrid linking engine.

For each span extracted by spaCy NER + noun chunks, matches it to
an existing page in two stages:

1. Fuzzy pre-filter — shortlists up to ``fuzzy_prefilter_top_k``
   candidate pages by rapidfuzz string similarity against titles
   and aliases.
2. Cosine rerank — embeds the span's context window and picks the
   candidate whose page-body embedding has the highest cosine.

Cosine thresholds (auto / medium / pending / drop) decide what
happens to the match. An embedder and a SQLite cache with
populated page embeddings are required — the linker has no pure-
fuzzy fallback path by design.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rapidfuzz import fuzz, process

from wikiloom.config import LinkingConfig
from wikiloom.frontmatter import Frontmatter, parse_frontmatter, render_frontmatter
from wikiloom.registry import PageEntry, Registry
from wikiloom.utils import now_iso, page_id_from_path, slugify

if TYPE_CHECKING:
    import spacy.language

    from wikiloom.cache import SQLiteCache
    from wikiloom.embeddings import Embedder


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
    score: int  # 0-100 fuzzy score (or 100 for exact alias hits)
    method: str  # "exact" | "fuzzy" | "hybrid"
    cosine_score: float | None = None  # populated only on the hybrid path


@dataclass
class ResolvedLink:
    original_text: str
    page_id: str
    score: int  # fuzzy score
    confidence: str  # "high" | "medium"
    start: int
    end: int
    cosine_score: float | None = None


@dataclass
class PendingLink:
    source_page: str
    matched_text: str
    candidate_page_id: str
    score: int  # fuzzy score
    label: str
    cosine_score: float | None = None


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
        *,
        embedder: "Embedder",
        cache: "SQLiteCache",
    ) -> None:
        import spacy  # imported lazily so tests that don't need NER stay fast

        self.registry = registry
        self.config = config or LinkingConfig()

        if nlp is None:
            self.nlp = spacy.load(self.config.ner_model)
        else:
            self.nlp = nlp

        self.alias_map: dict[str, str] = registry.get_all_aliases()
        # Cached keys list for rapidfuzz lookups.
        self._alias_keys: list[str] = list(self.alias_map.keys())
        self._build_entity_ruler()

        # Hybrid is the only path; both dependencies are required. Call
        # sites that might be invoked without embeddings (disabled in
        # config, failed backend load) must guard themselves before
        # constructing the engine.
        self.embedder = embedder
        self.cache = cache
        self._page_embeddings: dict[str, list[float]] | None = None

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
        self._alias_keys = list(self.alias_map.keys())
        self._build_entity_ruler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _should_skip(fm: Frontmatter | None) -> bool:
        """Whether a page is exempt from the linker entirely.

        ``status == "stub"`` pages have placeholder bodies (the
        boilerplate ``*Stub — awaiting content.*``); running NER on
        them produces no useful candidates and only burns embedder
        time. Source pages are intentionally *not* skipped — their
        summaries carry real concept references that benefit from
        being wikilinked.
        """
        return fm is not None and fm.status == "stub"

    def link_page(
        self,
        page_path: Path,
        *,
        doc: "spacy.tokens.Doc | None" = None,
    ) -> LinkingResult:
        """Run linking on a single wiki page and write back the result.

        ``doc`` is an optional pre-computed spaCy ``Doc``. ``link_all``
        populates it via one batched ``nlp.pipe`` call so this method
        doesn't have to invoke spaCy per page. When omitted, the doc
        is computed inline (preserves the original code path for
        callers that don't pre-batch).
        """
        page_path = Path(page_path)
        text = page_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)

        if self._should_skip(fm):
            return LinkingResult(page=page_path)

        result = self._link_text(
            body,
            source_page_id=page_id_from_path(
                self.registry.wiki_dir, page_path),
            doc=doc,
        )

        if fm is not None:
            # Preserve frontmatter when writing back
            page_path.write_text(render_frontmatter(
                fm) + "\n" + result.body, encoding="utf-8")
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

    def link_all(
        self,
        pages: list[Path],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """Run linking across multiple pages.

        ``progress`` is optional and, when provided, is called as
        ``progress(completed, total)`` after each page so callers
        (typically the CLI) can emit incremental progress on large
        wikis without every call site having to drive the loop itself.

        Returns the list of pages that actually got at least one link.

        Pre-reads the bodies and runs spaCy in one batched ``nlp.pipe``
        call before the per-page loop. This amortizes spaCy's
        pipeline overhead (entity ruler setup, tokenizer state) across
        the whole batch and is meaningfully faster than per-page
        ``nlp(body)``. Stub and source pages are filtered out of the
        batch so they don't pay the NLP cost only to be discarded.
        """
        # Pre-pass: read bodies for batched NLP. Skip pages we'd drop
        # anyway so we don't spend NLP time on them.
        page_bodies: list[tuple[int, str]] = []
        for i, page in enumerate(pages):
            try:
                text = Path(page).read_text(encoding="utf-8")
            except OSError:
                continue
            fm, body = parse_frontmatter(text)
            if self._should_skip(fm):
                continue
            page_bodies.append((i, body))

        docs_by_index: dict[int, Any] = {}
        if page_bodies:
            for (i, _), doc in zip(
                page_bodies,
                self.nlp.pipe(b for _, b in page_bodies),
            ):
                docs_by_index[i] = doc

        modified: list[Path] = []
        total = len(pages)
        for i, page in enumerate(pages, start=1):
            r = self.link_page(page, doc=docs_by_index.get(i - 1))
            if r.high_confidence_links + r.medium_confidence_links > 0:
                modified.append(r.page)
            if progress is not None:
                progress(i, total)
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

    def _link_text(
        self,
        body: str,
        source_page_id: str,
        *,
        doc: "spacy.tokens.Doc | None" = None,
    ) -> _LinkTextResult:
        """Apply linking to a body string. Returns the modified body and stats.

        Pulled out so it can be tested without writing to disk.

        ``doc`` lets callers (notably ``link_all``) supply a spaCy
        ``Doc`` that was produced by a batched ``nlp.pipe`` call. If
        omitted, the doc is computed inline so direct callers
        (single-page tests, the legacy path) keep working.
        """
        safe_zones = self._compute_safe_zones(body)
        if doc is None:
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
        candidates = [
            c for c in candidates if self._in_safe_zone(c, safe_zones)]
        candidates = [c for c in candidates if c[0].lower().strip()
                      not in STOP_WORDS]

        # Batch-embed every candidate context window in one call rather
        # than N per-span calls inside ``_resolve_with_rerank``. Skips
        # candidates that will short-circuit on an exact alias hit
        # (no embedding needed). If the batch call raises or returns
        # the wrong shape, the resolver falls back to its per-span
        # path so behavior degrades gracefully.
        span_vec_by_index: dict[int, list[float]] = {}
        contexts: list[str] = []
        context_owner: list[int] = []
        for i, (text, _label, start, end) in enumerate(candidates):
            normalized = text.lower().strip()
            if not normalized or normalized in self.alias_map:
                continue
            contexts.append(self._context_window(body, start, end))
            context_owner.append(i)
        if contexts:
            try:
                batch_vectors = self.embedder.embed_texts(contexts)
            except Exception:  # noqa: BLE001 — fall back to per-span
                batch_vectors = []
            if batch_vectors and len(batch_vectors) == len(contexts):
                for ci, vec in zip(context_owner, batch_vectors):
                    span_vec_by_index[ci] = vec

        high_links: list[ResolvedLink] = []
        medium_links: list[ResolvedLink] = []
        pending: list[PendingLink] = []
        linked_targets: set[str] = set()
        resolved_or_pending_texts: set[str] = set()

        for i, (text, label, start, end) in enumerate(candidates):
            match = self._resolve_with_rerank(
                text, body, start, end,
                span_vec=span_vec_by_index.get(i),
            )
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

            bucket = self._bucket_for(match)
            if bucket == "high":
                high_links.append(
                    ResolvedLink(
                        original_text=text,
                        page_id=match.page_id,
                        score=match.score,
                        confidence="high",
                        start=start,
                        end=end,
                        cosine_score=match.cosine_score,
                    )
                )
                linked_targets.add(match.page_id)
                resolved_or_pending_texts.add(text.lower())
            elif bucket == "medium":
                medium_links.append(
                    ResolvedLink(
                        original_text=text,
                        page_id=match.page_id,
                        score=match.score,
                        confidence="medium",
                        start=start,
                        end=end,
                        cosine_score=match.cosine_score,
                    )
                )
                linked_targets.add(match.page_id)
                resolved_or_pending_texts.add(text.lower())
            elif bucket == "pending":
                pending.append(
                    PendingLink(
                        source_page=source_page_id,
                        matched_text=text,
                        candidate_page_id=match.page_id,
                        score=match.score,
                        label=label,
                        cosine_score=match.cosine_score,
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
        """Resolve entity text to a manifest page by exact or fuzzy match.

        Returns the best fuzzy match at or above the pre-filter cutoff;
        exact alias hits short-circuit with score 100. Used directly by
        tests; the production link path goes through
        ``_resolve_with_rerank``.
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
            self._alias_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.config.fuzzy_prefilter_threshold,
        )
        if best is None:
            return None
        alias, score, _ = best
        return MatchResult(
            page_id=self.alias_map[alias], score=int(score), method="fuzzy"
        )

    def _resolve_top_k(
        self, text: str, k: int, score_cutoff: int
    ) -> list[tuple[str, int]]:
        """Return up to ``k`` fuzzy candidates as ``(page_id, score)`` pairs.

        Used by the hybrid path as a cheap pre-filter before cosine
        rerank. Exact alias hits collapse to a single top-1 result so
        the cosine pass is skipped when the answer is already certain.
        Duplicates (multiple aliases pointing to the same page) are
        deduped by page_id, keeping the best alias score.
        """
        normalized = text.lower().strip()
        if not normalized or not self.alias_map:
            return []
        if normalized in self.alias_map:
            return [(self.alias_map[normalized], 100)]
        matches = process.extract(
            normalized,
            self._alias_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=score_cutoff,
            limit=k * 3,  # overshoot to survive the alias→page_id dedup
        )
        seen: dict[str, int] = {}
        for alias, score, _ in matches:
            page_id = self.alias_map[alias]
            if page_id not in seen or score > seen[page_id]:
                seen[page_id] = int(score)
        return sorted(seen.items(), key=lambda x: x[1], reverse=True)[:k]

    def _resolve_with_rerank(
        self,
        text: str,
        body: str,
        start: int,
        end: int,
        *,
        span_vec: list[float] | None = None,
    ) -> MatchResult | None:
        """Hybrid resolver: fuzzy pre-filter → cosine rerank.

        Steps:

        1. Exact alias hit short-circuits with ``method="exact"``.
        2. Fuzzy top-K against the alias map at a relaxed cutoff.
        3. Build a context window (~200 chars either side of the span)
           and embed it. The span alone is often too terse — "account"
           on its own cosine-matches every account-related page — so
           surrounding context gives the embedder real signal about
           which target fits this mention.
        4. Cosine-compare the context embedding against each
           candidate's pre-loaded page embedding; return the best.

        Returns ``None`` when we can't produce a cosine score —
        either no candidates, no embeddings on any candidate, or the
        embedder raised. In those cases the span is dropped rather
        than linked; there's no fuzzy-only fallback by design.

        ``span_vec`` is an optional pre-computed embedding for the
        context window. ``_link_text`` populates it via one batched
        ``embed_texts`` call so this resolver doesn't have to issue
        a per-span call for every candidate on the page. When
        omitted, the resolver embeds the context itself (preserves
        the original code path for callers that don't pre-batch).
        """
        from wikiloom.embeddings import cosine_similarity

        normalized = text.lower().strip()
        if not normalized:
            return None
        # Exact alias hit — no need to burn an embedding call.
        if normalized in self.alias_map:
            return MatchResult(
                page_id=self.alias_map[normalized], score=100, method="exact"
            )

        candidates = self._resolve_top_k(
            text,
            k=self.config.fuzzy_prefilter_top_k,
            score_cutoff=self.config.fuzzy_prefilter_threshold,
        )
        if not candidates:
            return None

        self._ensure_page_embeddings_loaded()
        embeddings = self._page_embeddings or {}
        usable = [(pid, fscore)
                  for pid, fscore in candidates if pid in embeddings]
        if not usable:
            return None

        if span_vec is None:
            context = self._context_window(body, start, end)
            try:
                span_vectors = self.embedder.embed_texts([context])
            except Exception:  # noqa: BLE001 — degrade rather than crash a link pass
                return None
            if not span_vectors:
                return None
            span_vec = span_vectors[0]

        best_page_id: str | None = None
        best_fuzzy = 0
        best_cosine = -1.0
        for page_id, fuzzy_score in usable:
            cosine = cosine_similarity(span_vec, embeddings[page_id])
            if cosine > best_cosine:
                best_cosine = cosine
                best_fuzzy = fuzzy_score
                best_page_id = page_id

        if best_page_id is None:
            return None
        return MatchResult(
            page_id=best_page_id,
            score=best_fuzzy,
            method="hybrid",
            cosine_score=best_cosine,
        )

    def _bucket_for(self, match: MatchResult) -> str:
        """Map a MatchResult to ``"high" | "medium" | "pending" | "drop"``.

        Cosine is the decision metric; fuzzy score is a breadcrumb.
        Exact alias hits always land in ``"high"`` — they're
        unambiguous and skip the cosine step.
        """
        if match.method == "exact":
            return "high"
        c = match.cosine_score
        if c is None:
            # _resolve_with_rerank only returns a MatchResult when it
            # successfully produced a cosine score. Anything else is a
            # programming error, not a runtime case.
            return "drop"
        if c >= self.config.cosine_high_threshold:
            return "high"
        if c >= self.config.cosine_medium_threshold:
            return "medium"
        if c >= self.config.cosine_low_threshold:
            return "pending"
        return "drop"

    def _ensure_page_embeddings_loaded(self) -> None:
        if self._page_embeddings is not None or self.cache is None:
            return
        try:
            self._page_embeddings = self.cache.load_page_embeddings()
        except Exception:  # noqa: BLE001 — missing cache shouldn't kill linking
            self._page_embeddings = {}

    @staticmethod
    def _context_window(body: str, start: int, end: int, radius: int = 200) -> str:
        """Return ``body[start-radius : end+radius]`` clamped to bounds."""
        left = max(0, start - radius)
        right = min(len(body), end + radius)
        return body[left:right]

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
            body = body[: link.start] + wikilink + body[link.end:]
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
            # Cast both scores to plain Python types: cosine comes from
            # numpy ops (cosine_similarity → float32), and round() on a
            # numpy scalar still returns a numpy scalar — json.dumps
            # rejects those with a TypeError. Fuzzy score is usually an
            # int already but cast defensively.
            entry: dict[str, Any] = {
                "source_page": p.source_page,
                "matched_text": p.matched_text,
                "candidate_page_id": p.candidate_page_id,
                "score": int(p.score) if p.score is not None else None,
                "label": p.label,
                "added_at": timestamp,
            }
            if p.cosine_score is not None:
                entry["cosine_score"] = round(float(p.cosine_score), 4)
            existing_items.append(entry)

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
