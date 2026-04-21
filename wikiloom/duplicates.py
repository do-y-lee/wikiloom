"""Near-duplicate page detection.

Combines slug fuzzy matching and embedding cosine similarity to
surface pages that may have been created as near-duplicates by the
LLM synthesis loop. Used by ``wikiloom duplicates`` and the
``check_near_duplicates`` lint check.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from wikiloom.embeddings import cosine_similarity, deserialize_embedding


@dataclass(frozen=True)
class DuplicatePair:
    """Two pages flagged as suspected duplicates."""

    page_a: str
    page_b: str
    title_a: str
    title_b: str
    type_a: str
    type_b: str
    slug_score: float       # 0-100 from rapidfuzz token_sort_ratio
    embedding_score: float  # 0.0-1.0 cosine; -1.0 if either page lacks embedding
    inbound_a: int = 0      # cached inbound_links from wiki.db
    inbound_b: int = 0
    created_a: str = ""     # ISO timestamp from wiki.db
    created_b: str = ""

    @property
    def is_cross_type(self) -> bool:
        return self.type_a != self.type_b

    @property
    def combined_score(self) -> float:
        """Sort key. Prefers embedding signal when available, falls back to slug."""
        if self.embedding_score >= 0:
            # Weighted: embedding dominates (0.7), slug is a tiebreaker (0.3)
            return self.embedding_score * 100 * 0.7 + self.slug_score * 0.3
        return self.slug_score


@dataclass(frozen=True)
class WinnerSuggestion:
    """Heuristic suggestion for which page in a pair should be the winner."""

    winner_page_id: str
    loser_page_id: str
    reason: str             # human-readable rule that triggered
    is_safe_to_auto: bool   # True only when the pair fits a "near-identical slug" pattern


def _slug_part(page_id: str) -> str:
    """Strip the type prefix from a page_id (``concepts/foo`` → ``foo``)."""
    if "/" in page_id:
        return page_id.split("/", 1)[1]
    return page_id


def find_duplicates(
    project_root: Path,
    slug_threshold: float = 80.0,
    embedding_threshold: float = 0.85,
    same_type_only: bool = True,
) -> list[DuplicatePair]:
    """Return suspect duplicate page pairs sorted by combined score (desc).

    A pair is flagged if EITHER the slug fuzzy ratio meets
    ``slug_threshold`` OR the embedding cosine meets
    ``embedding_threshold``. When ``same_type_only`` is True (default),
    pairs across different types are skipped — the LLM rarely creates
    cross-type duplicates and including them tends to surface lookalike
    titles like a concept and an entity sharing a name.

    Reads from the SQLite cache at ``_registry/wiki.db``. Returns an
    empty list if the cache is missing or empty.
    """
    db_path = project_root / "_registry" / "wiki.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT page_id, title, type, inbound_links, created, embedding "
            "FROM pages WHERE status != 'deprecated'"
        ).fetchall()
    finally:
        conn.close()

    pages: list[dict] = []
    for row in rows:
        d = dict(row)
        blob = d.pop("embedding", None)
        d["vector"] = deserialize_embedding(blob) if blob else None
        pages.append(d)

    pairs: list[DuplicatePair] = []
    for i, a in enumerate(pages):
        for b in pages[i + 1:]:
            if same_type_only and a["type"] != b["type"]:
                continue
            slug_score = fuzz.token_sort_ratio(
                _slug_part(a["page_id"]), _slug_part(b["page_id"])
            )

            embedding_score = -1.0
            if a["vector"] is not None and b["vector"] is not None:
                embedding_score = cosine_similarity(a["vector"], b["vector"])

            slug_hit = slug_score >= slug_threshold
            embed_hit = embedding_score >= embedding_threshold
            if not (slug_hit or embed_hit):
                continue

            pairs.append(
                DuplicatePair(
                    page_a=a["page_id"],
                    page_b=b["page_id"],
                    title_a=a["title"],
                    title_b=b["title"],
                    type_a=a["type"],
                    type_b=b["type"],
                    slug_score=slug_score,
                    embedding_score=embedding_score,
                    inbound_a=a.get("inbound_links") or 0,
                    inbound_b=b.get("inbound_links") or 0,
                    created_a=a.get("created") or "",
                    created_b=b.get("created") or "",
                )
            )

    pairs.sort(key=lambda p: p.combined_score, reverse=True)
    return pairs


def _is_plural_of(short: str, long: str) -> bool:
    """True if ``long`` is just ``short`` with a trailing 's' (or 'es')."""
    if long == short + "s":
        return True
    if long == short + "es":
        return True
    return False


def _tokens(slug: str) -> list[str]:
    """Split a slug on hyphens into its word tokens."""
    return [t for t in slug.split("-") if t]


def _is_token_drop_variant(
    short_tokens: list[str], long_tokens: list[str]
) -> bool:
    """True if ``long`` has exactly one extra token inserted anywhere.

    In other words, ``short`` is a subsequence of ``long`` that
    skips exactly one element. Catches cases like
    ``zelle-transfer-limits`` vs ``zelle-daily-transfer-limits`` and
    ``transaction-reconciliation`` vs
    ``transaction-history-reconciliation`` — the shorter slug is the
    more general concept, the longer slug adds a qualifying word
    (``daily``, ``history``) that narrows it.
    """
    if len(long_tokens) != len(short_tokens) + 1:
        return False
    i = 0
    for t in long_tokens:
        if i < len(short_tokens) and short_tokens[i] == t:
            i += 1
    return i == len(short_tokens)


def _hyphen_collapsed(slug: str) -> str:
    """Slug with all hyphens removed, for matching hyphenation variants.

    ``jp-morgan-chase`` and ``j-p-morgan-chase`` both collapse to
    ``jpmorganchase``. Catches pairs where the same entity was
    written with different hyphenation conventions.
    """
    return slug.replace("-", "")


def suggest_winner(pair: DuplicatePair) -> WinnerSuggestion:
    """Pick a winner for ``pair`` using ordered heuristics.

    Rules, in priority order:

    1. **Singular over plural** — if one slug is the plural form of the
       other, the singular wins. Most canonical naming convention.
    2. **Prefix relationship** — if one slug is a strict prefix of the
       other (e.g. ``account-linking`` vs ``account-linking-banking``),
       the shorter one wins as the more general/canonical concept.
    3. **Hyphenation variant** — same slug with different hyphenation
       (e.g. ``jp-morgan-chase`` vs ``j-p-morgan-chase``). The version
       with fewer hyphen-separated tokens wins as the cleaner form.
    4. **Token-drop variant** — one slug has exactly one extra qualifier
       token (e.g. ``zelle-transfer-limits`` vs
       ``zelle-daily-transfer-limits``). The shorter, more general one
       wins.
    5. **More inbound links** — the more-linked page wins as the more
       established concept in the graph.
    6. **Older created date** — earlier creation wins as more
       established. Useful tiebreaker.
    7. **Alphabetical** — last resort to make the decision deterministic.

    ``is_safe_to_auto`` is True only when rules 1–4 fire AND the
    embedding similarity is ≥ 0.90. Other rules require human review
    because the pages may be related-but-distinct concepts.
    """
    slug_a = _slug_part(pair.page_a)
    slug_b = _slug_part(pair.page_b)

    # Rule 1: singular/plural
    if _is_plural_of(slug_a, slug_b):
        return WinnerSuggestion(
            winner_page_id=pair.page_a,
            loser_page_id=pair.page_b,
            reason="singular form preferred",
            is_safe_to_auto=pair.embedding_score >= 0.90,
        )
    if _is_plural_of(slug_b, slug_a):
        return WinnerSuggestion(
            winner_page_id=pair.page_b,
            loser_page_id=pair.page_a,
            reason="singular form preferred",
            is_safe_to_auto=pair.embedding_score >= 0.90,
        )

    # Rule 2: prefix relationship (one slug starts with the other + a separator)
    if slug_b.startswith(slug_a + "-"):
        return WinnerSuggestion(
            winner_page_id=pair.page_a,
            loser_page_id=pair.page_b,
            reason="shorter slug is prefix of longer (more general)",
            is_safe_to_auto=pair.embedding_score >= 0.90,
        )
    if slug_a.startswith(slug_b + "-"):
        return WinnerSuggestion(
            winner_page_id=pair.page_b,
            loser_page_id=pair.page_a,
            reason="shorter slug is prefix of longer (more general)",
            is_safe_to_auto=pair.embedding_score >= 0.90,
        )

    # Rule 3: hyphenation variant — same letters, different hyphenation.
    # Winner is the slug with fewer tokens (simpler form). Ties broken
    # alphabetically to stay deterministic.
    tokens_a = _tokens(slug_a)
    tokens_b = _tokens(slug_b)
    if (
        slug_a != slug_b
        and _hyphen_collapsed(slug_a) == _hyphen_collapsed(slug_b)
    ):
        if len(tokens_a) != len(tokens_b):
            if len(tokens_a) < len(tokens_b):
                return WinnerSuggestion(
                    winner_page_id=pair.page_a,
                    loser_page_id=pair.page_b,
                    reason="hyphenation variant (fewer tokens)",
                    is_safe_to_auto=pair.embedding_score >= 0.90,
                )
            return WinnerSuggestion(
                winner_page_id=pair.page_b,
                loser_page_id=pair.page_a,
                reason="hyphenation variant (fewer tokens)",
                is_safe_to_auto=pair.embedding_score >= 0.90,
            )
        # Same token count → deterministic alphabetical winner.
        if pair.page_a < pair.page_b:
            return WinnerSuggestion(
                winner_page_id=pair.page_a,
                loser_page_id=pair.page_b,
                reason="hyphenation variant (alphabetical)",
                is_safe_to_auto=pair.embedding_score >= 0.90,
            )
        return WinnerSuggestion(
            winner_page_id=pair.page_b,
            loser_page_id=pair.page_a,
            reason="hyphenation variant (alphabetical)",
            is_safe_to_auto=pair.embedding_score >= 0.90,
        )

    # Rule 4: token-drop variant — one slug has exactly one extra
    # qualifier token. Shorter (more general) concept wins.
    if _is_token_drop_variant(tokens_a, tokens_b):
        return WinnerSuggestion(
            winner_page_id=pair.page_a,
            loser_page_id=pair.page_b,
            reason="shorter slug drops one qualifier token",
            is_safe_to_auto=pair.embedding_score >= 0.90,
        )
    if _is_token_drop_variant(tokens_b, tokens_a):
        return WinnerSuggestion(
            winner_page_id=pair.page_b,
            loser_page_id=pair.page_a,
            reason="shorter slug drops one qualifier token",
            is_safe_to_auto=pair.embedding_score >= 0.90,
        )

    # Rule 5: more inbound links
    if pair.inbound_a != pair.inbound_b:
        if pair.inbound_a > pair.inbound_b:
            return WinnerSuggestion(
                winner_page_id=pair.page_a,
                loser_page_id=pair.page_b,
                reason=f"more inbound links ({pair.inbound_a} vs {pair.inbound_b})",
                is_safe_to_auto=False,
            )
        return WinnerSuggestion(
            winner_page_id=pair.page_b,
            loser_page_id=pair.page_a,
            reason=f"more inbound links ({pair.inbound_b} vs {pair.inbound_a})",
            is_safe_to_auto=False,
        )

    # Rule 4: older created
    if pair.created_a and pair.created_b and pair.created_a != pair.created_b:
        if pair.created_a < pair.created_b:
            return WinnerSuggestion(
                winner_page_id=pair.page_a,
                loser_page_id=pair.page_b,
                reason="created earlier",
                is_safe_to_auto=False,
            )
        return WinnerSuggestion(
            winner_page_id=pair.page_b,
            loser_page_id=pair.page_a,
            reason="created earlier",
            is_safe_to_auto=False,
        )

    # Rule 5: alphabetical fallback
    if pair.page_a < pair.page_b:
        return WinnerSuggestion(
            winner_page_id=pair.page_a,
            loser_page_id=pair.page_b,
            reason="alphabetical tiebreaker",
            is_safe_to_auto=False,
        )
    return WinnerSuggestion(
        winner_page_id=pair.page_b,
        loser_page_id=pair.page_a,
        reason="alphabetical tiebreaker",
        is_safe_to_auto=False,
    )
