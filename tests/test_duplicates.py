"""Tests for ``wikiloom.duplicates.suggest_winner``.

Covers the rules that drive ``is_safe_to_auto`` — singular/plural,
prefix, hyphenation, token-drop — plus the rules that don't
(inbound-links, older-created, alphabetical tiebreaker).
"""

from __future__ import annotations

from wikiloom.duplicates import DuplicatePair, suggest_winner


def _pair(
    a: str,
    b: str,
    *,
    slug_score: float = 95.0,
    embedding_score: float = 0.95,
    inbound_a: int = 0,
    inbound_b: int = 0,
    created_a: str = "",
    created_b: str = "",
    type_: str = "concept",
) -> DuplicatePair:
    return DuplicatePair(
        page_a=a,
        page_b=b,
        title_a=a.rsplit("/", 1)[-1].replace("-", " ").title(),
        title_b=b.rsplit("/", 1)[-1].replace("-", " ").title(),
        type_a=type_,
        type_b=type_,
        slug_score=slug_score,
        embedding_score=embedding_score,
        inbound_a=inbound_a,
        inbound_b=inbound_b,
        created_a=created_a,
        created_b=created_b,
    )


# ----------------------------------------------------------------------
# Singular/plural rule
# ----------------------------------------------------------------------


def test_plural_rule_catches_trailing_s() -> None:
    sug = suggest_winner(_pair("concepts/account", "concepts/accounts"))
    assert sug.winner_page_id == "concepts/account"
    assert sug.reason == "singular form preferred"
    assert sug.is_safe_to_auto is True


def test_plural_rule_catches_trailing_es() -> None:
    sug = suggest_winner(_pair("concepts/box", "concepts/boxes"))
    assert sug.winner_page_id == "concepts/box"
    assert sug.is_safe_to_auto is True


def test_plural_rule_catches_y_to_ies() -> None:
    """Regression: pairs like ``penalty`` / ``penalties`` used to fall
    through to the alphabetical tiebreaker because ``_is_plural_of``
    only handled +s and +es. Now the y→ies transformation is covered
    as a safe-to-auto case."""
    sug = suggest_winner(
        _pair(
            "concepts/cd-early-withdrawal-penalty",
            "concepts/cd-early-withdrawal-penalties",
        )
    )
    assert sug.winner_page_id == "concepts/cd-early-withdrawal-penalty"
    assert sug.reason == "singular form preferred"
    assert sug.is_safe_to_auto is True


def test_plural_rule_requires_embedding_floor() -> None:
    """Even a correct plural match falls back to is_safe_to_auto=False
    when the embedding score is below 0.90 — the semantic confirmation
    guard."""
    sug = suggest_winner(
        _pair(
            "concepts/account",
            "concepts/accounts",
            embedding_score=0.80,
        )
    )
    assert sug.winner_page_id == "concepts/account"
    assert sug.is_safe_to_auto is False


# ----------------------------------------------------------------------
# Prefix rule
# ----------------------------------------------------------------------


def test_prefix_rule_shorter_wins() -> None:
    sug = suggest_winner(
        _pair(
            "concepts/account-linking",
            "concepts/account-linking-banking",
        )
    )
    assert sug.winner_page_id == "concepts/account-linking"
    assert "prefix" in sug.reason
    assert sug.is_safe_to_auto is True


# ----------------------------------------------------------------------
# Fallbacks that must NOT be safe-to-auto
# ----------------------------------------------------------------------


def test_alphabetical_tiebreaker_is_not_auto_safe() -> None:
    """Two distinct concepts sharing a prefix (but not in a
    prefix-relationship) fall to the alphabetical tiebreaker and must
    not auto-merge."""
    sug = suggest_winner(
        _pair(
            "concepts/merchant-statement-reconciliation",
            "concepts/transaction-history-reconciliation",
            slug_score=69,
            embedding_score=0.89,
        )
    )
    assert sug.is_safe_to_auto is False
