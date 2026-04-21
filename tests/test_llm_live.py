"""Live-API smoke tests for the LLM provider abstraction.

**These tests hit a real provider and cost real money.** They are
skipped by default. To run them:

    ANTHROPIC_API_KEY=... pytest tests/test_llm_live.py -m live

The full suite (``pytest``) will not invoke them. CI should not
invoke them either. Use them as a manual "before release" check when
you want to verify end-to-end behavior against the current model.
"""

from __future__ import annotations

import json
import os

import pytest

from wikiloom.config import Config, LLMConfig
from wikiloom.llm import LLMClient

pytestmark = pytest.mark.live


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.fixture
def live_client() -> LLMClient:
    if not _has_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set; live tests require real creds")
    cfg = Config()
    cfg.llm = LLMConfig(
        provider="anthropic",
        default_model="claude-sonnet-4-20250514",
        max_tokens_per_operation=400,
    )
    return LLMClient(cfg)


def test_live_query_returns_nonempty(live_client: LLMClient) -> None:
    result = live_client.query(
        "You answer in exactly one short sentence.",
        "What color is the sky on a clear day?",
    )
    assert result.text.strip()
    assert result.metrics.tokens_in > 0
    assert result.metrics.tokens_out > 0
    assert result.metrics.cost_usd > 0


def test_live_synthesize_returns_structured_json(live_client: LLMClient) -> None:
    result = live_client.synthesize(
        "You output JSON only.",
        'Return a JSON object with one key "answer" whose value is 42.',
    )
    assert isinstance(result.result, dict)
    assert "answer" in result.result
    assert result.metrics.tokens_in > 0
    assert result.metrics.tokens_out > 0
