"""Unit tests for Component 20: LLM provider abstraction.

These tests monkeypatch ``litellm.completion`` so they run fast,
deterministic, and without hitting any real provider. The live-API
smoke tests live in ``tests/test_llm_live.py`` and are skipped by
default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import litellm

from wikiloom.config import Config, LLMConfig
from wikiloom.llm import (
    LLMCallMetrics,
    LLMClient,
    QueryResult,
    SynthesizeResult,
    estimate_cost,
)
from wikiloom.llm_errors import (
    LLMError,
    LLMProviderError,
    LLMResponseFormatError,
)


# ----------------------------------------------------------------------
# Fake litellm response shapes
# ----------------------------------------------------------------------


@dataclass
class _FakeMessage:
    content: str | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None


def _make_response(
    text: str,
    tokens_in: int = 100,
    tokens_out: int = 50,
    usage: bool = True,
) -> _FakeResponse:
    return _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=text))],
        usage=_FakeUsage(prompt_tokens=tokens_in, completion_tokens=tokens_out)
        if usage
        else None,
    )


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.llm = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        max_tokens_per_operation=4000,
    )
    return cfg


@pytest.fixture
def client(config: Config) -> LLMClient:
    return LLMClient(config)


@pytest.fixture
def mock_completion(monkeypatch: pytest.MonkeyPatch):
    """Patch ``litellm.completion`` with a recording mock.

    Returns a ``list`` that tests append ``(args, kwargs)`` to, plus
    a helper that installs a fixed response.
    """
    calls: list[tuple[tuple, dict]] = []

    def install(response_or_exc: Any) -> None:
        def fake_completion(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            if isinstance(response_or_exc, BaseException):
                raise response_or_exc
            return response_or_exc

        monkeypatch.setattr(litellm, "completion", fake_completion)

    return calls, install


# ----------------------------------------------------------------------
# estimate_cost
# ----------------------------------------------------------------------


def test_estimate_cost_known_model_nonzero() -> None:
    cost = estimate_cost(1000, 500, "claude-sonnet-4-20250514")
    assert cost > 0
    assert isinstance(cost, float)


def test_estimate_cost_unknown_model_returns_zero() -> None:
    cost = estimate_cost(1000, 500, "nonexistent-model-xyz-9999")
    assert cost == 0.0


def test_estimate_cost_zero_tokens_returns_zero() -> None:
    cost = estimate_cost(0, 0, "claude-sonnet-4-20250514")
    assert cost == 0.0


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_client_from_config(config: Config) -> None:
    client = LLMClient(config)
    assert client.model == "claude-sonnet-4-20250514"
    assert client.max_tokens == 4000


def test_client_from_llm_config_directly() -> None:
    """Accepts a bare LLMConfig too, not just a full Config."""
    llm_cfg = LLMConfig(model="foo-model", max_tokens_per_operation=1234)
    client = LLMClient(llm_cfg)
    assert client.model == "foo-model"
    assert client.max_tokens == 1234


# ----------------------------------------------------------------------
# synthesize — happy path
# ----------------------------------------------------------------------


def test_synthesize_returns_parsed_json(
    client: LLMClient, mock_completion
) -> None:
    calls, install = mock_completion
    install(
        _make_response(
            json.dumps({"title": "Transformer", "summary": "Attention-based."}),
            tokens_in=120,
            tokens_out=40,
        )
    )

    result = client.synthesize("You are a synthesizer.", "Summarize transformers.")
    assert isinstance(result, SynthesizeResult)
    assert result.result == {"title": "Transformer", "summary": "Attention-based."}
    assert result.metrics.tokens_in == 120
    assert result.metrics.tokens_out == 40
    assert result.metrics.cost_usd > 0
    assert result.metrics.model == "claude-sonnet-4-20250514"


def test_synthesize_sends_system_and_user_messages(
    client: LLMClient, mock_completion
) -> None:
    calls, install = mock_completion
    install(_make_response(json.dumps({"ok": True})))

    client.synthesize("SYS", "USER")
    assert len(calls) == 1
    _, kwargs = calls[0]
    msgs = kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("SYS")
    assert "JSON" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "USER"}
    assert "response_format" not in kwargs
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert kwargs["max_tokens"] == 4000


# ----------------------------------------------------------------------
# synthesize — error paths
# ----------------------------------------------------------------------


def test_synthesize_raises_provider_error_on_litellm_failure(
    client: LLMClient, mock_completion
) -> None:
    calls, install = mock_completion
    install(RuntimeError("rate limit"))

    with pytest.raises(LLMProviderError) as exc_info:
        client.synthesize("s", "u")
    assert exc_info.value.call_type == "synthesize"
    assert exc_info.value.model == "claude-sonnet-4-20250514"
    assert isinstance(exc_info.value.original, RuntimeError)


def test_synthesize_raises_format_error_on_invalid_json(
    client: LLMClient, mock_completion
) -> None:
    calls, install = mock_completion
    install(_make_response("not valid json {{{"))

    with pytest.raises(LLMResponseFormatError) as exc_info:
        client.synthesize("s", "u")
    assert exc_info.value.raw_text == "not valid json {{{"
    assert exc_info.value.parse_error  # non-empty error message from json.loads


def test_synthesize_rejects_non_dict_json(
    client: LLMClient, mock_completion
) -> None:
    """JSON list is valid JSON but not what synthesize expects."""
    calls, install = mock_completion
    install(_make_response(json.dumps([1, 2, 3])))

    with pytest.raises(LLMResponseFormatError) as exc_info:
        client.synthesize("s", "u")
    assert "expected JSON object" in exc_info.value.parse_error


# ----------------------------------------------------------------------
# query — happy path and errors
# ----------------------------------------------------------------------


def test_query_returns_plain_text(client: LLMClient, mock_completion) -> None:
    calls, install = mock_completion
    install(_make_response("The answer is 42.", tokens_in=50, tokens_out=10))

    result = client.query("You answer questions.", "What is the answer?")
    assert isinstance(result, QueryResult)
    assert result.text == "The answer is 42."
    assert result.metrics.tokens_in == 50
    assert result.metrics.tokens_out == 10


def test_query_does_not_append_json_instruction(
    client: LLMClient, mock_completion
) -> None:
    calls, install = mock_completion
    install(_make_response("hi"))
    client.query("SYS", "u")
    _, kwargs = calls[0]
    assert kwargs["messages"][0]["content"] == "SYS"
    assert "response_format" not in kwargs


def test_query_raises_provider_error(client: LLMClient, mock_completion) -> None:
    calls, install = mock_completion
    install(ValueError("auth failed"))

    with pytest.raises(LLMProviderError) as exc_info:
        client.query("s", "u")
    assert exc_info.value.call_type == "query"


# ----------------------------------------------------------------------
# vision_extract
# ----------------------------------------------------------------------


def test_vision_extract_sends_multimodal_message(
    client: LLMClient,
    mock_completion,
    tmp_path: Path,
) -> None:
    calls, install = mock_completion
    install(_make_response("A photo of a cat."))

    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")

    result = client.vision_extract(img, "Describe this image.")
    assert result.text == "A photo of a cat."
    assert len(calls) == 1
    _, kwargs = calls[0]
    msg = kwargs["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert msg["content"][0] == {"type": "text", "text": "Describe this image."}
    image_part = msg["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_vision_extract_raises_provider_error(
    client: LLMClient, mock_completion, tmp_path: Path
) -> None:
    calls, install = mock_completion
    install(TimeoutError("network"))

    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    with pytest.raises(LLMProviderError) as exc_info:
        client.vision_extract(img, "describe")
    assert exc_info.value.call_type == "vision_extract"


# ----------------------------------------------------------------------
# Response parsing edge cases
# ----------------------------------------------------------------------


def test_missing_usage_treated_as_zero_tokens(
    client: LLMClient, mock_completion
) -> None:
    """A provider that omits usage info shouldn't crash the call."""
    calls, install = mock_completion
    install(_make_response(json.dumps({"ok": True}), usage=False))

    result = client.synthesize("s", "u")
    assert result.metrics.tokens_in == 0
    assert result.metrics.tokens_out == 0
    assert result.metrics.cost_usd == 0.0


def test_null_content_becomes_empty_string(
    client: LLMClient, mock_completion
) -> None:
    """Some providers return None content on empty responses."""
    calls, install = mock_completion
    # query path — plain text, None becomes "".
    install(
        _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=None))],
            usage=_FakeUsage(prompt_tokens=10, completion_tokens=0),
        )
    )
    result = client.query("s", "u")
    assert result.text == ""


def test_malformed_response_shape_raises_llm_error(
    client: LLMClient, mock_completion
) -> None:
    """A response with no choices at all should surface a clean error."""
    calls, install = mock_completion
    install(_FakeResponse(choices=[], usage=None))

    with pytest.raises(LLMError):
        client.query("s", "u")


# ----------------------------------------------------------------------
# Retry on transient errors
# ----------------------------------------------------------------------


def _install_sequence(
    monkeypatch: pytest.MonkeyPatch, responses: list[Any]
) -> list[tuple[tuple, dict]]:
    """Patch litellm.completion to return/raise from a sequence in order."""
    calls: list[tuple[tuple, dict]] = []
    iterator = iter(responses)

    def fake_completion(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        item = next(iterator)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return calls


def _named_exception(name: str, message: str = "boom") -> Exception:
    """Build an exception whose class name matches the retry classifier."""
    cls = type(name, (Exception,), {})
    return cls(message)


def test_synthesize_retries_on_rate_limit_then_succeeds(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transient RateLimitError is retried; eventual success returns normally."""
    monkeypatch.setattr("wikiloom.llm.time.sleep", lambda _s: None)
    calls = _install_sequence(
        monkeypatch,
        [
            _named_exception("RateLimitError"),
            _make_response(json.dumps({"ok": True})),
        ],
    )

    result = client.synthesize("s", "u")
    assert result.result == {"ok": True}
    assert len(calls) == 2  # one failure + one success


def test_synthesize_retries_exhaust_then_raises(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent RateLimitError exhausts attempts and raises LLMProviderError."""
    monkeypatch.setattr("wikiloom.llm.time.sleep", lambda _s: None)
    calls = _install_sequence(
        monkeypatch,
        [
            _named_exception("RateLimitError"),
            _named_exception("RateLimitError"),
            _named_exception("RateLimitError"),
        ],
    )

    with pytest.raises(LLMProviderError):
        client.synthesize("s", "u")
    assert len(calls) == 3  # max attempts


def test_synthesize_does_not_retry_on_quota_exhausted(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quota errors raise immediately — backoff won't bring credits back."""
    monkeypatch.setattr("wikiloom.llm.time.sleep", lambda _s: None)
    calls = _install_sequence(
        monkeypatch,
        [
            _named_exception(
                "RateLimitError",
                "credit_balance_too_low: please top up",
            ),
        ],
    )

    with pytest.raises(LLMProviderError):
        client.synthesize("s", "u")
    assert len(calls) == 1  # no retries


def test_synthesize_does_not_retry_on_auth_error(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-retryable error types (auth, validation) fail immediately."""
    monkeypatch.setattr("wikiloom.llm.time.sleep", lambda _s: None)
    calls = _install_sequence(
        monkeypatch,
        [_named_exception("AuthenticationError", "invalid api key")],
    )

    with pytest.raises(LLMProviderError):
        client.synthesize("s", "u")
    assert len(calls) == 1


def test_query_also_retries_transient_errors(
    client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry behavior applies to query() the same as synthesize()."""
    monkeypatch.setattr("wikiloom.llm.time.sleep", lambda _s: None)
    calls = _install_sequence(
        monkeypatch,
        [
            _named_exception("APIConnectionError"),
            _make_response("done"),
        ],
    )

    result = client.query("s", "u")
    assert result.text == "done"
    assert len(calls) == 2
