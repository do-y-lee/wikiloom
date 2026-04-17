"""LLM provider abstraction.

Wraps litellm.completion with three call shapes: synthesize (JSON),
query (plain text), and vision_extract (multimodal). Returns token
usage and cost estimates on every call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm

litellm.suppress_debug_info = True

from wikiloom.config import Config, LLMConfig
from wikiloom.llm_errors import (
    LLMError,
    LLMProviderError,
    LLMResponseFormatError,
)

# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass
class LLMCallMetrics:
    """Token usage + estimated cost for a single completion call.

    Attached to every ``LLMClient`` return value so the event log and
    source catalog can track real spend instead of zeros. ``cost_usd``
    is best-effort — models litellm doesn't price yet surface as 0.0
    rather than raising.
    """

    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str


@dataclass
class SynthesizeResult:
    """Return shape of ``LLMClient.synthesize``.

    The ``result`` dict is whatever structured JSON the model emitted,
    validated as parseable JSON but not schema-checked — the ingest
    synthesis loop (a future C20 enablement pass) owns schema
    validation against ``output_formats/ingest_response.json``.
    """

    result: dict[str, Any]
    metrics: LLMCallMetrics


@dataclass
class QueryResult:
    """Return shape of ``LLMClient.query``."""

    text: str
    metrics: LLMCallMetrics


# ----------------------------------------------------------------------
# Cost estimation
# ----------------------------------------------------------------------


def estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    """Estimate USD cost for a completion call using litellm's pricing.

    Returns 0.0 for models litellm doesn't recognize rather than
    raising — a new model shipping shouldn't break ingests that
    happen to touch it before our litellm pin is bumped.
    """
    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
        )
    except Exception:
        return 0.0
    return float(input_cost) + float(output_cost)


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------


class LLMClient:
    """Provider-agnostic LLM client backed by litellm.

    Constructed with a full ``Config`` so per-component settings
    (``llm.model``, ``llm.max_tokens_per_operation``) are available
    without the caller plumbing them through. Holds no network state
    — each method call is an independent ``litellm.completion`` with
    its own retries governed by litellm's own defaults.
    """

    def __init__(self, config: Config | LLMConfig):
        llm_cfg = config.llm if isinstance(config, Config) else config
        self.model: str = llm_cfg.model
        self.max_tokens: int = llm_cfg.max_tokens_per_operation

    # ------------------------------------------------------------------
    # synthesize — structured JSON output
    # ------------------------------------------------------------------

    def synthesize(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> SynthesizeResult:
        """Run a completion that returns JSON and parse the result.

        Does NOT use ``response_format={"type": "json_object"}``
        because litellm's translation of that flag to Anthropic's
        tool-use mechanism can produce empty ``message.content``.
        Instead, the system prompt instructs the model to return
        raw JSON, and we strip any markdown code fences before
        parsing.

        Raises:
            LLMProviderError: litellm / provider error (rate limit,
                auth, network, etc.). Inspect ``exc.original`` for the
                underlying litellm exception.
            LLMResponseFormatError: the model returned text that isn't
                valid JSON. The raw response is available on the
                exception for debugging.
        """
        json_instruction = (
            "\n\nIMPORTANT: Return your response as raw JSON only. "
            "No markdown code fences, no explanation, no commentary. "
            "Just the JSON object."
        )
        messages = [
            {"role": "system", "content": system_prompt + json_instruction},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise LLMProviderError(
                model=self.model,
                call_type="synthesize",
                original=exc,
            ) from exc

        raw_text, metrics = _extract_text_and_metrics(response, self.model)
        cleaned = _strip_code_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                model=self.model,
                raw_text=raw_text,
                parse_error=str(exc),
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMResponseFormatError(
                model=self.model,
                raw_text=raw_text,
                parse_error=(
                    f"expected JSON object, got {type(parsed).__name__}"
                ),
            )
        return SynthesizeResult(result=parsed, metrics=metrics)

    # ------------------------------------------------------------------
    # query — plain-text completion
    # ------------------------------------------------------------------

    def query(self, system_prompt: str, user_prompt: str) -> QueryResult:
        """Run a plain-text completion and return the response string."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise LLMProviderError(
                model=self.model,
                call_type="query",
                original=exc,
            ) from exc

        text, metrics = _extract_text_and_metrics(response, self.model)
        return QueryResult(text=text, metrics=metrics)

    # ------------------------------------------------------------------
    # vision_extract — multimodal
    # ------------------------------------------------------------------

    def vision_extract(self, image_path: Path, prompt: str) -> QueryResult:
        """Ask the model to describe an image at ``image_path``.

        Encodes the image as a data URL and sends it as a multimodal
        message. Uses the configured ``model`` — caller's responsibility
        to pick one with vision support.
        """
        import base64
        import mimetypes

        image_path = Path(image_path)
        mime, _ = mimetypes.guess_type(image_path.name)
        if mime is None:
            mime = "image/png"
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime};base64,{data}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise LLMProviderError(
                model=self.model,
                call_type="vision_extract",
                original=exc,
            ) from exc

        text, metrics = _extract_text_and_metrics(response, self.model)
        return QueryResult(text=text, metrics=metrics)


import re

# ----------------------------------------------------------------------
# Response parsing helpers
# ----------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL
)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping a JSON response.

    LLMs often return ``json\\n{...}\\n``` `` even when told not to.
    This strips the outermost fence if present so ``json.loads``
    sees clean JSON.
    """
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _extract_text_and_metrics(
    response: Any, model: str
) -> tuple[str, LLMCallMetrics]:
    """Pull the message text + usage numbers out of a litellm response.

    litellm normalizes provider responses to an OpenAI-shaped object,
    so ``response.choices[0].message.content`` and
    ``response.usage.{prompt,completion}_tokens`` are the canonical
    access paths. Missing usage numbers are treated as 0 rather than
    raising, so a provider that omits them doesn't break the call.
    """
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, KeyError) as exc:
        raise LLMError(
            f"Unexpected litellm response shape: {exc}"
        ) from exc

    if content is None:
        content = ""

    usage = getattr(response, "usage", None)
    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    cost = estimate_cost(tokens_in, tokens_out, model)

    metrics = LLMCallMetrics(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        model=model,
    )
    return content, metrics
