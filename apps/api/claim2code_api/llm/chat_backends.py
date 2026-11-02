"""Chat backends behind :class:`claim2code_api.llm.client.LLMClient`.

``LLMClient`` was written against the OpenAI SDK: it calls
``backend.chat.completions.create(model=..., messages=...,
response_format=...)`` and reads ``response.choices[0].message.content``
and ``response.usage``. Each backend here keeps that call shape so the
client's prompts, safety gate, tracing and parsing stay provider-neutral:

  - :class:`OpenAICompatBackend` — the OpenAI SDK pointed at any
    chat-completions server. For servers without JSON-schema support
    (``json_mode="object"``) it swaps the schema for JSON mode and puts
    the schema in the system prompt.
  - :class:`AnthropicBackend` — Claude through the official Anthropic
    SDK, with the response schema enforced by structured outputs
    (``output_config.format``).

Both also expose :meth:`ping` for the provider "test connection" check.
"""
from __future__ import annotations

import copy
import json
import logging
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
from openai import OpenAI

from claim2code_api.llm.providers import ResolvedProvider

logger = logging.getLogger(__name__)


def build_backend(provider: ResolvedProvider, *, timeout: float) -> OpenAICompatBackend | AnthropicBackend:
    if provider.kind == "anthropic":
        return AnthropicBackend(provider, timeout=timeout)
    return OpenAICompatBackend(provider, timeout=timeout)


# ----------------------------------------------------------------------
# OpenAI-compatible
# ----------------------------------------------------------------------


class OpenAICompatBackend:
    def __init__(self, provider: ResolvedProvider, *, timeout: float) -> None:
        self._json_mode = provider.json_mode
        self._client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            # ``proxy=None`` disables proxying via env vars: httpx 0.28
            # does not understand ``socks://`` (see llm/client.py).
            http_client=httpx.Client(proxy=None),  # type: ignore[arg-type]
            timeout=timeout,
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        fmt = kwargs.get("response_format")
        if self._json_mode == "object" and fmt and fmt.get("type") == "json_schema":
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"] = _with_schema_instruction(
                kwargs["messages"], fmt["json_schema"]["schema"],
            )
        return self._client.chat.completions.create(**kwargs)

    def ping(self, model: str) -> None:
        self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=30.0,
        )


def _with_schema_instruction(
    messages: list[dict[str, Any]], schema: dict[str, Any],
) -> list[dict[str, Any]]:
    instruction = (
        "Reply with a single JSON object that validates against this JSON schema, "
        "and nothing else:\n" + json.dumps(schema)
    )
    out = [dict(m) for m in messages]
    for m in out:
        if m["role"] == "system":
            m["content"] = f"{m['content']}\n\n{instruction}"
            return out
    return [{"role": "system", "content": instruction}, *out]


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------

# Room for adaptive thinking plus a full claim list; requests stream so
# a large ceiling does not hit the SDK's non-streaming timeout guard.
_MAX_TOKENS = 64000

# Models that take ``output_config.effort``; others reject it.
_EFFORT_MODELS = (
    "claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5",
    "claude-haiku-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-4-6",
)
# Models that accept the server-side refusal fallback (``fallbacks="default"``).
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# JSON-schema keywords structured outputs does not accept. Pydantic
# still enforces them when the reply is validated.
_UNSUPPORTED_KEYWORDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
})


class AnthropicBackend:
    def __init__(self, provider: ResolvedProvider, *, timeout: float) -> None:
        self._client = anthropic.Anthropic(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=timeout,
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> Any:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"
        ]
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "messages": turns,
        }
        if system:
            params["system"] = system
        output_config: dict[str, Any] = {}
        if model.startswith(_EFFORT_MODELS):
            # Verification is judgement-heavy; "high" is the floor for that.
            output_config["effort"] = "high"
        if response_format and response_format.get("type") == "json_schema":
            output_config["format"] = {
                "type": "json_schema",
                "schema": strict_schema(response_format["json_schema"]["schema"]),
            }
        if output_config:
            params["output_config"] = output_config

        message: Any
        if model.startswith(_FALLBACK_MODELS):
            with self._client.beta.messages.stream(
                betas=[_FALLBACK_BETA], fallbacks="default", **params,
            ) as stream:
                message = stream.get_final_message()
        else:
            with self._client.messages.stream(**params) as stream:
                message = stream.get_final_message()
        return _as_chat_completion(message)

    def ping(self, model: str) -> None:
        # Confirms the key and the model id without spending tokens.
        self._client.models.retrieve(model, timeout=30.0)


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a Pydantic JSON schema to what structured outputs accepts.

    Drops the unsupported constraint keywords and closes every object
    (``additionalProperties: false``), recursively, on a copy.
    """
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}
            if out.get("type") == "object" or "properties" in out:
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(copy.deepcopy(schema))


def _as_chat_completion(message: Any) -> SimpleNamespace:
    """Shape an Anthropic ``Message`` like an OpenAI chat completion."""
    text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
    if message.stop_reason == "refusal":
        logger.warning("anthropic refusal model=%s details=%s", message.model, message.stop_details)
        text = ""
    elif message.stop_reason == "max_tokens":
        logger.warning("anthropic reply hit max_tokens model=%s", message.model)
    usage = message.usage
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text or None),
            finish_reason=message.stop_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
        ),
        model=message.model,
    )


__all__ = ["AnthropicBackend", "OpenAICompatBackend", "build_backend", "strict_schema"]
