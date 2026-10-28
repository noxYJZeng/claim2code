"""Chat-model providers the user can switch between at runtime.

Claim extraction, verification and related-work comparison all go
through one chat model. Which one is a runtime choice: the UI lists
the presets below, the user picks one (plus a model name, and a base
URL or API key where the preset needs one), and the next LLM call uses
it. Embeddings are NOT switchable here: every stored vector was made by
``OPENAI_MODEL_EMBED`` at ``OPENAI_EMBED_DIM`` dimensions, so changing
the embedding model means re-indexing, which stays an ``.env`` change.

Two wire formats cover every preset:

  - ``openai_compat`` — the OpenAI chat-completions API (OpenAI, LM
    Studio, Ollama, Gemini, DeepSeek, OpenRouter, any compatible
    server). ``json_mode="schema"`` sends the response JSON schema
    (``response_format.json_schema``); ``json_mode="object"`` is for
    servers that only accept ``{"type": "json_object"}`` and gets the
    schema in the system prompt instead. Either way the reply is
    validated against the Pydantic model afterwards.
  - ``anthropic`` — Claude through the official Anthropic SDK
    (:mod:`claim2code_api.llm.anthropic_chat`).

The choice (preset, model, base URL) is saved to
``<project root>/.claim2code/llm_provider.json`` so it survives a
restart. An API key typed into the UI is held in process memory only
and is never written to disk or returned by the API; a key that should
persist belongs in ``.env``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal

from claim2code_api.app.config import _PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

ProviderKind = Literal["openai_compat", "anthropic"]
JsonMode = Literal["schema", "object"]

STATE_FILE = _PROJECT_ROOT / ".claim2code" / "llm_provider.json"


@dataclass(frozen=True)
class ProviderPreset:
    """One selectable provider.

    ``key_setting`` names the :class:`Settings` field (i.e. the ``.env``
    variable) holding the provider's API key; ``None`` means the server
    needs no key. ``base_url=None`` means the user must supply one.
    """

    id: str
    label: str
    kind: ProviderKind
    base_url: str | None
    default_model: str
    models: tuple[str, ...] = ()
    key_setting: str | None = None
    json_mode: JsonMode = "schema"
    local: bool = False
    note: str = ""


PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="env",
        label="From .env",
        kind="openai_compat",
        base_url=None,  # resolved from OPENAI_BASE_URL
        default_model="",  # resolved from OPENAI_MODEL_CHAT
        key_setting="openai_api_key",
        note="Whatever OPENAI_BASE_URL / OPENAI_MODEL_CHAT point at.",
    ),
    ProviderPreset(
        id="lmstudio",
        label="LM Studio (local)",
        kind="openai_compat",
        base_url="http://localhost:1234/v1",
        default_model="qwen3.8-27b-mlx",
        local=True,
        note="Runs on this machine; load the model in LM Studio first.",
    ),
    ProviderPreset(
        id="ollama",
        label="Ollama (local)",
        kind="openai_compat",
        base_url="http://localhost:11434/v1",
        default_model="qwen3:32b",
        local=True,
        note="Runs on this machine; `ollama pull <model>` first.",
    ),
    ProviderPreset(
        id="anthropic",
        label="Anthropic (Claude)",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        default_model="claude-opus-5-5",
        models=("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
        key_setting="anthropic_api_key",
    ),
    ProviderPreset(
        id="openai",
        label="OpenAI",
        kind="openai_compat",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        models=("gpt-4o", "gpt-4o-mini"),
        key_setting="openai_platform_api_key",
    ),
    ProviderPreset(
        id="gemini",
        label="Google Gemini",
        kind="openai_compat",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-pro",
        models=("gemini-2.5-pro", "gemini-2.5-flash"),
        key_setting="gemini_api_key",
        note="Through Gemini's OpenAI-compatible endpoint.",
    ),
    ProviderPreset(
        id="deepseek",
        label="DeepSeek",
        kind="openai_compat",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        key_setting="deepseek_api_key",
        json_mode="object",
        note="JSON mode only (no schema enforcement); replies are validated afterwards.",
    ),
    ProviderPreset(
        id="openrouter",
        label="OpenRouter",
        kind="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        default_model="",
        key_setting="openrouter_api_key",
        note="Model names look like `vendor/model`; pick one that supports structured outputs.",
    ),
    ProviderPreset(
        id="custom",
        label="Other OpenAI-compatible server",
        kind="openai_compat",
        base_url=None,
        default_model="",
        note="Any server speaking the chat-completions API (vLLM, llama.cpp, a gateway, ...).",
    ),
)

_PRESETS_BY_ID = {p.id: p for p in PRESETS}


@dataclass(frozen=True)
class ResolvedProvider:
    """Everything an LLM client needs to make a call."""

    preset_id: str
    label: str
    kind: ProviderKind
    base_url: str
    api_key: str
    model: str
    json_mode: JsonMode


@dataclass
class _Selection:
    preset_id: str = "env"
    model: str = ""
    base_url: str = ""
    # Memory only — never persisted or returned.
    api_key: str = field(default="", repr=False)


class ProviderError(ValueError):
    """The requested provider selection is incomplete or unknown."""


_lock = Lock()
_selection: _Selection | None = None


def get_preset(preset_id: str) -> ProviderPreset:
    try:
        return _PRESETS_BY_ID[preset_id]
    except KeyError:
        raise ProviderError(f"unknown provider {preset_id!r}") from None


def _load() -> _Selection:
    try:
        data = json.loads(STATE_FILE.read_text())
        get_preset(data["preset_id"])
        return _Selection(
            preset_id=data["preset_id"],
            model=data.get("model", ""),
            base_url=data.get("base_url", ""),
        )
    except FileNotFoundError:
        return _Selection()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("ignoring unreadable %s: %s", STATE_FILE, exc)
        return _Selection()


def _current() -> _Selection:
    global _selection
    if _selection is None:
        _selection = _load()
    return _selection


def _settings_key(settings: Settings, preset: ProviderPreset) -> str:
    if preset.key_setting is None:
        return ""
    return str(getattr(settings, preset.key_setting) or "")


def resolve(
    settings: Settings,
    *,
    preset_id: str,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> ResolvedProvider:
    """Fill a selection's gaps from the preset and ``.env``.

    :raises ProviderError: unknown preset, or a required base URL,
        model or API key is missing.
    """
    preset = get_preset(preset_id)
    if preset.id == "env":
        url = base_url or settings.openai_base_url
        name = model or settings.openai_model_chat
    else:
        url = base_url or preset.base_url or ""
        name = model or preset.default_model
    key = api_key or _settings_key(settings, preset)
    if not url:
        raise ProviderError(f"{preset.label}: a base URL is required")
    if not name:
        raise ProviderError(f"{preset.label}: a model name is required")
    if preset.key_setting is not None and preset.id != "env" and not key:
        raise ProviderError(
            f"{preset.label}: no API key — set {preset.key_setting.upper()} in .env "
            "or enter one in the UI",
        )
    return ResolvedProvider(
        preset_id=preset.id,
        label=preset.label,
        kind=preset.kind,
        base_url=url,
        # The OpenAI SDK refuses an empty key even for keyless local servers.
        api_key=key or "not-needed",
        model=name,
        json_mode=preset.json_mode,
    )


def active(settings: Settings) -> ResolvedProvider:
    """The provider the next LLM call will use.

    Falls back to ``.env`` when the saved selection no longer resolves
    (e.g. a UI-entered key was lost on restart).
    """
    with _lock:
        sel = _current()
        try:
            return resolve(
                settings, preset_id=sel.preset_id, model=sel.model,
                base_url=sel.base_url, api_key=sel.api_key,
            )
        except ProviderError as exc:
            logger.warning("saved LLM provider unusable (%s); using .env", exc)
            return resolve(settings, preset_id="env")


def select(
    settings: Settings,
    *,
    preset_id: str,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> ResolvedProvider:
    """Make a selection active and persist it (without the key).

    An empty ``api_key`` keeps a key entered earlier for the same
    preset, so re-saving a model name does not demand the key again.
    """
    global _selection
    with _lock:
        sel = _current()
        if not api_key and sel.preset_id == preset_id:
            api_key = sel.api_key
        resolved = resolve(
            settings, preset_id=preset_id, model=model, base_url=base_url, api_key=api_key,
        )
        _selection = _Selection(
            preset_id=preset_id, model=model, base_url=base_url, api_key=api_key,
        )
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(
                {"preset_id": preset_id, "model": model, "base_url": base_url}, indent=2,
            ))
        except OSError as exc:
            logger.warning("could not save LLM provider to %s: %s", STATE_FILE, exc)
    return resolved


def session_key(preset_id: str) -> str:
    """A key entered in the UI this session for ``preset_id``, if any."""
    with _lock:
        sel = _current()
        return sel.api_key if sel.preset_id == preset_id else ""


def describe(settings: Settings) -> dict[str, object]:
    """Public view for the UI: presets, key status, and the active choice.

    Never includes an API key.
    """
    with _lock:
        sel = _current()
        session_key_for = sel.preset_id if sel.api_key else None
    current = active(settings)
    presets = []
    for p in PRESETS:
        env_default = p.id == "env"
        presets.append({
            "id": p.id,
            "label": p.label,
            "kind": p.kind,
            "base_url": settings.openai_base_url if env_default else p.base_url,
            "default_model": settings.openai_model_chat if env_default else p.default_model,
            "models": list(p.models),
            "needs_key": p.key_setting is not None and not env_default,
            "key_env": p.key_setting.upper() if p.key_setting else None,
            "key_configured": bool(_settings_key(settings, p)) or session_key_for == p.id,
            "json_mode": p.json_mode,
            "local": p.local,
            "note": p.note,
        })
    return {
        "active": {
            "provider_id": current.preset_id,
            "label": current.label,
            "model": current.model,
            "base_url": current.base_url,
        },
        "presets": presets,
        "embedding": {
            "model": settings.openai_model_embed,
            "base_url": settings.openai_base_url,
            "dim": settings.openai_embed_dim,
        },
    }


def reset_for_tests(state_file: Path | None = None) -> None:
    """Forget the in-memory selection (and optionally move the state file)."""
    global _selection, STATE_FILE
    with _lock:
        _selection = None
        if state_file is not None:
            STATE_FILE = state_file


__all__ = [
    "PRESETS",
    "ProviderError",
    "ProviderPreset",
    "ResolvedProvider",
    "active",
    "describe",
    "get_preset",
    "resolve",
    "select",
    "session_key",
]
