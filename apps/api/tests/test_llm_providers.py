"""Switchable chat providers: registry, backends and HTTP routes.

No network: the backends' SDK clients are replaced with fakes.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from claim2code_api.app.config import Settings
from claim2code_api.llm import chat_backends, providers


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    providers.reset_for_tests(tmp_path / "llm_provider.json")
    yield tmp_path / "llm_provider.json"
    providers.reset_for_tests()


def _settings(**overrides) -> Settings:
    base = dict(
        openai_api_key="lm-studio",
        openai_base_url="http://localhost:1234/v1",
        openai_model_chat="local-model",
        anthropic_api_key="",
    )
    return Settings(**{**base, **overrides})


def _resolved(kind="openai_compat", json_mode="schema") -> providers.ResolvedProvider:
    return providers.ResolvedProvider(
        preset_id="x", label="X", kind=kind, base_url="http://h", api_key="k",
        model="m", json_mode=json_mode,
    )


# ============================================================
# Registry
# ============================================================


def test_default_is_the_env_provider() -> None:
    p = providers.active(_settings())
    assert (p.preset_id, p.base_url, p.model) == ("env", "http://localhost:1234/v1", "local-model")


def test_select_persists_the_choice_but_not_the_key(_isolated_state) -> None:
    s = _settings()
    p = providers.select(s, preset_id="anthropic", api_key="sk-ant-session-only")
    assert (p.kind, p.model, p.api_key) == ("anthropic", "claude-opus-5-5", "sk-ant-session-only")
    saved = json.loads(_isolated_state.read_text())
    assert saved == {"preset_id": "anthropic", "model": "", "base_url": ""}
    assert "sk-ant" not in _isolated_state.read_text()

    # Re-saving the same provider without a key keeps the session key.
    assert providers.select(s, preset_id="anthropic", model="claude-haiku-5-5").api_key == (
        "sk-ant-session-only"
    )
    # After a restart the session key is gone, so the choice falls back to .env.
    providers.reset_for_tests(_isolated_state)
    assert providers.active(s).preset_id == "env"


def test_resolve_requires_a_key_url_and_model() -> None:
    s = _settings()
    with pytest.raises(providers.ProviderError, match="ANTHROPIC_API_KEY"):
        providers.resolve(s, preset_id="anthropic")
    assert providers.resolve(_settings(anthropic_api_key="from-env"), preset_id="anthropic").api_key == (
        "from-env"
    )
    with pytest.raises(providers.ProviderError, match="base URL"):
        providers.resolve(s, preset_id="custom", model="m")
    with pytest.raises(providers.ProviderError, match="model name"):
        providers.resolve(s, preset_id="custom", base_url="https://h/v1")
    with pytest.raises(providers.ProviderError, match="unknown provider"):
        providers.resolve(s, preset_id="nope")
    # Keyless local servers still get a placeholder the OpenAI SDK accepts.
    assert providers.resolve(s, preset_id="ollama").api_key == "not-needed"


def test_describe_never_returns_keys() -> None:
    s = _settings(anthropic_api_key="sk-ant-from-env")
    providers.select(s, preset_id="deepseek", api_key="sk-deepseek-session")
    view = cast(dict[str, Any], providers.describe(s))
    text = json.dumps(view)
    assert "sk-ant-from-env" not in text and "sk-deepseek-session" not in text
    by_id = {p["id"]: p for p in view["presets"]}
    assert by_id["anthropic"]["key_configured"] and by_id["deepseek"]["key_configured"]
    assert not by_id["openai"]["key_configured"]
    assert view["active"]["provider_id"] == "deepseek"
    assert view["embedding"]["base_url"] == "http://localhost:1234/v1"


# ============================================================
# Backends
# ============================================================


def test_openai_compat_object_mode_moves_the_schema_into_the_prompt() -> None:
    backend = chat_backends.OpenAICompatBackend(_resolved(json_mode="object"), timeout=5)
    backend._client = MagicMock()
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    backend.chat.completions.create(
        model="m",
        messages=[{"role": "system", "content": "Rules."}, {"role": "user", "content": "Go."}],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
    )
    kwargs = backend._client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"][0]["content"].startswith("Rules.")
    assert json.dumps(schema) in kwargs["messages"][0]["content"]


def test_openai_compat_schema_mode_passes_the_request_through() -> None:
    backend = chat_backends.OpenAICompatBackend(_resolved(), timeout=5)
    backend._client = MagicMock()
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    backend.chat.completions.create(model="m", messages=[], response_format=fmt)
    assert backend._client.chat.completions.create.call_args.kwargs["response_format"] is fmt


def test_strict_schema_closes_objects_and_drops_unsupported_constraints() -> None:
    out = chat_backends.strict_schema({
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "$defs": {"Inner": {"properties": {"x": {"type": "integer"}}}},
    })
    assert out["additionalProperties"] is False
    assert out["$defs"]["Inner"]["additionalProperties"] is False
    assert out["properties"]["name"] == {"type": "string"}
    assert out["properties"]["score"] == {"type": "number"}


def _fake_stream(message):
    stream = MagicMock()
    stream.__enter__.return_value.get_final_message.return_value = message
    return stream


def _message(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_anthropic_backend_maps_the_openai_call_shape() -> None:
    backend = chat_backends.AnthropicBackend(_resolved(kind="anthropic"), timeout=5)
    backend._client = MagicMock()
    backend._client.beta.messages.stream.return_value = _fake_stream(_message('{"a": "b"}'))
    schema = {"type": "object", "properties": {"a": {"type": "string", "minLength": 1}}}

    resp = backend.chat.completions.create(
        model="claude-opus-5-5",
        messages=[{"role": "system", "content": "Rules."}, {"role": "user", "content": "Go."}],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
    )

    kwargs = backend._client.beta.messages.stream.call_args.kwargs
    assert kwargs["system"] == "Rules."
    assert kwargs["messages"] == [{"role": "user", "content": "Go."}]
    assert kwargs["output_config"]["effort"] == "high"
    assert kwargs["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert kwargs["fallbacks"] == "default"
    assert resp.choices[0].message.content == '{"a": "b"}'
    assert resp.usage.total_tokens == 15


def test_anthropic_backend_skips_fallbacks_and_effort_on_older_models() -> None:
    backend = chat_backends.AnthropicBackend(_resolved(kind="anthropic"), timeout=5)
    backend._client = MagicMock()
    backend._client.messages.stream.return_value = _fake_stream(_message("{}"))
    backend.chat.completions.create(
        model="claude-haiku-4-5", messages=[{"role": "user", "content": "Go."}],
    )
    kwargs = backend._client.messages.stream.call_args.kwargs
    assert "output_config" not in kwargs and "fallbacks" not in kwargs
    backend._client.beta.messages.stream.assert_not_called()


def test_anthropic_refusal_comes_back_as_empty_content() -> None:
    backend = chat_backends.AnthropicBackend(_resolved(kind="anthropic"), timeout=5)
    backend._client = MagicMock()
    backend._client.beta.messages.stream.return_value = _fake_stream(_message("{}", "refusal"))
    resp = backend.chat.completions.create(model="claude-opus-5-5", messages=[])
    assert resp.choices[0].message.content is None
    assert resp.choices[0].finish_reason == "refusal"


# ============================================================
# HTTP routes
# ============================================================


@pytest.fixture
def client(monkeypatch):
    from claim2code_api.app.main import create_app
    from claim2code_api.llm import router as router_mod

    monkeypatch.setattr(router_mod, "get_settings", lambda: _settings(llm_management_allow_local=True))
    return TestClient(create_app(), client=("127.0.0.1", 50000))


def test_routes_list_select_and_reject_incomplete_choices(client) -> None:
    listed = client.get("/llm/providers").json()
    assert listed["active"]["provider_id"] == "env"
    assert {p["id"] for p in listed["presets"]} >= {"lmstudio", "anthropic", "openai", "custom"}

    bad = client.post("/llm/provider", json={"provider_id": "anthropic"})
    assert bad.status_code == 422 and "ANTHROPIC_API_KEY" in bad.json()["detail"]

    ok = client.post("/llm/provider", json={"provider_id": "ollama", "model": "qwen3:8b"})
    assert ok.status_code == 200
    assert ok.json()["active"] == {
        "provider_id": "ollama", "label": "Ollama (local)", "model": "qwen3:8b",
        "base_url": "http://localhost:11434/v1",
    }


def test_route_test_reports_failure_without_switching(client, monkeypatch) -> None:
    from claim2code_api.llm import router as router_mod

    backend = MagicMock()
    backend.ping.side_effect = ConnectionError("refused")
    monkeypatch.setattr(router_mod, "build_backend", lambda *_a, **_k: backend)

    res = client.post("/llm/provider/test", json={"provider_id": "ollama"}).json()
    assert res["ok"] is False and "refused" in res["error"]
    assert client.get("/llm/providers").json()["active"]["provider_id"] == "env"


@pytest.mark.parametrize("preset", ["openai", "anthropic", "gemini", "deepseek", "openrouter", "env"])
def test_saved_credentials_cannot_be_routed_to_another_server(preset) -> None:
    settings = _settings(
        openai_platform_api_key="SENTINEL_OPENAI", anthropic_api_key="SENTINEL_ANTHROPIC",
        gemini_api_key="SENTINEL_GEMINI", deepseek_api_key="SENTINEL_DEEPSEEK",
        openrouter_api_key="SENTINEL_OPENROUTER",
    )
    with pytest.raises(providers.ProviderError, match="cannot be overridden"):
        providers.resolve(settings, preset_id=preset, model="test", base_url="https://attacker.invalid/v1")


def test_custom_does_not_inherit_configured_or_other_endpoint_keys() -> None:
    settings = _settings(openai_platform_api_key="SENTINEL_CONFIGURED")
    assert providers.resolve(
        settings, preset_id="custom", model="test", base_url="https://custom.invalid/v1",
    ).api_key == "not-needed"
    providers.select(
        settings, preset_id="custom", model="test", base_url="https://first.invalid/v1",
        api_key="SENTINEL_SESSION",
    )
    assert providers.session_key(settings, "custom", "https://second.invalid/v1") == ""
    assert providers.select(
        settings, preset_id="custom", model="test", base_url="https://second.invalid/v1",
    ).api_key == "not-needed"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "https://user:password@host.invalid/v1",
    "https://host.invalid/v1?token=value", "https://host.invalid/v1#fragment",
    "http://remote.invalid/v1", "https://host.invalid:bad/v1",
])
def test_custom_url_validation(url) -> None:
    with pytest.raises(providers.ProviderError):
        providers.resolve(_settings(), preset_id="custom", model="test", base_url=url)


@pytest.mark.parametrize("preset", ["lmstudio", "ollama"])
def test_local_presets_only_accept_loopback_servers(preset) -> None:
    with pytest.raises(providers.ProviderError, match="loopback"):
        providers.resolve(_settings(), preset_id=preset, base_url="https://remote.invalid/v1")
    assert providers.resolve(
        _settings(), preset_id=preset, base_url="http://127.0.0.1:1234/v1",
    ).base_url == "http://127.0.0.1:1234/v1"


@pytest.mark.parametrize("path", ["/llm/provider", "/llm/provider/test"])
def test_provider_management_is_disabled_by_default(path, monkeypatch) -> None:
    from claim2code_api.app.main import create_app
    from claim2code_api.llm import router as router_mod

    monkeypatch.setattr(router_mod, "get_settings", _settings)
    backend = MagicMock()
    monkeypatch.setattr(router_mod, "build_backend", backend)
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    response = client.post(path, json={"provider_id": "ollama"})
    assert response.status_code == 403
    backend.assert_not_called()
    assert client.get("/llm/providers").json()["management"] == {
        "enabled": False, "auth_required": False,
    }


def test_local_opt_in_does_not_trust_forwarded_or_host_headers(monkeypatch) -> None:
    from claim2code_api.app.main import create_app
    from claim2code_api.llm import router as router_mod

    monkeypatch.setattr(router_mod, "get_settings", lambda: _settings(llm_management_allow_local=True))
    client = TestClient(create_app(), client=("203.0.113.10", 50000))
    response = client.post(
        "/llm/provider", json={"provider_id": "ollama"},
        headers={"host": "localhost", "x-forwarded-for": "127.0.0.1"},
    )
    assert response.status_code == 403


def test_admin_token_is_required_even_when_local_mode_is_enabled(monkeypatch) -> None:
    from claim2code_api.app.main import create_app
    from claim2code_api.llm import router as router_mod

    settings = _settings(llm_management_token="SENTINEL_ADMIN", llm_management_allow_local=True)
    monkeypatch.setattr(router_mod, "get_settings", lambda: settings)
    backend = MagicMock()
    monkeypatch.setattr(router_mod, "build_backend", lambda *_a, **_k: backend)
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    payload = {"provider_id": "ollama"}
    assert client.post("/llm/provider/test", json=payload).status_code == 401
    assert client.post(
        "/llm/provider/test", json=payload, headers={"Authorization": "Bearer incorrect"},
    ).status_code == 401
    backend.ping.assert_not_called()
    response = client.post(
        "/llm/provider/test", json=payload, headers={"Authorization": "Bearer SENTINEL_ADMIN"},
    )
    assert response.status_code == 200 and response.json()["ok"]
    assert "SENTINEL_ADMIN" not in client.get("/llm/providers").text
    assert client.get("/llm/providers").json()["management"] == {
        "enabled": True, "auth_required": True,
    }


def test_authorized_probe_rejects_endpoint_override_before_building_sdk(client, monkeypatch) -> None:
    from claim2code_api.llm import router as router_mod

    monkeypatch.setattr(router_mod, "get_settings", lambda: _settings(
        llm_management_allow_local=True, openai_platform_api_key="SENTINEL_SERVER",
    ))
    backend = MagicMock()
    monkeypatch.setattr(router_mod, "build_backend", backend)
    response = client.post("/llm/provider/test", json={
        "provider_id": "openai", "base_url": "https://attacker.invalid/v1",
    })
    assert response.status_code == 422
    assert "SENTINEL_SERVER" not in response.text
    backend.assert_not_called()
