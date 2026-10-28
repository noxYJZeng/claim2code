"""Public read-only demos refuse mutations before handlers or dependencies."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from claim2code_api.app import main
from claim2code_api.app.config import Settings
from claim2code_api.app.deps import get_db
from claim2code_api.llm import router as llm_router


@pytest.mark.parametrize("method,path", [
    ("POST", "/papers"), ("POST", "/repositories"),
    ("POST", "/llm/provider"), ("POST", "/llm/provider/test"),
    ("PUT", "/papers"), ("PATCH", "/repositories"), ("DELETE", "/papers"),
])
def test_read_only_blocks_mutations_before_dependencies(method, path, monkeypatch) -> None:
    settings = Settings(api_read_only=True, llm_management_token="SENTINEL_ADMIN")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
    app = main.create_app()
    database = MagicMock(side_effect=AssertionError("read-only request reached database"))
    backend = MagicMock(side_effect=AssertionError("read-only request reached provider"))
    app.dependency_overrides[get_db] = database
    monkeypatch.setattr(llm_router, "build_backend", backend)
    client = TestClient(app)
    response = client.request(
        method, path, json={"provider_id": "ollama"},
        headers={"Authorization": "Bearer SENTINEL_ADMIN"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "This API is read-only"
    database.assert_not_called()
    backend.assert_not_called()


def test_read_only_keeps_health_and_provider_listing_available(monkeypatch) -> None:
    settings = Settings(api_read_only=True, llm_management_token="SENTINEL_ADMIN")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_router, "get_settings", lambda: settings)
    client = TestClient(main.create_app())
    assert client.get("/health").json()["status"] == "ok"
    listed = client.get("/llm/providers")
    assert listed.status_code == 200
    assert listed.json()["management"]["enabled"] is False
    assert "SENTINEL_ADMIN" not in listed.text
