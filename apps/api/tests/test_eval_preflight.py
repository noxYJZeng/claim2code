"""evaluation/preflight.py — report every check, fail on any."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import preflight  # noqa: E402


def test_runs_every_check_and_fails_on_any(capsys) -> None:
    def boom() -> tuple[bool, str]:
        raise RuntimeError("connection\n  refused")

    ready = preflight.run([
        ("ok", lambda: (True, "fine")),
        ("raises", boom),
        ("fails", lambda: (False, "model not loaded")),
    ])
    out = capsys.readouterr().out.splitlines()
    assert not ready
    assert out == [
        "  ✓ ok             fine",
        "  ✗ raises         RuntimeError: connection refused",
        "  ✗ fails          model not loaded",
    ]
    assert preflight.run([("ok", lambda: (True, ""))])


def test_endpoints_print_without_credentials_or_paths() -> None:
    assert preflight._host("https://user:secret@api.example.com:8443/v1?key=x") == "https://api.example.com:8443"
    assert preflight._host("") == "the provider's default endpoint"


def test_an_unlisted_model_is_reported_with_what_is_served(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_served_models", lambda *_: {"qwen", "nomic"})
    assert preflight._not_served("qwen", "http://localhost:1234/v1", "k") is None
    assert preflight._not_served("gpt", "http://localhost:1234/v1", "k") == (
        "gpt is not served there; available: nomic, qwen"
    )
    monkeypatch.setattr(preflight, "_served_models", lambda *_: None)
    assert preflight._not_served("gpt", "http://localhost:1234/v1", "k") is None
