"""Process-wide configuration loaded from environment variables / .env file.

`pydantic-settings` reads from the process environment and from a ``.env``
file. The ``.env`` location is resolved relative to this module so the
backend works the same whether launched from ``apps/api`` (local dev) or
from the project root (Docker, CI).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/claim2code_api/app/config.py → project root is 3 levels up.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime configuration for the API process."""

    model_config = SettingsConfigDict(
        env_file=str(_DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- API server ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: str = "info"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ---- Database ----
    database_url: str = (
        "postgresql://claim2code:change_me_local_dev@localhost:5432/claim2code"
    )
    db_health_timeout_seconds: float = 5.0

    # ---- LLM (placeholder; not used in Phase 1) ----
    openai_api_key: str = "sk-replace-me"
    openai_base_url: str = "https://api.openai.com/v1"

    # ---- Workflow ----
    workflow_timeout_total: int = 900


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    The cache lets tests call ``get_settings.cache_clear()`` to pick up
    changes to the environment between test cases.
    """
    return Settings()
