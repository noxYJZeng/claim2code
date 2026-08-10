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
        # ``forbid`` surfaces env-var typos (e.g. ``DATBASE_URL``) at process
        # boot instead of silently swallowing them. The Settings model must
        # therefore declare every env var that may appear in ``.env``; see
        # ``.env.example`` for the full set.
        extra="forbid",
    )

    # ---- API server ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: str = "info"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ---- Database ----
    # The ``+psycopg`` driver suffix tells SQLAlchemy to use the psycopg
    # v3 driver (which we declare in pyproject.toml). Omitting the suffix
    # would make SQLAlchemy default to psycopg2, which is not installed.
    database_url: str = (
        "postgresql+psycopg://claim2code:change_me_local_dev@localhost:5432/claim2code"
    )
    db_health_timeout_seconds: float = 5.0

    # ---- Postgres connection parts (interpolation vars for docker-compose;
    # the API process itself reads only ``database_url``) ----
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "claim2code"
    postgres_user: str = "claim2code"
    postgres_password: str = "change_me_local_dev"

    # ---- LLM (placeholder; consumed in Phase 2 by ClaimExtractor) ----
    openai_api_key: str = "sk-replace-me"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_chat: str = "gpt-4o"
    openai_model_embed: str = "text-embedding-3-small"
    openai_embed_dim: int = 1536

    # ---- Web (Vite reads these at build/dev time; declared here so a
    # shared ``.env`` does not trip extra="forbid") ----
    web_port: int = 5173
    vite_api_base_url: str = "http://localhost:8000"

    # ---- GitHub (Phase 3; rate-limit token for popular repos) ----
    github_token: str = ""

    # ---- Safety limits (Phase 3; gating repo ingestion) ----
    max_repo_mb_megs: int = 200
    max_repo_files: int = 20000
    max_file_kb: int = 512

    # ---- Observability (Phase 9) ----
    otel_service_name: str = "claim2code"
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_console: bool = True

    # ---- Workflow timeouts ----
    workflow_timeout_paper: int = 180
    workflow_timeout_repo: int = 300
    workflow_timeout_total: int = 900


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    The cache lets tests call ``get_settings.cache_clear()`` to pick up
    changes to the environment between test cases.
    """
    return Settings()
