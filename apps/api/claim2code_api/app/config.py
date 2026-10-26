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

# apps/api/claim2code_api/app/config.py → the project root (with
# ``.env``) is 4 levels up in a checkout. Search upward for ``.env``
# rather than hard-indexing; fall back to ``parents[3]`` (``/app`` in
# the Docker image, which gets its config from env vars instead).
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = next(
    (p for p in _HERE.parents if (p / ".env").is_file()),
    _HERE.parents[3],
)
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
    # Per-call chat timeout. 60s suits hosted gpt-4o; a local model
    # (e.g. LM Studio) prefilling a whole paper needs several minutes.
    llm_request_timeout_seconds: float = 60.0

    # ---- Web (Vite reads these at build/dev time; declared here so a
    # shared ``.env`` does not trip extra="forbid") ----
    web_port: int = 5173
    vite_api_base_url: str = "http://localhost:8000"

    # ---- GitHub (Phase 3; rate-limit token for popular repos) ----
    github_token: str = ""

    # ---- Phase 7: Related Work / literature ----
    # Optional API key for Semantic Scholar's Graph API. Without a
    # key the rate limit is ~1 req/s; with a key it jumps to
    # 100 req/5min via the ``x-api-key`` header.
    # Get a key at https://www.semanticscholar.org/product/api.
    semantic_scholar_api_key: str | None = None

    # ---- Safety limits (Phase 3; gating repo ingestion) ----
    max_repo_mb_megs: int = 200
    max_repo_files: int = 20000
    max_file_kb: int = 512

    # ---- Phase 3 repo ingestion knobs ----
    # File extensions the parser walks (mirrors
    # ``claim2code_api.reps.parser._EXTENSION_MAP``).
    repo_parse_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",
            ".ts",
            ".tsx",
            ".java",
            ".go",
            ".rs",
        ]
    )
    # Directory basenames the walk skips — see
    # :func:`claim2code_api.repos.storage.walk_repo_files`.
    repo_parse_ignore_dirs: list[str] = Field(
        default_factory=lambda: [
            "node_modules",
            ".venv",
            "__pycache__",
            "dist",
            "build",
            "target",
            ".gradle",
            ".idea",
            ".vscode",
            "venv",
            "env",
            # Vendored third-party code: indexing it floods retrieval
            # with symbols the paper's authors did not write.
            "third_party",
            "vendor",
        ]
    )
    # HTTP timeout for the GitHub HEAD pre-flight size guard.
    repo_size_estimate_timeout_seconds: float = 5.0

    # ---- Paper ingestion (Phase 2) ----
    max_pdf_size_mb: int = 50

    # ---- Observability (Phase 9) ----
    otel_service_name: str = "claim2code"
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_console: bool = True

    # ---- Workflow timeouts ----
    workflow_timeout_paper: int = 180
    workflow_timeout_repo: int = 300
    workflow_timeout_total: int = 900

    # ---- Phase 4 — semantic indexing (per-symbol embeddings) ----
    # How many strings to embed per OpenAI batch call. The OpenAI API
    # accepts up to 2 048 inputs per request; 128 is conservative.
    embed_batch_size: int = 128
    # Retries on transient errors (timeout, connection, 5xx). The
    # OpenAI SDK retries 401/429 itself; this is the wrapping loop.
    embed_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    The cache lets tests call ``get_settings.cache_clear()`` to pick up
    changes to the environment between test cases.
    """
    return Settings()
