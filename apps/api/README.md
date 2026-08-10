# Claim2Code API

FastAPI backend for the paper-to-code verification agent.

## Local dev (no Docker)

```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# From the project root, with .env present:
PYTHONPATH=/path/to/claim2code/apps/api uvicorn claim2code_api.app.main:app --reload
```

The `DATABASE_URL` in `.env` must point at a running Postgres+pgvector
(see `docker-compose.yml` at the project root for the canonical local DB).

## Layout

```
claim2code_api/
├── app/                 # FastAPI app, config, dependency wiring
│   ├── main.py          # create_app(), /health, /health/ready, CORS
│   ├── config.py        # pydantic-settings Settings (loads .env)
│   └── deps.py          # get_db()
├── db/                  # SQLAlchemy session + ORM models
│   ├── base.py
│   ├── session.py       # engine (lazy), SessionLocal, check_db_ready()
│   └── models.py        # AnalysisState
├── paper/               # Phase 2: PyMuPDF parsing + claim extraction
├── repository/          # Phase 3: GitHub clone + AST
├── retrieval/           # Phase 4-5: embeddings + hybrid ranker
├── verification/        # Phase 6: verifier + discrepancy classifier
├── literature/          # Phase 7: scholarly search + comparison
├── workflow/            # Phase 9: orchestrated background workflow
├── llm/                 # Phase 2: LLM client (provider abstraction)
├── observability/       # Phase 9: OpenTelemetry setup
└── safety/              # Phase 8: secret redaction + injection defense
```

## Tests

```bash
cd apps/api
pytest                 # unit tests (no DB required)
pytest -m integration  # integration tests (require docker / testcontainers)
```

## Migrations

```bash
cd apps/api
alembic upgrade head       # apply
alembic revision -m "..."  # create new (autogenerate is wired)
```

## Endpoints (Phase 1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe — always 200 |
| GET | `/health/ready` | Readiness probe — 200 if DB `SELECT 1` succeeds, else 503 |
| GET | `/docs` | OpenAPI / Swagger UI |
| GET | `/redoc` | ReDoc |
