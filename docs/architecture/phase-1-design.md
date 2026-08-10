# Phase 1 — Project Foundation Design

> Goal (from `phase-roadmap.md:33-47`): bring up the full local stack
> with a working health endpoint. Acceptance: `docker compose up`
> starts the stack, `curl localhost:8000/health` returns 200, web
> loads at `localhost:5173`, and backend lint + test + frontend
> build all run in CI.

## 1. What ships

| Layer | Change |
|---|---|
| `apps/api/pyproject.toml` | Real runtime + dev dependencies |
| `apps/api/claim2code_api/app/main.py` | FastAPI factory with `/health`, `/health/ready`, CORS |
| `apps/api/claim2code_api/app/config.py` | `pydantic-settings` `Settings` reading `.env` from project root |
| `apps/api/claim2code_api/app/deps.py` | `get_db()` session dependency |
| `apps/api/claim2code_api/db/{base,session,models}.py` | SQLAlchemy 2.0 declarative base, lazy engine, `AnalysisState` ORM model |
| `apps/api/alembic.ini` + `migrations/{env,script}.*` + `versions/0001_init.py` | First migration: `CREATE EXTENSION vector` + `analysis_states` table |
| `apps/api/tests/` | `test_health.py` (unit, monkeypatched DB probe) + `integration/test_db_migrations.py` (real DB) |
| `apps/web/` | Tailwind, `src/api.ts` (typed `getHealth`), `App.tsx` (online/offline badge), Vitest, ESLint flat config |
| `deployments/Dockerfile.api` | Single `COPY apps/api` + `pip install` (no dev extras in the runtime image) |
| `.github/workflows/ci.yml` | Four jobs: `scaffold-smoke`, `backend-lint`, `backend-test`, `frontend-build` |
| `apps/api/README.md` | Local dev + test + migration commands |

## 2. Key decisions

### 2.1 Synchronous engine, not async

Phase 1 uses SQLAlchemy 2.0 in **synchronous** mode. The verifier, paper
parser, repo cloner, and embedding client are all blocking I/O, and
SQLAlchemy 2.0's sync engine handles the small workload (a handful of
analyses per day in dev) with no measurable cost. Switching to
`AsyncEngine` later is straightforward: replace `get_engine()` and
`get_session_local()`, add `await` in the few call sites. We do **not**
introduce that complexity in Phase 1 to keep the migration code path
linear.

### 2.2 `psycopg[binary]` v3

The v3 driver is the SQLAlchemy 2.0 default, ships its own libpq build
in the `[binary]` extra (no system `libpq` needed), and is async-ready
if we ever flip the engine. `connect_timeout=5` is set on the engine so
`/health/ready` cannot hang.

### 2.3 Lazy engine

`get_engine()` constructs the engine on first call, not on import. That
lets the API boot in environments where the database is briefly
unreachable (e.g. a fresh `docker compose up`). The engine is then
cached for the rest of the process. `reset_for_tests()` exists for the
test suite to swap `DATABASE_URL` between cases.

### 2.4 Split health probes

`/health` is a pure liveness probe (always 200, no I/O). `/health/ready`
runs `SELECT 1` and returns 503 with a JSON body when the DB is
unreachable. This split matches what Kubernetes, Docker healthchecks,
and most uptime probes expect, and it lets the orchestrator keep the
container alive through transient DB blips while still routing traffic
away when the backend is degraded.

### 2.5 Wide JSONB columns

`analysis_states.claims|symbols|mappings|verifications|errors` are
JSONB. The shapes of those nested objects will only stabilise after
Phases 2-6. JSONB lets us evolve them without DDL churn. Once stable
(likely Phase 6 or 10), we will split them into proper tables with
typed columns and foreign keys to `analysis_states.analysis_id`.

### 2.6 DB probe stubbed in unit tests

The health-endpoint unit tests don't need a real database. We
`monkeypatch` `claim2code_api.app.main.check_db_ready` with two
fixtures (`db_ready`, `db_unready`) to cover the 200 and 503 paths. The
integration test (`tests/integration/test_db_migrations.py`) hits a
real Postgres to confirm the migration applied and the `vector`
extension is installed. CI runs both.

### 2.7 CORS limited to the dev port

`Settings.cors_origins` defaults to `["http://localhost:5173"]` only.
The list is configurable via env var; production deployments will need
to set it to the real web origin. The allowed methods are restricted
to `GET, POST, OPTIONS` to keep the surface narrow while we are still
designing the API.

### 2.8 Frontend test framework: Vitest

Vitest is the natural pick for a Vite project (shared config, no
separate transpiler), and `@testing-library/react` is the de-facto
React testing API. jsdom provides the DOM. We do **not** introduce
`react-router` or a state library in Phase 1 — the welcome page is a
single component and adding them now would be premature.

## 3. CI design

| Job | Triggers | What it proves |
|---|---|---|
| `scaffold-smoke` | every push / PR | Project shape intact, compose + `.env.example` valid |
| `backend-lint` | every push / PR | ruff + mypy clean |
| `backend-test` | every push / PR | pytest unit + integration pass; migration applies; `vector` extension installs |
| `frontend-build` | every push / PR | ESLint, `tsc -b`, `vite build`, Vitest all succeed |

The `backend-test` job uses GitHub Actions' built-in `services.postgres`
with `pgvector/pgvector:pg16` (same image as `docker-compose.yml`) and
a `DATABASE_URL` env var, so the integration test does not need
testcontainers. A 30-second `pg_isready` wait guards against the
service starting before the migration runs.

## 4. What we explicitly do NOT do in Phase 1

- ❌ No OpenTelemetry setup (Phase 9)
- ❌ No LLM client / OpenAI wiring (Phase 2)
- ❌ No PDF parsing or claim extraction (Phase 2)
- ❌ No GitHub cloning or AST walking (Phase 3)
- ❌ No embeddings or pgvector `code_embeddings` table (Phase 4)
- ❌ No retrieval ranker (Phase 5)
- ❌ No verifier (Phase 6)
- ❌ No literature search (Phase 7)
- ❌ No safety guardrails (Phase 8)
- ❌ No Prettier (Phase 1 only needs ESLint; formatting lands in a polish pass)
- ❌ No Docker image push (image build verified locally; registry push lands when needed)
- ❌ No `react-router-dom` (single page; added when Phase 2 introduces navigation)

## 5. Acceptance — the command list

```bash
# 1. backend unit + integration
cd apps/api
pip install -e ".[dev]"
ruff check . && mypy .
pytest -v                # unit (no DB) + integration (needs DATABASE_URL)

# 2. database migration
docker compose -f docker-compose.yml up -d postgres
DATABASE_URL=postgresql://claim2code:change_me_local_dev@localhost:5432/claim2code \
  alembic upgrade head

# 3. full stack
cp .env.example .env       # fill POSTGRES_PASSWORD
bash scripts/dev-up.sh
curl -sS http://localhost:8000/health         # {"status":"ok","version":"0.1.0"}
curl -sS http://localhost:8000/health/ready  # {"status":"ok","db":true}
curl -sS http://localhost:5173/               # HTML with "Claim2Code" + "online" badge

# 4. frontend
cd apps/web
npm ci
npm run lint && npm run build && npm test
```

A green run of all four blocks is the Phase 1 done-state.
