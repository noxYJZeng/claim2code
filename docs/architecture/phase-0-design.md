# Phase 0 — Scaffold Design

> Goal: a single commit that puts down the monorepo skeleton, the
> spec-derived documentation, and the CI baseline. Nothing runs yet —
> we just want a green starting point.

## 1. What we're building (and why each piece exists)

| File / dir | Purpose |
|---|---|
| `.gitignore` | Excludes `interview-prep/` (lives in `~/Desktop/interview-prep/claim2code/`), `.env`, builds, runtime state |
| `README.md` | Project landing page; v1 capability list + stack + quick-start + phase roadmap |
| `docs/specs/spec-summary.md` | One-page condensation of `Claim2Code.pdf` so we don't re-read 41 pages each session |
| `docs/architecture/data-flow.md` | End-to-end pipeline picture + the data classes that flow through |
| `docs/architecture/phase-roadmap.md` | 11 phases (0–10), each with tag, deliverables, acceptance criteria |
| `apps/api/` (empty Python package) | Placeholder for FastAPI backend |
| `apps/web/` (empty placeholder) | Placeholder for React + Vite frontend |
| `evaluation/` (empty dir) | Placeholder for golden datasets, baselines, eval scripts |
| `scripts/` | `dev-up.sh`, `dev-down.sh`, `bootstrap.sh` |
| `deployments/` | `docker-compose.yml` (postgres+pgvector) |
| `.github/workflows/ci.yml` | CI baseline (placeholder `echo ok`) |
| `.env.example` | Documents all required env vars |
| `Makefile` | Single entry point: `make dev`, `make test`, `make lint`, `make help` |

## 2. Data flow at a glance (full picture in `data-flow.md`)

```
       User (PDF + URL)
              │
              ▼
       FastAPI backend
              │
              ├──► Paper Processor ──► Claims
              │
              ├──► Repo Analyzer ──► CodeSymbols
              │
              └──► Hybrid Retriever ──► Verifier ──► Report
```

## 3. The artifact-atlas pattern we're mirroring

| artifact-atlas | claim2code |
|---|---|
| `apps/api/` (Go) | `apps/api/` (Python + FastAPI) |
| `apps/web/` (React + Vite) | `apps/web/` (React + Vite + Tailwind) |
| `connectors/met/` | `apps/api/app/repository/` |
| `docs/architecture/ingestion.md` | `docs/architecture/data-flow.md` |
| `docs/specs/initial-schema-proposed.sql` | `docs/specs/spec-summary.md` |
| `Makefile` (single entry) | `Makefile` (single entry) |
| `.gitignore` (excludes backups, .env, volumes) | `.gitignore` (same + `interview-prep/`) |
| `interview-prep/` outside repo | `~/Desktop/interview-prep/claim2code/` |

## 4. Conventions borrowed from artifact-atlas

- **One commit per phase**, prefix tag matches scope
- **`main` is always shippable**; phase branches merge back with `--no-ff`
- **`Makefile` is the single entry** — `make help` lists everything
- **`.env.example` only** — never commit real `.env`
- **Conventional commits**: `feat(...)`, `feat(api)`, `feat(web)`,
  `fix(...)`, `docs(...)`, `chore(...)`, `polish(...)`, `refactor(...)`,
  `ci(...)`

## 5. What we're explicitly NOT doing in Phase 0

- ❌ Writing any FastAPI code (that's Phase 1)
- ❌ Writing any React code (that's Phase 1)
- ❌ Writing any Python package code (that's Phase 1+)
- ❌ Installing any dependencies (Phase 1+)
- ❌ Setting up pgvector (Phase 1)
- ❌ Connecting to any external API (Phase 2+)

## 6. Acceptance

```bash
cd ~/Desktop/claim2code
git log --oneline    # shows one commit
ls                   # README.md, .gitignore, Makefile, apps/, docs/, …
cat .gitignore | grep interview-prep   # excluded
make help            # shows available targets
docker compose config # validates (if docker available)
```

That's it for Phase 0. Green baseline established; Phase 1 brings up the actual stack.