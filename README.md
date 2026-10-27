# Claim2Code

> AI Research Agent for Paper–Code Verification and Reproducibility Analysis.
>
> Upload a paper PDF + a GitHub URL, get a grounded verification report:
> every claim traced to a paper section and a source-code location, with
> an explicit `VERIFIED / PARTIALLY_VERIFIED / CONTRADICTED / UNVERIFIED`
> verdict and detected discrepancies.

[![status](https://img.shields.io/badge/status-Phase%2010%20(evaluation)-blue)]()
[![stack](https://img.shields.io/badge/stack-Python%20%7C%20React%20%7C%20Postgres-blue)]()

## What's in v1

- 📄 **Paper Intelligence** — PDF upload → structured claims with paper-evidence spans
- 🧠 **Repository Intelligence** — GitHub clone → AST symbols → config files
- 🔎 **Hybrid Claim-to-Code Retrieval** — semantic + lexical + structural, weighted ensemble
- ✅ **Verification** — every claim gets a verdict + paper evidence + code evidence
- ⚠️ **Discrepancy Detection** — hyperparameter / objective / dataset / metric mismatches
- 📚 **Related Work** — selected claim → 5 structured comparable papers
- 🛡️ **Responsible-AI Guardrails** — secret redaction, prompt-injection defense, no code execution
- 📊 **Observability** — OpenTelemetry traces + per-step latency/token/retrieval metrics
- 🧪 **Evaluation Harness** — hand-annotated paper/repo pairs; retrieval Recall@1/5 + MRR, verification accuracy + discrepancy precision/recall + grounding rate, safety redaction/detection rates ([measured results](#measured-results))

## Measured results

From [`evaluation/report.md`](evaluation/report.md), measured locally with
`qwen3.8-27b-mlx` (LM Studio) and `nomic-embed-text-v1.5` embeddings on the
5 annotated pairs whose repos are indexed (DDIM, DPO, LoRA, MAE, Reflexion;
44 claims). Re-run with `make eval-all`.

| Area | Metric | Claim2Code | Baseline |
|---|---|---|---|
| Retrieval (top 5) | Recall@5, macro avg | **0.70** | 0.61 embedding-only · 0.49 lexical-only |
| | MRR, macro avg | **0.487** | 0.383 embedding-only · 0.315 lexical-only |
| Verification | Status accuracy (4-way) | 77.3% | — |
| | Support rate (claim judged VERIFIED / PARTIALLY_VERIFIED) | 84.1% | — |
| | Discrepancy precision / recall (claim level) | 57.1% / 66.7% | — |
| | Grounding rate (cited code exists in the cited file) | 100% of 111 citations | — |
| | Line accuracy (cited code at the cited lines) | 91.0% | — |
| Safety | Secrets protected, covered / held-out forms | 10/10 · 2/8 | — |
| | Injections detected, covered / held-out forms | 6/6 · 0/6 | — |
| | Clean text falsely flagged | 3/12 | — |

Caveats: 5 pairs / 44 claims is below the 8–12-pair target, and only 6 of the
44 claims carry an annotated discrepancy, so the discrepancy rates rest on few
cases. The annotations were written during development, not by an independent
annotator, and five were corrected after a first measured run (2026-10-27,
each with a note in its annotation): lora_c3/c5/c6 named the wrong symbol or
misstated the code, and ddim_c1/mae_c5 were labelled PARTIALLY_VERIFIED with no
paper/code mismatch. Scored against the corrected labels, the earlier pipeline
reached 63.6% accuracy and 36.4% / 66.7% discrepancy precision / recall.
The remaining misses are mostly retrieval (5 claims got no code evidence, 3 of
them in LoRA, whose `examples/NLU` is a vendored transformers checkout) and
3 PARTIALLY_VERIFIED verdicts on claims annotated VERIFIED. The safety
`held_out` cases (GitHub fine-grained / Slack / Hugging Face tokens,
paraphrased injections) are coverage gaps of the regex gate, which matches only
the forms listed in Phase 8.

## Why

ML papers say "Algorithm 1", "β = 0.1", "trained on Qwen2.5-7B for 3 epochs". The repo
might implement a different β, a different model, or skip the algorithm entirely.
**Reading both by hand is slow and error-prone.** Claim2Code automates the cross-check,
gives every conclusion a citation, and admits `UNVERIFIED` when evidence is missing.

## Stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript + Vite + Tailwind |
| Backend | Python 3.11+ / FastAPI / Pydantic |
| Workflow | Microsoft Agent Framework (preferred) or explicit stateful executor |
| LLM | Azure OpenAI or OpenAI-compatible endpoint, behind `LLMClient` interface |
| Embeddings | Same provider abstraction as LLM |
| PDF | PyMuPDF (digitally generated academic PDFs only — no OCR in v1) |
| Code Parsing | Tree-sitter (preferred) or Python `ast` |
| Database | PostgreSQL + pgvector |
| Cache | FastAPI BackgroundTasks (v1); Redis optional later |
| Observability | OpenTelemetry + structured logging |
| Repo integration | GitHub REST API + `git clone` (public repos only) |

## Quick start (5 min)

```bash
# 1. clone
git clone git@github.com:noxYJZeng/claim2code.git
cd claim2code

# 2. configure
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD and OPENAI_API_KEY (the LLM is
# required by Phase 2's claim extractor).

# 3. start stack
make dev          # brings up postgres+pgvector, api, web

# 4. open
#   Web:           http://localhost:5173
#   API health:    http://localhost:8000/health
#   API ready:     http://localhost:8000/health/ready
#   API docs:      http://localhost:8000/docs
```

**Verify the foundation + paper pipeline:**

```bash
curl -sS http://localhost:8000/health        # {"status":"ok","version":"0.1.0"}
curl -sS http://localhost:8000/health/ready # {"status":"ok","db":true}

# Upload a paper — runs parse + LLM extract synchronously.
curl -X POST -F "file=@samples/attention.pdf" \
    http://localhost:8000/papers
# {"analysis_id":"...","paper_id":"...","parse_status":"done","n_claims":7}

# Fetch the extracted claims.
curl -sS http://localhost:8000/analyses/<analysis_id>/claims | jq '.claims | length'
# 7
```

## Getting Started

> **The "Quick start" above assumes Docker Compose.** The instructions below
> walk through a fully manual setup that works on **any** machine — including
> one without the Docker Compose plugin (which is the case on most dev
> laptops / CI runners / bare-metal servers).
>
> If you do have Docker + Compose v2 installed, you can skip this section
> and go straight back to "Quick start".

### 1. Prerequisites

| Tool | Required? | Install (macOS) | Install (Ubuntu) |
|---|---|---|---|
| **Python 3.11+** | YES | `brew install python@3.11` | `sudo apt install python3.11 python3.11-venv` |
| **PostgreSQL 16 + pgvector** | YES (for paper pipeline, repo indexing, retrieval) | `brew install postgresql@16 pgvector` | `sudo apt install postgresql-16 postgresql-16-pgvector` |
| **OpenAI API key** | YES (for `gpt-4o` + `text-embedding-3-small`) | https://platform.openai.com/api-keys | same |
| **Node 20+** | YES (only if you want the React frontend) | `brew install node` | `nvm install 20` |
| **Semantic Scholar API key** | optional (raises rate-limit 1/s → 100/5min) | https://www.semanticscholar.org/product/api | same |
| **GitHub token** | optional (raises REST rate limit on popular repos) | https://github.com/settings/tokens | same |
| **Docker + Compose v2** | OPTIONAL — only needed for `make dev` | https://docs.docker.com/desktop/install/mac-install/ | https://docs.docker.com/engine/install/ubuntu/ |

### 2. Clone + venv + pip install

```bash
git clone https://github.com/noxYJZeng/claim2code.git
cd claim2code

# Python venv — pinned to 3.11 (project requires >=3.11, pyproject.toml:12)
python3.11 -m venv /tmp/c2c-venv
/tmp/c2c-venv/bin/pip install -U pip setuptools wheel

# Backend deps + the `[dev]` extra (pytest, ruff, mypy)
cd apps/api && /tmp/c2c-venv/bin/pip install -e ".[dev]" && cd ../..

# Frontend deps (skip if you don't need the UI)
cd apps/web && pnpm install   # or `npm install`
cd ../..
```

> **Why `/tmp/c2c-venv`?** Keeps the venv outside the repo so it doesn't
> get committed, and outside `.volumes/` so a `make reset` doesn't
> delete it. Use any path you like — just keep it out of the repo tree.

### 3. Configure `.env`

`pydantic-settings` reads `.env` from the **project root**
(`apps/api/claim2code_api/app/config.py:18-19`), not from `apps/api/`.
The `Settings` model uses `extra="forbid"`, so every variable must
match a declared field — typos surface as a `ValidationError` at boot
instead of silently defaulting.

```bash
cat > .env <<'EOF'
# ---- Postgres ----
DATABASE_URL=postgresql+psycopg://claim2code:devpw@localhost:5432/claim2code

# ---- LLM ----
OPENAI_API_KEY=sk-...                                  # required
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_CHAT=gpt-4o
OPENAI_MODEL_EMBED=text-embedding-3-small
OPENAI_EMBED_DIM=1536

# ---- Optional ----
SEMANTIC_SCHOLAR_API_KEY=                              # empty = anonymous mode
GITHUB_TOKEN=                                          # empty = public rate limit

# ---- Safety limits (Phase 8) ----
MAX_REPO_MB_MEGS=200
MAX_REPO_FILES=20000
MAX_FILE_KB=512
MAX_PDF_SIZE_MB=50

# ---- API + CORS ----
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=info
CORS_ORIGINS=["http://localhost:5173"]

# ---- Phase 9 OpenTelemetry ----
OTEL_SERVICE_NAME=claim2code
OTEL_EXPORTER_CONSOLE=true                             # dev: print spans to stdout
OTEL_EXPORTER_OTLP_ENDPOINT=                           # empty = console only; set in prod

# ---- Workflow timeouts (seconds) ----
WORKFLOW_TIMEOUT_PAPER=180
WORKFLOW_TIMEOUT_REPO=300
WORKFLOW_TIMEOUT_TOTAL=900
EOF
```

Every name in the template maps to a field declared at
`apps/api/claim2code_api/app/config.py:22-142`. If you add a name
that doesn't match, the API will refuse to start.

### 4. Bring up Postgres + enable pgvector (manual, no Docker)

```bash
# macOS — Homebrew's postgres runs as your macOS user (not `postgres`)
brew services start postgresql@16
createdb claim2code

# Ubuntu — peer-auth as the `postgres` OS user
sudo systemctl start postgresql
sudo -u postgres createuser -s claim2code
sudo -u postgres createdb -O claim2code claim2code

# BOTH platforms: enable pgvector in the database
psql claim2code -c "CREATE EXTENSION IF NOT EXISTS vector;"

# BOTH platforms: set the password the .env template expects
sudo -u postgres psql -c "ALTER USER claim2code WITH PASSWORD 'devpw';"
```

If `CREATE EXTENSION vector` fails with
`could not open extension control file`, the pgvector package isn't
installed (see step 1) — install it, then retry.

### 5. Run migrations

```bash
cd apps/api && /tmp/c2c-venv/bin/alembic upgrade head && cd ../..
```

Alembic config lives at `apps/api/alembic.ini`; revisions live at
`apps/api/alembic/versions/` (9 revisions at HEAD as of Phase 9:
`0001_init` through `0009_latest`).

### 6. Run the dev server (two processes)

The Docker path runs everything in one container; the manual path runs
the API and the frontend in two terminal windows.

**Backend** — terminal 1:

```bash
cd apps/api
/tmp/c2c-venv/bin/uvicorn claim2code_api.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend** — terminal 2 (optional, only if you want the UI):

```bash
cd apps/web
pnpm dev       # or `npm run dev`
```

Open:

- API root:        http://localhost:8000
- API health:      http://localhost:8000/health
- API ready (DB):  http://localhost:8000/health/ready
- API docs:        http://localhost:8000/docs
- Web UI:          http://localhost:5173

### 7. Run the test suite

`apps/api/pyproject.toml` line 85 sets
`addopts = "-m 'not integration and not safety and not observability'"`,
so plain `pytest` runs **only unit tests** and skips the opt-in
marker suites.

```bash
# Backend default suite (unit tests, ~250 tests, <30s)
/tmp/c2c-venv/bin/pytest          # from apps/api/
# or
make test                         # from project root

# Per-marker opt-in suites
cd apps/api && /tmp/c2c-venv/bin/pytest -m safety          # 57 adversarial safety tests
cd apps/api && /tmp/c2c-venv/bin/pytest -m observability    # 29 OTel tracing/metrics tests
cd apps/api && /tmp/c2c-venv/bin/pytest -m integration      # opt-in (needs testcontainers)

# Lint
make lint                          # ruff + mypy on backend, eslint on frontend

# Frontend unit tests
cd apps/web && pnpm test           # vitest + @testing-library/react, no backend needed
```

### 8. Run the evaluators

Each evaluator writes `evaluation/results/<name>.json`; `make eval-report`
folds them into `evaluation/results.json` + `evaluation/report.md`.
Repositories are matched to annotation pairs by URL, so index a pair's repo
first (pairs without an indexed repo are skipped).

| Script | Make target | Needs | Measures |
|---|---|---|---|
| `evaluation/evaluate_retrieval.py` | `make eval-retrieval` | DB + embeddings | Recall@1/5 + MRR for embedding-only, lexical-only and hybrid |
| `evaluation/evaluate_verification.py` | `make eval-verification` | DB + embeddings + chat model | status accuracy, support rate, discrepancy P/R, grounding, line accuracy |
| `evaluation/evaluate_safety.py` | `make eval-safety` | nothing | secret protection, injection detection, false positives on `datasets/safety_corpus.json` |
| `evaluation/report.py` | `make eval-report` | the results above | `results.json` + `report.md` |
| `evaluation/evaluate_all.py` | `make eval-all` | all of the above | runs every evaluator, then the report |

```bash
make eval-safety                       # seconds, no services needed
make eval-retrieval                    # ~1 min
make eval-verification                 # one LLM call per claim (~1-2 h for 44 claims locally)
make eval-verification ARGS=--resume   # continue an interrupted run
make eval-report                       # render evaluation/report.md
```

`python evaluation/evaluate_verification.py --regrade` re-scores the stored
citations of the last run without calling the LLM.

### 9. End-to-end demo with curl

After the backend is running on `:8000` and you've got at least one
indexed repository (see `docs/architecture/phase-3-design.md` for the
`POST /repositories/` clone + index pipeline):

```bash
# 1. Upload a PDF (synchronous: ~30s for a 20-page PDF)
curl -X POST http://localhost:8000/papers/ -F "file=@/path/to/paper.pdf"
# → {"analysis_id": "<UUID>", "paper_id": "<UUID>",
#    "parse_status": "done", "n_claims": 7, ...}

# 2. Read extracted claims
curl -sS http://localhost:8000/analyses/<analysis_id>/claims \
  | jq '.claims | length'
# → 7

# 3. Bind an indexed repository
#    (POST /repositories/ returns {"repository_id": "<UUID>", ...})
REPO_ID=<repo-uuid>

# 4. Run verification (LLM verdict per claim against repo symbols)
curl -sS -X POST http://localhost:8000/analyses/<analysis_id>/verifications \
  -H "Content-Type: application/json" \
  -d "{\"repository_id\": \"$REPO_ID\"}"
# → {"verifications": [
#      {"claim_id": "...", "status": "VERIFIED",     "paper_evidence": [...], "code_evidence": [...]},
#      {"claim_id": "...", "status": "CONTRADICTED",  "discrepancies": [{"type": "hyperparameter_mismatch", ...}]},
#      {"claim_id": "...", "status": "UNVERIFIED",   ...},
#      ...
#    ]}

# 5. Run related-work (5 comparable papers + 6-field LLM compare each)
curl -sS -X POST http://localhost:8000/analyses/<analysis_id>/related-work/<claim_id>
# → {"comparisons": [
#      {"paper_id": "...", "problem": "...", "method": "...",
#       "dataset": "...", "metric": "...", "difference": "...", "agreement": "..."},
#      ...
#    ]}
```

The 4 verdict statuses (`VERIFIED` / `PARTIALLY_VERIFIED` /
`CONTRADICTED` / `UNVERIFIED`) and the structural guard
`paper_evidence AND code_evidence OR status==UNVERIFIED` come from
Phase 6 — see `docs/architecture/phase-6-design.md` §2 for the
contract.

### 10. Troubleshooting

1. **`could not open extension control file`** when running
   `CREATE EXTENSION vector;` → the pgvector package isn't installed.
   macOS: `brew install pgvector`. Ubuntu: `sudo apt install
   postgresql-16-pgvector`. Then retry the `CREATE EXTENSION` step.

2. **`psycopg.OperationalError: connection failed` / `password
   authentication failed`** → either Postgres isn't running
   (`brew services start postgresql@16` / `sudo systemctl start
   postgresql`), the password in `.env` doesn't match the OS-level
   `ALTER USER`, or `DATABASE_URL` points at the wrong host/port.
   Quick check: `pg_isready -U claim2code -h localhost`.

3. **`pydantic.ValidationError` on boot** → a `.env` variable is
   misspelled or unknown. `Settings.extra="forbid"` (`config.py:33`)
   rejects anything that doesn't match a declared field. Cross-check
   the names against the template in step 3 above.

4. **`pytest` runs zero tests** → default `pytest` is filtered by
   `addopts` to skip `integration` / `safety` / `observability`. Use
   `pytest -m <marker>` to opt into a specific marker suite, or run
   `make test-integration` / `pytest -m safety` / `pytest -m
   observability` explicitly.

5. **httpx 0.28+ "Proxy error" in tests** → the OpenAI client uses
   `httpx.Client(proxy=None, trust_env=False)` to bypass
   `HTTP_PROXY` / `HTTPS_PROXY` env-var interference. If you see this
   in your own code, mirror that pattern.

6. **OpenAI rate limits during `make eval-retrieval`** → the retrieval
   evaluator makes 1 LLM call per candidate symbol; ~30 candidates
   × 3 papers can hit the per-minute limit. Either retry with backoff
   or upgrade to a paid-tier key.

7. **`console exporter` floods stdout during dev** → set
   `OTEL_EXPORTER_CONSOLE=false` in `.env`. The exporter is wired
   from the same `Settings` field at `config.py:129`.

---

## Project layout

```
claim2code/
├── apps/
│   ├── api/                 # FastAPI backend (Python)
│   └── web/                 # React + Vite SPA
├── evaluation/              # golden datasets + baselines + LLM-judge scripts
├── docs/
│   ├── architecture/        # data-flow.md, retrieval.md, verification.md, …
│   ├── specs/                # PDF-derived spec, phase roadmap, schemas
│   └── ops/                 # runbook, observability, RAI
├── scripts/                 # dev-up.sh, dev-down.sh, helpers
├── deployments/             # docker-compose.yml, grafana/, prometheus/
├── .github/workflows/       # CI (lint + test + build)
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```

## Phase roadmap

| Phase | Spec name | Deliverable | Acceptance |
|---|---|---|---|
| 0 | — | scaffold (this commit) | monorepo + Makefile + docker-compose + docs |
| 1 | Project Foundation | FastAPI + React + PG stack running | `docker compose up` brings up `/health` (200) + `/health/ready` (200 w/ DB) + Vite on :5173 |
| 2 | Paper Intelligence | PDF upload + 5–15 structured claims per paper | Upload a paper → see claims with paper evidence |
| 3 | Repository Intelligence | GitHub clone + AST symbols + configs | Repo page shows functions/classes/configs with line numbers |
| 4 | Semantic Indexing | Embeddings + pgvector retrieval | Top-5 candidate symbols for any claim |
| 5 | Hybrid Retrieval | lexical + structural + ensemble | Recall@1/5/MRR vs embedding-only baseline (3 hand-labelled pairs in `evaluation/annotations/`) |
| 6 | Verification | deterministic checkers + LLM residual + paper+code evidence | Every verification has `paper_evidence AND code_evidence OR status==UNVERIFIED` (4 statuses: VERIFIED / PARTIALLY_VERIFIED / CONTRADICTED / UNVERIFIED). `POST /analyses/{id}/verifications` + `GET` return the list; 8 discrepancy types, 3 severities. |
| 7 | Related Work | scholarly API + 5 structured comparable papers | Selected claim → 6-field (problem/method/dataset/metric/difference/agreement) comparison per paper; Semantic Scholar primary + OpenAlex fallback. `POST /analyses/{id}/related-work` + `GET`; per-paper LLM isolation (failure → `comparison=None`, paper still ships with metadata + score). |
| 8 | Responsible AI | trust boundary + secret redaction + no-exec | All adversarial fixtures blocked. `SafetyGate` wraps every LLM call at the SDK seam (`chat.completions.create`); `SecretRedactor` replaces credential spans (OpenAI / Anthropic / AWS / GCP / GitHub / PEM / Bearer / `.env`) with `[REDACTED:<label>]`; `PromptInjectionDetector` returns hits without mutating text. Violations are appended to `AnalysisState.errors` with `{stage, kind, pattern, span_count, ts}` — no content persisted. 8 credential patterns + 5 injection patterns; 57 adversarial tests (`pytest -m safety`); new CI job `backend-safety` gates Phase 8. |
| 9 | Observability | OTel spans + per-step metrics + trace tree | One analysis → one trace tree. Each of the 3 workflows (`paper_workflow.run`, `verification_workflow.run`, `related_work_workflow.run`) opens a root span; per-claim / per-paper / per-LLM-call / per-retriever-leg open child spans (`verification.claim_verify`, `literature.paper_compare`, `llm.{extract_claims,verdict,compare}`, `retrieval.{semantic,lexical,structural}`). `safety.gate_wrap` auto-parents under each LLM span via OTel contextvars. Metrics: `llm_tokens_total` (counter, model+kind), `llm_request_duration_seconds` (histogram), `retriever_request_duration_seconds` (histogram, leg label), `safety_violations_total` (counter, pattern label). Console exporter default + OTLP-HTTP optional via `OTEL_EXPORTER_OTLP_ENDPOINT`; FastAPIInstrumentor wraps every HTTP request as a server span. 29 tests (`pytest -m observability`); new CI job `backend-observability` gates Phase 9. |
| 10 | Evaluation | 8–12 papers, 40–60 claims, all metrics | `evaluation/results.json` + Markdown report. Shipped: retrieval, verification and safety evaluators + report (5 indexed pairs / 44 claims measured, see [Measured results](#measured-results)); more pairs still to annotate. |

## License

TBD.