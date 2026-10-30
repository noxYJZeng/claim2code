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
- 🔎 **Hybrid Claim-to-Code Retrieval** — semantic + lexical + structural, weighted ensemble, then an LLM rerank of the top 45 candidates
- ✅ **Verification** — every claim gets a verdict + paper evidence + code evidence
- ⚠️ **Discrepancy Detection** — hyperparameter / objective / dataset / metric mismatches
- 📚 **Related Work** — selected claim → 5 structured comparable papers
- 🛡️ **Input checks** — pattern-based secret redaction and injection detection, with measured coverage gaps; ingested code is never executed
- 📊 **Observability** — OpenTelemetry traces + per-step latency/token/retrieval metrics
- 🧪 **Evaluation Harness** — hand-annotated paper/repo pairs; retrieval Recall@1/5 + MRR, verification accuracy + discrepancy precision/recall + grounding rate, safety redaction/detection rates ([measured results](#measured-results))

## Try the portfolio demo

The **View sample analysis** entry opens a self-contained, read-only example.
It needs no API server, database, model, or API key. Start the frontend and open
it with `?demo=1`, or use the sample button on the landing page:

```bash
cd apps/web
npm ci --registry=https://registry.npmjs.org --replace-registry-host=always
npm run dev
# npm run build produces dist/ for a static website deployment.
```

The sample uses recorded development results. Its curated claim text is not
presented as an independently verified quotation from a PDF. For a personal
website, publish the static demo with a short explanation of the architecture,
retrieval comparison, and known failure cases. It does not expose model
configuration or accept uploads.

## Measured results

Benchmark measured locally on 2026-10-30 at commit `d38c62d` with
`qwen3.8-27b-mlx` (LM Studio) and `nomic-embed-text-v1.5` embeddings, by
[`evaluation/run_local.py --prepare`](evaluation/README.md#run-the-resume-target-benchmark-on-a-local-model).
The suite has 15 paper/repository pairs, 12 of them PyTorch. Headline numbers
use the original **79-claim regression set** (8 pairs: Attention Is All You
Need, BERT, DDIM, DDPM, DPO, LoRA, MAE, Reflexion), frozen at `77e2e97`; the
35 claims on the 7 newer pairs (CLIP, DeiT, DETR, DINO, MoCo, SAM, Whisper)
are reported separately and cannot change it. The run's target check,
report and per-claim results are checked in under
[`evaluation/benchmarks/resume-targets-d38c62d/`](evaluation/benchmarks/resume-targets-d38c62d/targets.md),
with provenance in each results file.

| Area | Metric | Claim2Code | Baseline |
|---|---|---|---|
| Retrieval (top 5) | Hit@5, macro avg (annotated symbol in the top 5) | **97.2%** | 72.7% hybrid without rerank · 64.5% embedding-only |
| | Recall@5, per symbol, macro avg | **85.8%** | 60.6% hybrid without rerank · 54.8% embedding-only |
| | MRR, macro avg | **0.837** | 0.456 hybrid without rerank · 0.462 embedding-only |
| Verification | Status accuracy | **87.3%** (69/79) | 88.6% always-VERIFIED baseline |
| | Macro F1 over gold labels (VERIFIED, PARTIALLY_VERIFIED) | 0.904 | 0.470 always-VERIFIED |
| | Discrepancy precision / recall (claim level) | 100% / 77.8% | — |
| | Grounding rate (cited code exists in the cited file) | **100%** of 237 citations | — |
| | Line accuracy (cited code at the cited lines) | 100% | — |
| Newer pairs (35 claims) | Hit@5 · status accuracy | 97.1% · 100% | — |
| Safety | Secrets protected, covered / held-out forms | 10/10 · 2/8 | — |
| | Injections detected, covered / held-out forms | 6/6 · 0/6 | — |
| | Clean text flagged | 3/12 | — |

Hit@5 is the definition behind the earlier reported 74% (hybrid retrieval at
`93cd752`); re-measured on the same claims it is 72.7%. The gain comes from the
LLM rerank stage: hybrid retrieval proposes 45 candidates and the chat model
picks the most direct implementations from their source heads
(`retrieval/reranker.py`). Rerank reaches the pool's ceiling: 97.2% of claims
have an annotated symbol in hybrid's top 45.

The gold set contains 70 `VERIFIED` and 9 `PARTIALLY_VERIFIED` claims, with no
`CONTRADICTED` or `UNVERIFIED` examples, so status accuracy sits just below the
always-VERIFIED baseline; macro F1 shows that the partial cases are found. It
does not establish four-way classification quality. Evaluation supplies
curated claims, using those texts as paper quotes; it does not measure
PDF-to-claim extraction or paper-citation correctness. Code grounding measures
quote existence, not verdict correctness. Grounding and line accuracy are 100%
by construction: the runtime check keeps only quotes found in source shown to
the model and moves each to its actual lines.

History on the same 79 claims: 81.0% (99.5% grounding, 92.1% line accuracy)
before the citation check; 70.9% with a first, over-strict version of it;
79.7% at `93cd752`; 82.3% with rerank; 87.3% after treating differing
defaults and initialisations as partial support rather than contradiction
(`verification/discrepancies.py`). That rule was chosen on 27 development
claims from the newer pairs (`evaluation/dev_annotations/`, 22/27 → 24/27),
then measured once on the regression set. Accuracy is close to the 87% line:
one claim is 1.3 points, and the local model's verdicts vary between runs
(ddim_c4 and ddpm_c4 changed between the last two runs).

Caveats: only 9 of the 79 claims carry an annotated discrepancy, so the
discrepancy rates rest on few cases. The annotations were written during
development, not by an independent annotator. Five were corrected after a first
measured run (2026-10-27, each with a note in its annotation): lora_c3/c5/c6
named the wrong symbol or misstated the code, and ddim_c1/mae_c5 were labelled
PARTIALLY_VERIFIED with no paper/code mismatch. Scored against the corrected
labels, the earlier pipeline reached 63.6% accuracy on the first 5 pairs. The
Attention, BERT and DDPM pairs were annotated afterwards against
harvardnlp/annotated-transformer, google-research/bert and
hojonathanho/diffusion; BERT's pre-training hyperparameters are absl flags,
which the parser now indexes as config symbols. Of the 10 remaining misses, 8 are UNVERIFIED verdicts on claims annotated VERIFIED (2 with no code evidence; LoRA's `examples/NLU` is a vendored transformers checkout), one partial claim is UNVERIFIED, and one is VERIFIED where a mismatch was expected. The safety `held_out` cases (GitHub
fine-grained / Slack / Hugging Face tokens, paraphrased injections) are coverage
gaps of the regex gate, which matches only the forms listed in Phase 8.
Injection detection records a match; it is not a guarantee that the input was
blocked. Use the static demo for public portfolio access.

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
# Edit .env for the database and a model endpoint. Hosted providers use
# their own API keys; a local OpenAI-compatible server needs no cloud key.

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
| **Model endpoint** | YES for live analysis; local LM Studio/Ollama or a hosted provider | Local endpoint, or provider credentials | same |
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

# ---- Other chat providers (optional; pick one in the web UI) ----
ANTHROPIC_API_KEY=
OPENAI_PLATFORM_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
OPENROUTER_API_KEY=

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

**Choosing the chat model.** The `OPENAI_*` block is the default chat
model and the embedding model. The **Chat model** panel at the top of the
web UI (or `GET/POST /llm/providers`, `/llm/provider`, `/llm/provider/test`)
switches the chat model used for claim extraction, verification and
related work between LM Studio, Ollama, Anthropic (Claude, through the
Anthropic SDK with structured outputs), OpenAI, Google Gemini, DeepSeek,
OpenRouter, or any OpenAI-compatible server. "Test connection" checks a
choice without switching to it. The choice is saved to
`.claim2code/llm_provider.json` (git-ignored); an API key typed in the UI
stays in the API's memory only, so put keys you want kept in `.env`.
Embeddings are not switchable there: stored vectors come from one model,
so changing `OPENAI_MODEL_EMBED` means re-indexing. DeepSeek only offers
JSON mode, so its answers are checked against the schema afterwards
rather than constrained by it.

Model-management writes are disabled by default. For a private workstation,
set `LLM_MANAGEMENT_ALLOW_LOCAL=true` and connect from the same machine. For
administrative access, set `LLM_MANAGEMENT_TOKEN` securely and enter it in the
model panel; the token stays in browser memory, not browser storage. Hosted
provider addresses cannot be overridden to carry saved keys elsewhere. A custom
endpoint uses only its explicitly supplied key.
Behind a reverse proxy, use the administrator token and leave the loopback
opt-in disabled; the proxy itself may appear as a loopback peer.

For a read-only API deployment, set `API_READ_ONLY=true`; mutating requests
are rejected before model or database work. The static portfolio demo needs no
API at all. A public service accepting uploads additionally needs deployment
authentication, request quotas and resource budgets; it is not provided by the
static demo configuration.

If your Docker builder uses an HTTPS-inspecting proxy, supply its trusted CA
bundle as a build-only secret while retaining certificate verification:

```bash
docker build --secret id=pip_ca,src=/path/to/trusted-ca-bundle.crt \
  -f deployments/Dockerfile.api -t claim2code-api .
```

The CA bundle is optional and is not retained in the image. Standard builds
need no extra trust configuration.

**Local LM Studio.** Load a chat model and an embedding model, enable the
OpenAI-compatible server, and set `OPENAI_BASE_URL`, `OPENAI_MODEL_CHAT`,
`OPENAI_MODEL_EMBED`, and `OPENAI_EMBED_DIM` to the actual model identifiers and
embedding dimensions. Use the local server's supported authentication setting.
For an unauthenticated local endpoint, its SDK configuration value is not a
cloud credential. Changing embedding model or dimensions requires a fresh
database/index, not mixing old vectors with new ones. If the API runs in a
container, its `localhost` is the container; configure access to the host model
server explicitly.

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

In the web UI: upload a PDF (or paste the `analysis_id` of an earlier
upload under "Open an earlier analysis"), clone the paper's repository,
then use the **Verification** panel to verify one claim or all of them.
Verdicts render as each claim finishes (a few minutes per claim on a local
model) and are saved, so reopening the analysis shows them again. The open
analysis is kept in the URL (`http://localhost:5173/?analysis=<id>`), so a
refresh or a bookmarked link returns to it. "Show details" on a verdict
lists the explanation and discrepancies, the cited paper passage on the
rendered PDF page with the quote highlighted (plus a link that opens the
PDF at that page), and the cited code as read from the cloned repository,
with line numbers and a GitHub link pinned to the cloned commit.

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
| `evaluation/evaluate_retrieval.py` | `make eval-retrieval` | DB + embeddings | Recall@1/5 + MRR for embedding-only, lexical-only, hybrid and hybrid + LLM rerank |
| `evaluation/evaluate_verification.py` | `make eval-verification` | DB + embeddings + chat model | status accuracy, support rate, discrepancy P/R, grounding, line accuracy |
| `evaluation/evaluate_safety.py` | `make eval-safety` | nothing | secret protection, injection detection, false positives on `datasets/safety_corpus.json` |
| `evaluation/report.py` | `make eval-report` | the results above | `results.json` + `report.md` |
| `evaluation/evaluate_all.py` | `make eval-all` | all of the above | runs every evaluator, then the report |

```bash
make eval-safety                       # seconds, no services needed
make eval-retrieval                    # ~1 min
make eval-verification                 # one LLM call per claim (~2-3 h for 79 claims locally)
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
| 10 | Evaluation | 8–12 papers, 40–60 claims, all metrics | `evaluation/results.json` + Markdown report. Shipped: retrieval, verification and safety evaluators + report (8 pairs / 79 claims measured, see [Measured results](#measured-results)). |

## License

MIT — see [LICENSE](LICENSE).
