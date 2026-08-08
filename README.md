# Claim2Code

> AI Research Agent for Paper–Code Verification and Reproducibility Analysis.
>
> Upload a paper PDF + a GitHub URL, get a grounded verification report:
> every claim traced to a paper section and a source-code location, with
> an explicit `VERIFIED / PARTIALLY_VERIFIED / CONTRADICTED / UNVERIFIED`
> verdict and detected discrepancies.

[![status](https://img.shields.io/badge/status-Phase%201%20(scaffold)-blue)]()
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
- 🧪 **Evaluation Harness** — 8–12 paper/repo pairs, Recall@5 + verification accuracy + grounding rate

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
git clone git@github.com:YOUR_ORG/claim2code.git
cd claim2code

# 2. configure
cp .env.example .env
# Edit .env — at minimum set POSTGRES_PASSWORD and OPENAI_API_KEY.

# 3. start stack
make dev          # brings up postgres+pgvector, api, web

# 4. open
#   Web:           http://localhost:5173
#   API health:    http://localhost:8000/health
#   API docs:      http://localhost:8000/docs
```

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
| 1 | Project Foundation | FastAPI + React + PG stack running | `docker compose up` brings up health endpoint |
| 2 | Paper Intelligence | PDF upload + 5–15 structured claims per paper | Upload a paper → see claims with paper evidence |
| 3 | Repository Intelligence | GitHub clone + AST symbols + configs | Repo page shows functions/classes/configs with line numbers |
| 4 | Semantic Indexing | Embeddings + pgvector retrieval | Top-5 candidate symbols for any claim |
| 5 | Hybrid Retrieval | lexical + structural + ensemble | Recall@1/5/MRR vs embedding-only baseline |
| 6 | Verification | verdicts + deterministic hyperparam checks | Every result has evidence or UNVERIFIED |
| 7 | Related Work | scholarly API + 5 structured comparable papers | Selected claim → related-work table |
| 8 | Responsible AI | trust boundary + secret redaction + no-exec | All adversarial fixtures blocked |
| 9 | Observability | OTel spans + per-step metrics + trace tree | One analysis → one trace tree |
| 10 | Evaluation | 8–12 papers, 40–60 claims, all metrics | `evaluation/results.json` + Markdown report |

## License

TBD.