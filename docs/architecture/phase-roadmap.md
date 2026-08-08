# Claim2Code — Phase Roadmap

> The PDF defines 10 phases. We add an internal **Phase 0** (scaffold) to
> align with the artifact-atlas workflow pattern.

Each phase = 1 commit (or a small logical group of commits). The commit
message starts with the phase tag so `git log --oneline` reads as a roadmap.

---

## Phase 0 — Scaffold (this commit)

**Tag in commit:** `chore(scaffold):`
**Why separate:** gives us a green CI baseline before any code. Matches the
artifact-atlas Phase 0 pattern.

**What ships:**
- Monorepo skeleton (`apps/api/`, `apps/web/`, `evaluation/`, `docs/`,
  `scripts/`, `deployments/`, `.github/workflows/`)
- `docker-compose.yml` (postgres+pgvector)
- `Makefile` (`make dev`, `make test`, `make lint`, `make help`)
- `.env.example`
- `README.md` (project landing page)
- `docs/specs/spec-summary.md` (PDF condensation)
- `docs/architecture/data-flow.md` (end-to-end picture)
- `docs/architecture/phase-roadmap.md` (this file)
- `.gitignore` (excludes `interview-prep/`, secrets, builds, runtime state)

**Acceptance:** `git log` shows one commit. `make help` works. `docker compose config` validates.

---

## Phase 1 — Project Foundation (PDF Phase 1)

**Tag:** `feat(foundation):`
**Goal:** bring up the full local stack with a working health endpoint.

**What ships:**
- FastAPI app with `/health` and `/health/ready` probes
- React + Vite + Tailwind skeleton with a single "welcome" page
- PostgreSQL + pgvector schema migration 0001 (empty migrations table + analysis_states)
- `pytest` backend test (one trivial health test passes)
- `npm run build` succeeds for the web
- GitHub Actions: backend lint + backend test + frontend build
- `scripts/dev-up.sh` brings up everything end-to-end

**Acceptance:** `docker compose up` starts the stack. `curl localhost:8000/health` returns 200. Web loads at `localhost:5173`.

---

## Phase 2 — Paper Intelligence (PDF Phase 2)

**Tag:** `feat(paper):`
**Goal:** upload a PDF → see 5–15 structured technical claims with paper-evidence spans.

**What ships:**
- `POST /papers` accepts PDF upload, hashes it, persists
- PyMuPDF-based parser → `Paper`, `PaperSection`, `Equation`, `Table`
- LLM-driven `ClaimExtractor` with Pydantic-validated structured output
- `GET /papers/{id}/claims` returns the claim list
- Frontend: drag-drop upload + claims list view

**Acceptance:** Upload a small paper (e.g. SDAR or any short arXiv paper) → see 5–15 claims, each with paper section + page + character offset.

---

## Phase 3 — Repository Intelligence (PDF Phase 3)

**Tag:** `feat(repo):`
**Goal:** paste a GitHub URL → see functions/classes/configs with exact file:line locations.

**What ships:**
- `POST /repositories` accepts URL, clones into sandbox, filters tree
- tree-sitter AST walker → `CodeSymbol`s (class, method, function, signature, docstring, imports, calls)
- Symbol graph (who calls whom) persisted
- Config parser for argparse / dataclasses / YAML / JSON / TOML → `ConfigEntry`
- `GET /repositories/{id}/symbols` returns the symbol list
- Frontend: repo page with file-tree + symbol detail

**Acceptance:** Paste a public PyTorch repo → see all functions/classes with line numbers and config values.

---

## Phase 4 — Semantic Indexing (PDF Phase 4)

**Tag:** `feat(index):`
**Goal:** embed each symbol's structured representation → retrieve Top-5 candidate symbols for any claim.

**What ships:**
- `EmbeddingClient` interface (provider-agnostic)
- Structured symbol representation string (`File: …, Symbol: …, Signature: …, Docstring: …, Source: …, Dependencies: …`)
- pgvector table `code_embeddings` with HNSW index
- `retrieve_semantic(claim, top_k=5)` end-to-end function

**Acceptance:** Given a claim string, returns 5 candidate symbols ranked by cosine similarity.

---

## Phase 5 — Hybrid Retrieval (PDF Phase 5)

**Tag:** `feat(retrieval):`
**Goal:** semantic + lexical + structural ensemble. Run the first retrieval eval.

**What ships:**
- Lexical scorer (BM25 or token overlap on algorithm names + variable names + function names + config keys + module names)
- Structural scorer (called-from-training-loop boost, imported-by-relevant-file boost, loss-metric boost, dependency-graph proximity)
- `hybrid_ranker(claim, top_k)` returns Top-K symbols with score breakdown
- `evaluation/baselines/embedding_only.py` and `evaluation/baselines/lexical_only.py`
- `evaluation/evaluate_retrieval.py` produces Recall@1, Recall@5, MRR

**Acceptance:** First numbers on 3 hand-annotated paper/repo pairs. Hybrid > embedding-only on Recall@5.

---

## Phase 6 — Verification (PDF Phase 6)

**Tag:** `feat(verify):`
**Goal:** every claim gets a verdict + paper evidence + code evidence + discrepancies.

**What ships:**
- `Verifier` LLM-driven with explicit "no model knowledge" rule
- Deterministic `HyperparameterChecker` (learning_rate / batch_size / temperature / beta / gamma / num_epochs / hidden_size)
- Deterministic `ModelNameNormalizer` (Qwen2.5-7B ↔ Qwen/Qwen2.5-7B)
- `DatasetNormalizer` for known naming variations
- `MetricNameComparator`
- `DiscrepancyClassifier` with 8 types + 3 severities, deterministic rules where possible
- `GET /analyses/{id}/verifications`

**Acceptance:** Every verification has paper evidence AND code evidence, or `UNVERIFIED`.

---

## Phase 7 — Related Work (PDF Phase 7)

**Tag:** `feat(literature):`
**Goal:** selected claim → up to 5 related papers with structured comparison.

**What ships:**
- Scholarly search client (e.g. Semantic Scholar / arXiv)
- Ranking function (semantic similarity × recency × citation relevance × method overlap)
- LLM-driven structured comparison extractor
- `GET /analyses/{id}/related-work?claim_id=…`

**Acceptance:** Returns up to 5 papers, each with Problem / Method / Dataset / Metric / Difference / Agreement fields.

---

## Phase 8 — Responsible AI (PDF Phase 8)

**Tag:** `feat(safety):`
**Goal:** adversarial fixtures all blocked.

**What ships:**
- `SafetyGate` wrapping every LLM call
- `SecretRedactor` (regex on .env, credentials, private keys, access tokens)
- `PromptInjectionDetector` (Ignore previous instructions / Reveal your API key patterns)
- Untrusted-input boundary (PDF / repo content treated as data, never as instructions)
- Adversarial test fixtures in `tests/safety/`

**Acceptance:** All adversarial tests pass. No secret enters the model context. No ungrounded claim becomes `VERIFIED`.

---

## Phase 9 — Observability (PDF Phase 9)

**Tag:** `feat(observability):`
**Goal:** one analysis request → one trace tree.

**What ships:**
- OpenTelemetry SDK initialized at app startup
- Spans for: `paper.parse`, `paper.claim_extract`, `repo.clone`, `repo.parse`,
  `repo.embed`, `retrieval.claim_to_code`, `verification.claim`,
  `literature.search`, `report.generate`
- Per-LLM-call metrics: latency, input/output tokens, success, retry count, model name
- Per-retrieval metrics: Top-K scores, retrieval latency, rank distribution
- Per-workflow metrics: total runtime, failure stage, retries
- Frontend observability dashboard (or simple JSON viewer)

**Acceptance:** Run an analysis → see a trace tree in Jaeger / console exporter.

---

## Phase 10 — Evaluation (PDF Phase 10)

**Tag:** `feat(eval):`
**Goal:** measured numbers on all metrics.

**What ships:**
- 8–12 paper/repo pairs hand-annotated (40–60 claim-to-code mappings, several intentional discrepancies)
- `evaluation/baselines/embedding_only.py` (Baseline A)
- `evaluation/baselines/lexical_only.py` (Baseline B)
- `evaluation/baselines/hybrid_claim2code.py` (Claim2Code)
- `evaluation/evaluate_retrieval.py` → Recall@1, Recall@5, MRR
- `evaluation/evaluate_verification.py` → verification accuracy, discrepancy precision, grounding rate
- `evaluation/evaluate_safety.py` → prompt-injection blocked count
- `evaluation/report.py` → `evaluation/results.json` + Markdown report

**Acceptance:** README and resume use measured values, not target ranges.

---

## Why this order?

- Each phase has an **observable acceptance criterion** — you can run the
  command and see whether it passed.
- Earlier phases have **no LLM dependency** (parsing, repo cloning, AST,
  indexing). They build confidence in the deterministic substrate before
  we trust the LLM-driven components.
- Verification (Phase 6) comes **after** retrieval (Phase 5) because the
  verifier needs candidate symbols to reason about.
- Related Work (Phase 7) is **separate** from verification because it's a
  secondary feature and shouldn't block the main verification path.
- Evaluation (Phase 10) is **last** because we need all the components to
  evaluate them.

## Branching / git convention

- `main` is always shippable.
- One branch per phase: `phase/1-foundation`, `phase/2-paper`, …
- Merge back to `main` with `--no-ff` to keep phase boundaries visible in history.
- Commit message prefix mirrors artifact-atlas:
  `feat(api):`, `feat(repo):`, `feat(retrieval):`, `feat(safety):`,
  `feat(observability):`, `feat(eval):`, `chore(scaffold):`,
  `fix(...)`, `docs(...)`, `polish(...)`, `refactor(...)`.