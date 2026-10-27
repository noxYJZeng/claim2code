# Claim2Code — Evaluation

The evaluation harness lives at the repo root in `evaluation/` so it's
visible from the project layout and runnable via `make eval-*`.

## Layout (Phase 10)

```
evaluation/
├── annotations/              # one JSON per (paper, repo) pair: claims with
│                             #   relevant_symbols + expected_verification (+ notes)
├── datasets/
│   └── safety_corpus.json    # fabricated secrets, injections and clean text
├── baselines/
│   ├── embedding_only.py     # Baseline A: cosine similarity only
│   ├── lexical_only.py       # Baseline B: Postgres full-text rank only
│   └── hybrid_claim2code.py  # Our system: RRF(semantic, lexical) × structural boost
├── evaluate_retrieval.py     # Recall@1, Recall@5, MRR vs annotations
├── evaluate_verification.py  # status accuracy, discrepancy P/R, grounding, line accuracy
├── evaluate_safety.py        # secret protection, injection detection, false positives
├── report.py                 # results/*.json → results.json + report.md
├── evaluate_all.py           # runs every evaluator, then report.py
├── results/                  # per-evaluator output (retrieval/verification/safety.json)
├── results.json              # combined, with commit + model names
└── report.md                 # the rendered report
```

Annotation pairs are matched to indexed repositories by `repo.url`
(`CLAIM2CODE_EVAL_REPO_IDS='{"pair_id": "<uuid>"}'` overrides). A pair
whose repository is not indexed is skipped and listed in the report.

### What the verification metrics mean

- **Status accuracy** — exact match with `expected_verification`.
- **Support rate** — every annotated claim is implemented in its repo, so
  VERIFIED or PARTIALLY_VERIFIED counts as supported.
- **Discrepancy precision / recall** — claim level: annotated positive when
  `expected_verification` is PARTIALLY_VERIFIED (the mismatch is recorded in
  `notes`), predicted positive when the verdict is PARTIALLY_VERIFIED or
  CONTRADICTED.
- **Grounding rate** — the cited code exists in the cited file. A quote is
  split at line breaks and `...` elisions and every fragment must occur
  verbatim (whitespace and `\` line continuations ignored), so skipped or
  elided lines pass while edited code does not.
- **Line accuracy** — the cited code also sits at the cited lines (±2).
- **Evidence hit rate** — supported verdicts citing an annotated symbol.

### Safety corpus subsets

`covered` cases use the forms the Phase 8 patterns target, so they act as a
regression check. `held_out` cases are common real-world forms written
without looking at the patterns, so their rates measure coverage gaps. A
secret counts as leaked when any 12-character window of it survives the
gate, which catches partial redactions.

## Status

Phase 0: empty layout.
Phase 1+: wire annotation schema.
Phase 5: ✅ retrieval evaluator — 7 hand-labelled (paper, repo) pairs
  in `annotations/` (attention, BERT, LoRA, MAE, DDIM, Reflexion, DPO) + `baselines/{embedding_only,lexical_only,
  hybrid_claim2code}.py` + `evaluate_retrieval.py`. Run via
  `make eval-retrieval` (requires live Postgres + OpenAI).
Phase 6: ✅ verifier implemented — 4-status output
  (VERIFIED / PARTIALLY_VERIFIED / CONTRADICTED / UNVERIFIED),
  8 discrepancy types × 3 severities, deterministic
  HyperparameterChecker + 3 normalizers (models / datasets /
  metrics) + LLMClient.verdict_for_claim residual. Acceptance
  rule (`paper_evidence AND code_evidence OR UNVERIFIED`) is
  enforced as the structural guard in `compose_verification`.
  106 unit tests + 1 opt-in integration test cover the
  pipeline. Wiring `evaluate_verification.py` is Phase 10's job.
Phase 7: ✅ related-work pipeline — SemanticScholarClient primary +
  OpenAlexClient fallback, 4-leg linear-combination ranker
  (semantic / recency / citation / method_overlap),
  LLMClient.extract_comparison (one LLM call per paper, per-paper
  isolation mirrors Phase 6's per-claim pattern). 6-field
  Comparison schema (problem / method / dataset / metric /
  difference / agreement) enforced via Structured Outputs +
  `extra="forbid"`. `POST /analyses/{id}/related-work` +
  `GET`. 80 unit tests + 3 opt-in integration tests. Wiring
  `evaluate_related_work.py` (S2-vs-OpenAlex coverage + comparison
  field accuracy) is Phase 10's job.
Phase 8: ✅ responsible-AI safety gate — `SafetyGate` wraps every
  LLM call at the SDK seam (`chat.completions.create`). Composes
  `SecretRedactor` (8 credential patterns: OpenAI, Anthropic, AWS,
  GCP, GitHub, PEM, Bearer, `.env`) + `PromptInjectionDetector`
  (5 heuristic patterns). Redacts secrets in-place; detects
  injections (read-only). Violations appended to
  `AnalysisState.errors` with `{stage, kind, pattern, span_count,
  ts}` — no content persisted. 57 adversarial tests
  (`pytest -m safety`); new CI job `backend-safety` gates
  Phase 8. Wiring `evaluate_safety.py` (secret-leak + injection-
  block rate vs target 100%) is Phase 10's job.
Phase 9: ✅ OpenTelemetry spans + per-step metrics. Every workflow
  call opens a root span (`paper_workflow.run`,
  `verification_workflow.run`, `related_work_workflow.run`); child
  spans for per-claim (`verification.claim_verify`), per-paper
  (`literature.paper_compare`), per-LLM-call
  (`llm.{extract_claims,verdict,compare}`), per-retriever-leg
  (`retrieval.{semantic,lexical,structural}`); `safety.gate_wrap`
  auto-parents under each LLM span via OTel contextvars. 4 metric
  instruments — `llm_tokens_total` (counter, model+kind),
  `llm_request_duration_seconds` (histogram),
  `retriever_request_duration_seconds` (histogram, leg label),
  `safety_violations_total` (counter, pattern label). Console
  exporter default + OTLP-HTTP optional via
  `OTEL_EXPORTER_OTLP_ENDPOINT`; `FastAPIInstrumentor` wraps every
  HTTP request as a server span. 29 tests (`pytest -m
  observability`); new CI job `backend-observability` gates
  Phase 9. Wiring `evaluate_observability.py` (per-leg latency
  distribution + token cost + violation rate) is Phase 10's job.
Phase 10: ✅ evaluators shipped — `evaluate_verification.py`,
  `evaluate_safety.py`, `report.py`, `evaluate_all.py`; measured on 8
  pairs (Attention, BERT, DDIM, DDPM, DPO, LoRA, MAE, Reflexion; 79 claims),
  numbers in `report.md` and the top-level README.