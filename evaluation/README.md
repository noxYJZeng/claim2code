# Claim2Code — Evaluation

The evaluation harness lives at the repo root in `evaluation/` so it's
visible from the project layout and runnable via `make eval-*`.

## Layout (Phase 10)

```
evaluation/
├── datasets/             # paper + repo pairs, raw + curated
├── annotations/          # hand-annotated claim-to-code mappings
│                         #   paper.json
│                         #   repo.json
│                         #   claims.json         ← list of Claim objects
│                         #   mappings.json       ← relevant_symbols + expected_verification
├── baselines/
│   ├── embedding_only.py     # Baseline A: cosine similarity only
│   ├── lexical_only.py       # Baseline B: BM25 / token overlap only
│   └── hybrid_claim2code.py  # Our system: S = 0.55·sem + 0.25·lex + 0.20·struct
├── evaluate_retrieval.py     # Recall@1, Recall@5, MRR vs annotations
├── evaluate_verification.py  # Verification accuracy, discrepancy precision, grounding rate
├── evaluate_safety.py        # Prompt-injection / secret-leak tests
├── evaluate_all.py           # Runs all evaluators + renders results.json + report.md
└── README.md
```

## Status

Phase 0: empty layout.
Phase 1+: wire annotation schema.
Phase 5: ✅ retrieval evaluator — 6 hand-labelled (paper, repo) pairs
  in `annotations/` (attention, BERT, LoRA, MAE, DDIM, Reflexion) + `baselines/{embedding_only,lexical_only,
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
Phase 10: full eval suite + measured numbers back into README.