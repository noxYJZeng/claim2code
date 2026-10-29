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
├── preflight.py              # checks a run can start (used by scripts/eval-local.sh)
├── results/                  # per-evaluator output (retrieval/verification/safety.json)
├── results.json              # combined, with commit + model names
└── report.md                 # the rendered report
```

Annotation pairs are matched to indexed repositories by `repo.url`
(`CLAIM2CODE_EVAL_REPO_IDS='{"pair_id": "<uuid>"}'` overrides). A pair
whose repository is not indexed is skipped and listed in the report.

### One-command local run

```bash
scripts/eval-local.sh --check    # preflight only, about a minute
scripts/eval-local.sh            # preflight, every evaluator, report (≈2.5 h with a local 27B model)
scripts/eval-local.sh --resume   # continue an interrupted verification run
```

The script uses the database and models in `.env` (for example LM Studio
on `localhost:1234`). Before spending hours it checks that the database
answers, every annotated pair is indexed, the chat and embedding models
are actually served (LM Studio answers an unknown model id with whatever
is loaded), and the embedding dimension matches `OPENAI_EMBED_DIM` and the
stored embeddings. On macOS it keeps the machine awake with `caffeinate`.
The full log goes to `evaluation/logs/` (gitignored); results and
`report.md` are written as usual. Commit first: a run on uncommitted code
is recorded as dirty.

## Measurement scope and reproducibility

The checked-in numbers are **development-set measurements** of commit
`93cd752` (2026-10-28), with provenance in each results file. The saved
verification predictions have 70 VERIFIED and 9 PARTIALLY_VERIFIED gold
labels, with no CONTRADICTED or UNVERIFIED gold examples. The 79.7% status
accuracy should be compared with the 88.6%
always-VERIFIED majority baseline. Reports now include that baseline,
label counts, per-class metrics and macro F1 over gold-present classes;
metrics for absent gold classes remain undefined.

Verification starts from curated claims, not PDF upload. The annotation
text is also used as its paper quote, so these numbers do **not** measure
claim extraction accuracy or the correctness of paper evidence against
the original PDF. Code grounding checks textual existence; it does not
measure whether a conclusion follows from that text.

New evaluator runs record their own `provenance` when executing: run ID,
start/end timestamps, full code SHA and source digest, non-secret model
IDs/settings, annotation hashes, and the actual indexed clone revisions.
Model revision and quantization are explicitly unknown when the runtime
does not expose them. API keys and full environment dumps are never
captured. The report propagates the saved metadata; it does not read the
current model configuration to relabel old results. Legacy files remain
marked as provenance-unknown. Their original combined-report header is
preserved only as an unverified historical declaration.

Several legacy annotations still contain `commit: "main"` and placeholder
PDF hashes. New development runs warn about these inputs. For a frozen
benchmark, supply real full repository SHAs and PDF hashes, then run:

```bash
python evaluation/evaluate_retrieval.py --require-reproducible
python evaluation/evaluate_verification.py --require-reproducible
python evaluation/report.py --require-provenance
```

A pinned annotation whose SHA differs from the indexed clone is rejected
even in development mode; repository-ID overrides cannot silently point
at a different project. `--require-reproducible` also refuses unknown or
dirty clone states. This input gate does not promise deterministic model
outputs or validate PDF bytes that the curated-claim evaluator never reads.
`--require-provenance` refuses missing, mixed or unfrozen result provenance
instead of producing a falsely unified comparison.

`--resume` requires the same code, model, settings, annotations and clone
revisions as the run's `verification_run.json` sidecar; legacy JSONL files
without that identity cannot be resumed. `--regrade` retains the original
inference identity and label, records a separate grading identity, and
leaves unknown historical inference metadata unknown. Regrading old
predictions never means that the current model or verifier has been rerun.

### What the verification metrics mean

- **Status accuracy** — exact match with `expected_verification` on the
  labels present in the selected data, accompanied by majority baseline.
- **Support rate** — the proportion predicted VERIFIED or
  PARTIALLY_VERIFIED; a descriptive rate, not accuracy.
- **Discrepancy precision / recall** — claim level: annotated positive when
  `expected_verification` is PARTIALLY_VERIFIED or CONTRADICTED (the
  mismatch is recorded in `notes`), predicted positive when the verdict is
  PARTIALLY_VERIFIED or CONTRADICTED.
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
Phase 5: ✅ retrieval evaluator — 8 hand-labelled (paper, repo) pairs
  in `annotations/` (attention, BERT, LoRA, MAE, DDIM, DDPM, Reflexion, DPO) + `baselines/{embedding_only,lexical_only,
  hybrid_claim2code}.py` + `evaluate_retrieval.py`. Run via
  `make eval-retrieval` (requires live Postgres + OpenAI).
Phase 6: ✅ verifier implemented — 4-status output
  (VERIFIED / PARTIALLY_VERIFIED / CONTRADICTED / UNVERIFIED),
  8 discrepancy types × 3 severities, deterministic
  HyperparameterChecker + 3 normalizers (models / datasets /
  metrics) + LLMClient.verdict_for_claim residual. Acceptance
  rule (`paper_evidence AND code_evidence OR UNVERIFIED`) is
  enforced as the structural guard in `compose_verification`.
  Unit and opt-in integration tests cover the pipeline;
  `evaluate_verification.py` supplies the curated-claim evaluation.
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
  Phase 8. `evaluate_safety.py` measures secret protection and
  injection detection, including held-out failures; detection does
  not establish that a malicious instruction was blocked.
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
