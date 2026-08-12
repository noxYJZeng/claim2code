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
Phase 5: ✅ retrieval evaluator — 3 hand-labelled (paper, repo) pairs
  in `annotations/` + `baselines/{embedding_only,lexical_only,
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
Phase 8+: wire safety evaluator.
Phase 10: full eval suite + measured numbers back into README.