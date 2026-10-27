# Claim2Code — Evaluation Report

Generated 2026-10-27T20:46:33+00:00 at commit `be1e5a5` with chat model `qwen3.8-27b-mlx` and embeddings `text-embedding-nomic-embed-text-v1.5`.

## Retrieval

### ddim

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.12     | 0.62     | 0.279   |
| lexical_only      | 0.38     | 0.38     | 0.375   |
| hybrid_claim2code | 0.38     | 0.50     | 0.417   |

### dpo

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.44     | 0.78     | 0.611   |
| lexical_only      | 0.44     | 0.67     | 0.522   |
| hybrid_claim2code | 0.33     | 0.78     | 0.500   |

### lora

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.00     | 0.50     | 0.225   |
| lexical_only      | 0.00     | 0.00     | 0.000   |
| hybrid_claim2code | 0.20     | 0.70     | 0.408   |

### mae

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.44     | 0.89     | 0.611   |
| lexical_only      | 0.22     | 0.89     | 0.396   |
| hybrid_claim2code | 0.78     | 0.89     | 0.806   |

### reflexion

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.12     | 0.25     | 0.188   |
| lexical_only      | 0.12     | 0.50     | 0.281   |
| hybrid_claim2code | 0.12     | 0.62     | 0.302   |

### Aggregate

| Baseline          | Avg R@1  | Avg R@5  | Avg MRR  |
|-------------------|----------|----------|----------|
| embedding_only    | 0.23     | 0.61     | 0.383    |
| lexical_only      | 0.23     | 0.49     | 0.315    |
| hybrid_claim2code | 0.36     | 0.70     | 0.487    |

Aggregates are macro averages over datasets. Baselines: embedding_only, lexical_only, hybrid_claim2code.

## Verification

44 annotated claims, production pipeline (hybrid retrieval top_k=8 → deterministic checkers → LLM verdict).

| Metric | Value |
|---|---|
| Status accuracy (exact, 4-way) | 77.3% |
| Support rate (VERIFIED or PARTIALLY_VERIFIED) | 84.1% |
| Discrepancy precision | 57.1% (7 flagged) |
| Discrepancy recall | 66.7% (6 annotated) |
| Grounding rate (cited code exists in the cited file) | 100.0% (111 citations) |
| Line accuracy (cited code at the cited lines) | 91.0% |
| Evidence hit rate (supported verdicts citing an annotated symbol) | 94.6% |
| LLM errors | 0 |

### Per pair

| Pair | Claims | Accuracy | Support | Disc. P | Disc. R | Grounding |
|---|---|---|---|---|---|---|
| ddim | 8 | 75.0% | 87.5% | 0.0% | n/a | 100.0% |
| dpo | 9 | 77.8% | 77.8% | 100.0% | 100.0% | 100.0% |
| lora | 10 | 70.0% | 70.0% | n/a | 0.0% | 100.0% |
| mae | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| reflexion | 8 | 62.5% | 87.5% | 50.0% | 66.7% | 100.0% |

### Confusion (rows = annotated, columns = predicted)

| | VERIFIED | PARTIALLY_VERIFIED | CONTRADICTED | UNVERIFIED |
|---|---|---|---|---|
| VERIFIED | 31 | 3 | 0 | 4 |
| PARTIALLY_VERIFIED | 0 | 3 | 1 | 2 |

Skipped (repository not indexed): attention_is_all_you_need, bert.

## Safety

`covered` cases use the forms the Phase 8 patterns target; `held_out` cases are common real-world forms written without reference to them.

| Check | Covered | Held-out | All |
|---|---|---|---|
| Secrets protected | 10/10 (100.0%) | 2/8 (25.0%) | 12/18 (66.7%) |
| Injections detected | 6/6 (100.0%) | 0/6 (0.0%) | 6/12 (50.0%) |
| Clean text flagged (false positive) | | | 3/12 (25.0%) |

| Failed case | Category | Subset | Patterns hit |
|---|---|---|---|
| s11 | secret | held_out | — |
| s13 | secret | held_out | — |
| s14 | secret | held_out | — |
| s15 | secret | held_out | anthropic_key |
| s17 | secret | held_out | — |
| s18 | secret | held_out | — |
| i07 | injection | held_out | — |
| i08 | injection | held_out | — |
| i09 | injection | held_out | — |
| i10 | injection | held_out | — |
| i11 | injection | held_out | — |
| i12 | injection | held_out | — |
| c04 | clean | clean | role_injection |
| c06 | clean | clean | disregard_prior |
| c10 | clean | clean | you_are_now |
