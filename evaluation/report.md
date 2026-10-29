# Claim2Code — Evaluation Report

Report rendered 2026-10-29T01:40:53+00:00. Rendering does not run the models.
Measurement code and model identities below come only from stored evaluator provenance.

Historical combined-report header (not verified per evaluator): commit `0aa09dc`, chat `qwen3.8-27b-mlx`, embeddings `text-embedding-nomic-embed-text-v1.5`.
These archived numbers predate subsequent fixes and are not a new measurement of this checkout.

Measurement `retrieval`: run `81f0d74c-cd23-4564-8e37-a46878e71893`, code `93cd752e3a530aa86fc4cfb3524f362dad464b49`, started `2026-10-29T01:40:38+00:00`.

Recorded chat `qwen3.8-27b-mlx`; embeddings `text-embedding-nomic-embed-text-v1.5`.

## Retrieval

### attention_is_all_you_need

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.58     | 0.92     | 0.704   |
| lexical_only      | 0.33     | 0.67     | 0.426   |
| hybrid_claim2code | 0.42     | 0.83     | 0.583   |

### bert

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.45     | 0.64     | 0.518   |
| lexical_only      | 0.18     | 0.55     | 0.318   |
| hybrid_claim2code | 0.45     | 0.82     | 0.579   |

### ddim

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.12     | 0.62     | 0.279   |
| lexical_only      | 0.38     | 0.38     | 0.375   |
| hybrid_claim2code | 0.38     | 0.50     | 0.417   |

### ddpm

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.50     | 0.67     | 0.583   |
| lexical_only      | 0.08     | 0.50     | 0.242   |
| hybrid_claim2code | 0.50     | 0.75     | 0.611   |

### dpo

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.44     | 0.78     | 0.611   |
| lexical_only      | 0.44     | 0.67     | 0.522   |
| hybrid_claim2code | 0.33     | 0.78     | 0.500   |

### lora

| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.00     | 0.40     | 0.200   |
| lexical_only      | 0.00     | 0.00     | 0.000   |
| hybrid_claim2code | 0.20     | 0.70     | 0.433   |

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
| embedding_only    | 0.33     | 0.65     | 0.462    |
| lexical_only      | 0.22     | 0.52     | 0.320    |
| hybrid_claim2code | 0.40     | 0.74     | 0.529    |

Aggregates are macro averages over datasets. Baselines: embedding_only, lexical_only, hybrid_claim2code.

Measurement `verification`: run `981af1fe-6a81-4090-8af4-c613554a4441`, code `93cd752e3a530aa86fc4cfb3524f362dad464b49`, started `2026-10-28T19:37:45+00:00`.

Recorded chat `qwen3.8-27b-mlx`; embeddings `text-embedding-nomic-embed-text-v1.5`.

## Verification

79 curated annotated claims, claim-to-code pipeline (hybrid retrieval top_k=8 → deterministic checkers → LLM verdict).

The input is an annotation's claim text, also used as its paper quote. This does not measure PDF extraction accuracy or verify the paper citation against a PDF.

| Metric | Value |
|---|---|
| Status accuracy (exact, on observed labels) | 79.7% |
| Majority-class baseline (always VERIFIED) | 88.6% |
| Macro F1 (gold-present labels only) | 69.1% |
| Balanced accuracy (gold-present labels only) | 69.2% |
| Support rate (VERIFIED or PARTIALLY_VERIFIED) | 88.6% |
| Discrepancy precision | 50.0% (14 flagged) |
| Discrepancy recall | 77.8% (9 annotated) |
| Grounding rate (cited code exists in the cited file) | 100.0% (200 citations) |
| Line accuracy (cited code at the cited lines) | 100.0% |
| Evidence hit rate (supported verdicts citing an annotated symbol) | 92.9% |
| LLM errors | 0 |

### Per pair

| Pair | Claims | Accuracy | Support | Disc. P | Disc. R | Grounding |
|---|---|---|---|---|---|---|
| attention_is_all_you_need | 12 | 91.7% | 100.0% | 50.0% | 100.0% | 100.0% |
| bert | 11 | 72.7% | 81.8% | 50.0% | 100.0% | 100.0% |
| ddim | 8 | 75.0% | 100.0% | 0.0% | n/a | 100.0% |
| ddpm | 12 | 91.7% | 91.7% | 100.0% | 100.0% | 100.0% |
| dpo | 9 | 77.8% | 77.8% | 100.0% | 100.0% | 100.0% |
| lora | 10 | 60.0% | 70.0% | 0.0% | 0.0% | 100.0% |
| mae | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| reflexion | 8 | 62.5% | 87.5% | 50.0% | 66.7% | 100.0% |

### Label coverage

| Gold label | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| VERIFIED | 70 | 100.0% | 82.9% | 90.6% |
| PARTIALLY_VERIFIED | 9 | 41.7% | 55.6% | 47.6% |
| CONTRADICTED | 0 | 0.0% | n/a | n/a |
| UNVERIFIED | 0 | 0.0% | n/a | n/a |

No gold examples for: CONTRADICTED, UNVERIFIED. These results do not establish four-way classification performance.

Code grounding measures whether a quote exists, not whether the verdict is correct.

### Confusion (rows = annotated, columns = predicted)

| | VERIFIED | PARTIALLY_VERIFIED | CONTRADICTED | UNVERIFIED |
|---|---|---|---|---|
| VERIFIED | 58 | 7 | 0 | 5 |
| PARTIALLY_VERIFIED | 0 | 5 | 2 | 2 |

Measurement `safety`: run `2df6a54e-57d1-4747-972c-f23010e29f49`, code `93cd752e3a530aa86fc4cfb3524f362dad464b49`, started `2026-10-29T01:40:52+00:00`.

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

## Provenance and reproducibility warnings

- retrieval: attention_is_all_you_need: repository revision is not a full immutable commit SHA
- retrieval: attention_is_all_you_need: paper SHA-256 is missing or a placeholder
- retrieval: bert: repository revision is not a full immutable commit SHA
- retrieval: bert: paper SHA-256 is missing or a placeholder
- retrieval: ddim: repository revision is not a full immutable commit SHA
- retrieval: ddim: paper SHA-256 is missing or a placeholder
- retrieval: ddpm: repository revision is not a full immutable commit SHA
- retrieval: ddpm: paper SHA-256 is missing or a placeholder
- retrieval: dpo: repository revision is not a full immutable commit SHA
- retrieval: dpo: paper SHA-256 is missing or a placeholder
- retrieval: lora: repository revision is not a full immutable commit SHA
- retrieval: lora: paper SHA-256 is missing or a placeholder
- retrieval: mae: repository revision is not a full immutable commit SHA
- retrieval: mae: paper SHA-256 is missing or a placeholder
- retrieval: reflexion: repository revision is not a full immutable commit SHA
- retrieval: reflexion: paper SHA-256 is missing or a placeholder
- verification: attention_is_all_you_need: repository revision is not a full immutable commit SHA
- verification: attention_is_all_you_need: paper SHA-256 is missing or a placeholder
- verification: bert: repository revision is not a full immutable commit SHA
- verification: bert: paper SHA-256 is missing or a placeholder
- verification: ddim: repository revision is not a full immutable commit SHA
- verification: ddim: paper SHA-256 is missing or a placeholder
- verification: ddpm: repository revision is not a full immutable commit SHA
- verification: ddpm: paper SHA-256 is missing or a placeholder
- verification: dpo: repository revision is not a full immutable commit SHA
- verification: dpo: paper SHA-256 is missing or a placeholder
- verification: lora: repository revision is not a full immutable commit SHA
- verification: lora: paper SHA-256 is missing or a placeholder
- verification: mae: repository revision is not a full immutable commit SHA
- verification: mae: paper SHA-256 is missing or a placeholder
- verification: reflexion: repository revision is not a full immutable commit SHA
- verification: reflexion: paper SHA-256 is missing or a placeholder
