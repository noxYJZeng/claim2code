# Claim2Code — Evaluation Report

Report rendered 2026-10-30T08:14:55+00:00. Rendering does not run the models.
Measurement code and model identities below come only from stored evaluator provenance.

Measurement `retrieval`: run `c1431184-fd51-49da-8dce-ea33d4e7c85c`, code `d38c62dea5ea66f464ccd8af915c44da129a937e`, started `2026-10-29T23:44:56+00:00`.

Recorded chat `qwen3.8-27b-mlx`; embeddings `text-embedding-nomic-embed-text-v1.5`.

## Retrieval

Recall@k is the fraction of annotated relevant symbols found; Hit@k needs any one relevant symbol.
Each annotated class counts once, accepting that class or a method as an alternative location.
Unresolved annotated symbols remain in the recall denominator.

### attention_is_all_you_need

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.49 | 0.85 | 0.58 | 0.92 | 0.704 |
| lexical_only | 0.28 | 0.57 | 0.33 | 0.67 | 0.426 |
| hybrid_claim2code | 0.28 | 0.78 | 0.33 | 0.83 | 0.517 |
| claim2code_rerank | 0.82 | 0.96 | 0.92 | 1.00 | 0.944 |

### bert

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.41 | 0.48 | 0.45 | 0.64 | 0.518 |
| lexical_only | 0.12 | 0.39 | 0.27 | 0.55 | 0.379 |
| hybrid_claim2code | 0.08 | 0.64 | 0.18 | 0.82 | 0.403 |
| claim2code_rerank | 0.53 | 0.86 | 0.82 | 1.00 | 0.909 |

### clip

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.60 | 0.60 | 0.80 | 0.80 | 0.800 |
| lexical_only | 0.10 | 0.10 | 0.20 | 0.20 | 0.200 |
| hybrid_claim2code | 0.10 | 0.60 | 0.20 | 0.80 | 0.340 |
| claim2code_rerank | 0.60 | 0.80 | 0.80 | 1.00 | 0.900 |

### ddim

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.06 | 0.50 | 0.12 | 0.62 | 0.279 |
| lexical_only | 0.31 | 0.38 | 0.38 | 0.38 | 0.375 |
| hybrid_claim2code | 0.31 | 0.44 | 0.38 | 0.50 | 0.417 |
| claim2code_rerank | 0.60 | 0.79 | 0.88 | 1.00 | 0.906 |

### ddpm

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.33 | 0.54 | 0.50 | 0.67 | 0.583 |
| lexical_only | 0.04 | 0.42 | 0.08 | 0.50 | 0.242 |
| hybrid_claim2code | 0.33 | 0.62 | 0.42 | 0.75 | 0.569 |
| claim2code_rerank | 0.42 | 1.00 | 0.58 | 1.00 | 0.778 |

### deit

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.00 | 0.40 | 0.00 | 0.40 | 0.167 |
| lexical_only | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| hybrid_claim2code | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| claim2code_rerank | 0.60 | 0.80 | 0.60 | 0.80 | 0.700 |

### detr

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.50 | 0.90 | 0.60 | 1.00 | 0.717 |
| lexical_only | 0.00 | 0.70 | 0.00 | 0.80 | 0.290 |
| hybrid_claim2code | 0.00 | 0.50 | 0.00 | 0.60 | 0.267 |
| claim2code_rerank | 0.70 | 1.00 | 0.80 | 1.00 | 0.900 |

### dino

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.40 | 0.80 | 0.40 | 0.80 | 0.600 |
| lexical_only | 0.40 | 0.50 | 0.40 | 0.60 | 0.450 |
| hybrid_claim2code | 0.20 | 0.50 | 0.20 | 0.60 | 0.307 |
| claim2code_rerank | 0.90 | 1.00 | 1.00 | 1.00 | 1.000 |

### dpo

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.33 | 0.72 | 0.44 | 0.78 | 0.611 |
| lexical_only | 0.39 | 0.61 | 0.44 | 0.67 | 0.522 |
| hybrid_claim2code | 0.28 | 0.78 | 0.33 | 0.78 | 0.481 |
| claim2code_rerank | 0.78 | 0.89 | 0.89 | 1.00 | 0.944 |

### lora

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.00 | 0.35 | 0.00 | 0.40 | 0.200 |
| lexical_only | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| hybrid_claim2code | 0.20 | 0.50 | 0.20 | 0.50 | 0.317 |
| claim2code_rerank | 0.45 | 0.75 | 0.60 | 0.90 | 0.725 |

### mae

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.39 | 0.75 | 0.44 | 0.89 | 0.611 |
| lexical_only | 0.17 | 0.50 | 0.22 | 0.89 | 0.396 |
| hybrid_claim2code | 0.24 | 0.64 | 0.44 | 0.89 | 0.620 |
| claim2code_rerank | 0.43 | 0.92 | 0.56 | 1.00 | 0.759 |

### moco

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.27 | 0.93 | 0.40 | 1.00 | 0.650 |
| lexical_only | 0.10 | 0.43 | 0.20 | 0.60 | 0.400 |
| hybrid_claim2code | 0.00 | 0.63 | 0.00 | 0.80 | 0.257 |
| claim2code_rerank | 0.77 | 1.00 | 1.00 | 1.00 | 1.000 |

### reflexion

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.12 | 0.19 | 0.12 | 0.25 | 0.188 |
| lexical_only | 0.06 | 0.23 | 0.12 | 0.50 | 0.281 |
| hybrid_claim2code | 0.06 | 0.46 | 0.12 | 0.75 | 0.323 |
| claim2code_rerank | 0.27 | 0.69 | 0.62 | 0.88 | 0.729 |

### sam

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.20 | 0.20 | 0.20 | 0.20 | 0.200 |
| lexical_only | 0.20 | 0.40 | 0.20 | 0.60 | 0.317 |
| hybrid_claim2code | 0.20 | 0.50 | 0.20 | 0.80 | 0.330 |
| claim2code_rerank | 0.70 | 0.80 | 1.00 | 1.00 | 1.000 |

### whisper

| Baseline | Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| embedding_only | 0.00 | 0.30 | 0.00 | 0.40 | 0.117 |
| lexical_only | 0.20 | 0.20 | 0.20 | 0.20 | 0.200 |
| hybrid_claim2code | 0.20 | 0.40 | 0.20 | 0.40 | 0.267 |
| claim2code_rerank | 0.60 | 0.80 | 0.80 | 1.00 | 0.900 |

### Aggregate

| Baseline | Avg R@1 | Avg R@5 | Avg Hit@1 | Avg Hit@5 | Avg MRR |
|---|---|---|---|---|---|
| embedding_only | 0.27 | 0.57 | 0.34 | 0.65 | 0.463 |
| lexical_only | 0.16 | 0.36 | 0.20 | 0.48 | 0.299 |
| hybrid_claim2code | 0.17 | 0.53 | 0.21 | 0.65 | 0.361 |
| claim2code_rerank | 0.61 | 0.87 | 0.79 | 0.97 | 0.873 |

Aggregates are macro averages over datasets. Baselines: embedding_only, lexical_only, hybrid_claim2code, claim2code_rerank.

Measurement `verification`: run `35a913ab-72c5-477a-a3ee-099e16918811`, code `d38c62dea5ea66f464ccd8af915c44da129a937e`, started `2026-10-30T01:26:26+00:00`.

Recorded chat `qwen3.8-27b-mlx`; embeddings `text-embedding-nomic-embed-text-v1.5`.

## Verification

114 curated annotated claims, claim-to-code pipeline (hybrid retrieval top_k=8 → deterministic checkers → LLM verdict).

The input is an annotation's claim text, also used as its paper quote. This does not measure PDF extraction accuracy or verify the paper citation against a PDF.

| Metric | Value |
|---|---|
| Status accuracy (exact, on observed labels) | 91.2% |
| Majority-class baseline (always VERIFIED) | 92.1% |
| Macro F1 (gold-present labels only) | 91.5% |
| Balanced accuracy (gold-present labels only) | 85.1% |
| Support rate (VERIFIED or PARTIALLY_VERIFIED) | 92.1% |
| Discrepancy precision | 100.0% (7 flagged) |
| Discrepancy recall | 77.8% (9 annotated) |
| Grounding rate (cited code exists in the cited file) | 100.0% (355 citations) |
| Line accuracy (±2-line grading tolerance) | 100.0% |
| Exact line accuracy (no tolerance) | 100.0% |
| Evidence hit rate (supported verdicts citing an annotated symbol) | 99.0% |
| LLM errors | 0 |

### Per pair

| Pair | Claims | Accuracy | Support | Disc. P | Disc. R | Grounding |
|---|---|---|---|---|---|---|
| attention_is_all_you_need | 12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| bert | 11 | 81.8% | 81.8% | 100.0% | 100.0% | 100.0% |
| clip | 5 | 100.0% | 100.0% | n/a | n/a | 100.0% |
| ddim | 8 | 75.0% | 75.0% | n/a | n/a | 100.0% |
| ddpm | 12 | 91.7% | 100.0% | n/a | 0.0% | 100.0% |
| deit | 5 | 100.0% | 100.0% | n/a | n/a | 100.0% |
| detr | 5 | 100.0% | 100.0% | n/a | n/a | 100.0% |
| dino | 5 | 100.0% | 100.0% | n/a | n/a | 100.0% |
| dpo | 9 | 88.9% | 88.9% | 100.0% | 100.0% | 100.0% |
| lora | 10 | 80.0% | 80.0% | 100.0% | 100.0% | 100.0% |
| mae | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| moco | 5 | 100.0% | 100.0% | n/a | n/a | 100.0% |
| reflexion | 8 | 75.0% | 75.0% | 100.0% | 66.7% | 100.0% |
| sam | 5 | 100.0% | 100.0% | n/a | n/a | 100.0% |
| whisper | 5 | 100.0% | 100.0% | n/a | n/a | 100.0% |

### Label coverage

| Gold label | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| VERIFIED | 105 | 99.0% | 92.4% | 95.6% |
| PARTIALLY_VERIFIED | 9 | 100.0% | 77.8% | 87.5% |
| CONTRADICTED | 0 | n/a | n/a | n/a |
| UNVERIFIED | 0 | 0.0% | n/a | n/a |

No gold examples for: CONTRADICTED, UNVERIFIED. These results do not establish four-way classification performance.

Code grounding measures whether a quote exists, not whether the verdict is correct.

### Confusion (rows = annotated, columns = predicted)

| | VERIFIED | PARTIALLY_VERIFIED | CONTRADICTED | UNVERIFIED |
|---|---|---|---|---|
| VERIFIED | 97 | 0 | 0 | 8 |
| PARTIALLY_VERIFIED | 1 | 7 | 0 | 1 |

Measurement `safety`: run `0de03af5-c08a-4e94-a86a-a547b46eb49b`, code `d38c62dea5ea66f464ccd8af915c44da129a937e`, started `2026-10-30T08:14:54+00:00`.

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

- retrieval: attention_is_all_you_need: paper SHA-256 is missing or a placeholder
- retrieval: bert: paper SHA-256 is missing or a placeholder
- retrieval: clip: paper SHA-256 is missing or a placeholder
- retrieval: ddim: paper SHA-256 is missing or a placeholder
- retrieval: ddpm: paper SHA-256 is missing or a placeholder
- retrieval: deit: paper SHA-256 is missing or a placeholder
- retrieval: detr: paper SHA-256 is missing or a placeholder
- retrieval: dino: paper SHA-256 is missing or a placeholder
- retrieval: dpo: paper SHA-256 is missing or a placeholder
- retrieval: lora: paper SHA-256 is missing or a placeholder
- retrieval: mae: paper SHA-256 is missing or a placeholder
- retrieval: moco: paper SHA-256 is missing or a placeholder
- retrieval: reflexion: paper SHA-256 is missing or a placeholder
- retrieval: sam: paper SHA-256 is missing or a placeholder
- retrieval: whisper: paper SHA-256 is missing or a placeholder
- verification: attention_is_all_you_need: paper SHA-256 is missing or a placeholder
- verification: bert: paper SHA-256 is missing or a placeholder
- verification: clip: paper SHA-256 is missing or a placeholder
- verification: ddim: paper SHA-256 is missing or a placeholder
- verification: ddpm: paper SHA-256 is missing or a placeholder
- verification: deit: paper SHA-256 is missing or a placeholder
- verification: detr: paper SHA-256 is missing or a placeholder
- verification: dino: paper SHA-256 is missing or a placeholder
- verification: dpo: paper SHA-256 is missing or a placeholder
- verification: lora: paper SHA-256 is missing or a placeholder
- verification: mae: paper SHA-256 is missing or a placeholder
- verification: moco: paper SHA-256 is missing or a placeholder
- verification: reflexion: paper SHA-256 is missing or a placeholder
- verification: sam: paper SHA-256 is missing or a placeholder
- verification: whisper: paper SHA-256 is missing or a placeholder
