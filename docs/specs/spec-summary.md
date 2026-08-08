# Claim2Code — Spec Summary

> Source: `Claim2Code.pdf` (41 pages, AI-Research-Agent design doc).
> This file condenses it into a single-page reference so we don't have to
> re-read the PDF every time we open an editor.

---

## 1. The product question

> **"Does the implementation actually correspond to what the paper claims,
> and where is the evidence?"**

Not a paper chatbot. Not a research-assistant autocomplete. **A verifier.**

## 2. Inputs

1. ML/AI paper (PDF, English, digitally generated)
2. GitHub repository URL (public, Python, PyTorch)
3. (Optional) Research topic or claim for related-work analysis

## 3. V1 14 core capabilities

| # | Capability |
|---|---|
| 1 | Paper parsing (PDF → text blocks / sections / equations / tables) |
| 2 | Structured claim extraction (LLM, Pydantic-validated) |
| 3 | Equation / algorithm extraction |
| 4 | Experimental configuration extraction |
| 5 | GitHub repository ingestion (clone + tree filter + size limits) |
| 6 | AST-based Python analysis (tree-sitter preferred) |
| 7 | Embedding-based semantic code retrieval |
| 8 | **Hybrid claim-to-code retrieval** (semantic + lexical + structural) |
| 9 | Claim-to-code verification (verdict + evidence) |
| 10 | Discrepancy detection (8 types, deterministic where possible) |
| 11 | Grounded report generation |
| 12 | Observability (OTel, latency, token, retrieval metrics) |
| 13 | Evaluation framework (Recall@1/5, MRR, verification accuracy, grounding rate) |
| 14 | Responsible-AI guardrails (secret redaction, prompt-injection defense) |

## 4. V1 explicit non-goals

- Reproducing experiments / training large models
- Executing untrusted research code
- Supporting every programming language
- Solving arbitrary dependency environments
- Auto-proving math equivalences
- Auto-claiming novelty
- Auto-submitting PRs / modifying repos
- Building an autonomous scientist
- >2–3 LLM-driven workflow components

## 5. Core data models (PDF page 12)

```python
class Paper:                id, title, authors, abstract, sections, equations, tables
class PaperSection:         section_id, title, page_start, page_end, text
class Claim:                claim_id, text, claim_type, evidence_locations,
                            entities, metrics, baselines, hyperparameters, confidence
class CodeSymbol:           symbol_id, repository_id, file_path, symbol_name,
                            symbol_type, start_line, end_line, signature, source,
                            imports, calls
class ClaimCodeMapping:     claim_id, symbol_id, semantic_score, lexical_score,
                            structural_score, final_score
class Verification:         claim_id, status, confidence, explanation,
                            paper_evidence, code_evidence, discrepancies
```

`ClaimType ∈ {METHOD, IMPLEMENTATION, PERFORMANCE, ABLATION, HYPERPARAMETER,
DATASET, COMPLEXITY, TRAINING, EVALUATION}`

`VerificationStatus ∈ {VERIFIED, PARTIALLY_VERIFIED, CONTRADICTED, UNVERIFIED}`

`Discrepancy.type ∈ {HYPERPARAMETER_MISMATCH, IMPLEMENTATION_MISSING,
UNDOCUMENTED_IMPLEMENTATION, OBJECTIVE_MISMATCH, DATASET_MISMATCH,
METRIC_MISMATCH, CONFIGURATION_MISMATCH, AMBIGUOUS}`

`Discrepancy.severity ∈ {LOW, MEDIUM, HIGH}`

## 6. The 10-phase dev roadmap (PDF page 34+)

| # | Phase | Acceptance |
|---|---|---|
| 1 | Project Foundation | `docker compose up` starts full local stack with health endpoint |
| 2 | Paper Intelligence | Upload paper → 5–15 structured technical claims with paper evidence |
| 3 | Repository Intelligence | Repo page shows functions/classes/configs with exact line numbers |
| 4 | Semantic Indexing | Given claim → Top-5 candidate code symbols via embeddings |
| 5 | Hybrid Retrieval | lexical + structural + combined ranker; Recall@1/5/MRR vs baseline |
| 6 | Verification | Every result has exact evidence or UNVERIFIED |
| 7 | Related Work | Selected claim → up to 5 related papers with explicit differences |
| 8 | Responsible AI | All adversarial fixtures blocked |
| 9 | Observability | One analysis → one trace tree |
| 10 | Evaluation | `evaluation/results.json` + Markdown report from 8–12 papers |

## 7. The workflow (PDF page 22)

```
INGEST_PAPER
    ↓
PARSE_PAPER
    ↓
EXTRACT_CLAIMS
        ↓                  (parallel branch)
INGEST_REPOSITORY
    ↓
PARSE_CODE
    ↓
INDEX_CODE
    ↓
RETRIEVE_CODE
    ↓
VERIFY_CLAIMS
    ↓
DETECT_DISCREPANCIES
    ↓
GENERATE_REPORT
```

Related-work runs **after** claim extraction.

## 8. Hybrid retrieval formula (PDF page 18)

```
S_final = 0.55 · S_semantic
        + 0.25 · S_lexical
        + 0.20 · S_structural
```

Top-K = 5. These are **implementation defaults, not research claims** —
tune with the evaluation dataset.

## 9. Target results (PDF page 38)

> These are project targets, not fabricated values.

| Metric | Target |
|---|---|
| Paper/repo pairs in eval | ~10 |
| Manually annotated claims | ~50 |
| Embedding-only Recall@5 | 70–80% |
| **Hybrid Recall@5** | **85–95%** |
| Verification accuracy | 80–90% |
| Grounded finding rate | >95% |
| Prompt-injection blocked | 100% |
| End-to-end analysis | <2 min (excl. external API) |

## 10. Verifier's crucial rule (PDF page 19)

> The verifier **CANNOT** use model knowledge as evidence. Every statement
> must be supported by **paper evidence AND code evidence**. If insufficient:
> `UNVERIFIED`.