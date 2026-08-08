# Claim2Code — Data Flow

> The end-to-end pipeline as one picture. If a new component doesn't appear
> here, it's not part of v1.

---

## 1. The big picture

```
                       ┌─────────────────────────────────────────┐
                       │             React UI (Vite)              │
                       │   upload PDF / paste URL / view report  │
                       └────────────┬───────────────┬─────────────┘
                                  REST │            │ progress
                                       ▼            ▼
                       ┌─────────────────────────────────────────┐
                       │             FastAPI backend              │
                       │    /papers /repositories /analyses      │
                       └────────────┬───────────────┬─────────────┘
                                    │               │
            ┌───────────────────────┼───────────────┼──────────────────┐
            │                       │               │                  │
            ▼                       ▼               ▼                  ▼
   ┌─────────────────┐   ┌──────────────────┐ ┌────────────┐ ┌──────────────┐
   │ Paper Processor │   │Repository Analyzer│ │Literature  │ │  Background   │
   │                 │   │                    │ │  Search    │ │  Workflow     │
   │  parse → claims │   │ clone → AST → embed│ │            │ │  (FastAPI BG) │
   └────────┬────────┘   └─────────┬──────────┘ └─────┬──────┘ └──────┬───────┘
            │                      │                  │              │
            └──────────┬───────────┘                  │              │
                       ▼                              │              │
              ┌─────────────────┐                     │              │
              │ Hybrid Claim    │                     │              │
              │ Retriever       │                     │              │
              │ sem+lex+struct  │                     │              │
              └────────┬────────┘                     │              │
                       ▼                              │              │
              ┌─────────────────┐                     │              │
              │ Claim Verifier  │◀────────────────────┴──────────────┘
              │ + discrepancy   │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐         ┌──────────────────────────┐
              │ Verification    │────────▶│ OpenTelemetry traces     │
              │ Report          │         │ + per-step metrics       │
              └────────┬────────┘         └──────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ PostgreSQL +    │
              │ pgvector        │
              └─────────────────┘
```

## 2. Step-by-step (what each box does)

| # | Step | Determinism | LLM? |
|---|---|---|---|
| 1 | **INGEST_PAPER** — validate PDF, save to disk, hash | 100% | — |
| 2 | **PARSE_PAPER** — PyMuPDF → text blocks / sections / equations / tables | 100% | — |
| 3 | **EXTRACT_CLAIMS** — LLM extracts structured `Claim` objects, Pydantic-validated | mixed | yes |
| 4 | **INGEST_REPOSITORY** — validate URL, `git clone`, filter tree, size limits | 100% | — |
| 5 | **PARSE_CODE** — tree-sitter AST → `CodeSymbol`s + symbol graph | 100% | — |
| 6 | **INDEX_CODE** — embed each symbol's structured representation → pgvector | 100% | — |
| 7 | **RETRIEVE_CODE** — hybrid ranker (semantic + lexical + structural) per claim | 100% | — |
| 8 | **VERIFY_CLAIMS** — LLM judges each claim against retrieved symbols + paper evidence | mixed | yes |
| 9 | **DETECT_DISCREPANCIES** — deterministic hyperparam / config / model-name / dataset / metric comparisons + LLM-assisted ambiguity | mostly | partial |
| 10 | **GENERATE_REPORT** — render the verification report + discrepancy table | 100% | — |
| (opt) | **RELATED_WORK** — scholarly API search + structured comparison | mixed | yes |

## 3. Data classes that flow through (PDF §10)

```
PDF ──► Paper
         ├── sections: list[PaperSection]   (page_anchored)
         ├── equations: list[Equation]
         └── tables: list[Table]

Paper ──► list[Claim]                      (LLM, Pydantic-validated)
         ├── text
         ├── claim_type ∈ {METHOD, IMPLEMENTATION, …}
         ├── evidence_locations: list[PaperEvidence]
         └── confidence: float

Repo ──► list[CodeSymbol]                  (AST, deterministic)
         ├── file_path, start_line, end_line
         ├── signature, docstring
         └── imports, calls

(list[Claim], list[CodeSymbol])
       ──► list[ClaimCodeMapping]          (hybrid retriever)
         ├── semantic_score
         ├── lexical_score
         ├── structural_score
         └── final_score = 0.55·sem + 0.25·lex + 0.20·struct

(list[Claim], list[Mapping])
       ──► list[Verification]              (LLM verifier)
         ├── status ∈ {VERIFIED, PARTIALLY_VERIFIED,
         │              CONTRADICTED, UNVERIFIED}
         ├── confidence, explanation
         ├── paper_evidence
         ├── code_evidence
         └── discrepancies: list[Discrepancy]

list[Verification] ──► Verification Report (grounded)
```

## 4. State persistence

After every major step, persist `AnalysisState` to Postgres. If a step
crashes, the next request resumes from the last completed step. **No work
is recomputed; no claim is half-verified.**

```python
class AnalysisState:
    analysis_id: UUID
    paper_status: str
    repository_status: str
    claims: list[Claim]
    symbols: list[CodeSymbol]
    mappings: list[ClaimCodeMapping]
    verifications: list[Verification]
    errors: list[WorkflowError]
    started_at: datetime
    completed_at: datetime | None
```

## 5. What the user sees

A claim card (from PDF §6 example):

```
┌─ Claim C03 ──────────────────────────────────────────────────────┐
│ "Adaptive weighting improves long-horizon task success over       │
│  uniform weighting."                                              │
│                                                                  │
│ Type:  PERFORMANCE_CLAIM            Confidence: 0.92             │
│                                                                  │
│ Paper component: Adaptive token weighting                        │
│ Mapped implementation:  src/trainer/loss.py                      │
│ Function:  compute_adaptive_weight()    Lines: 118–164           │
│ Mapping confidence: 0.87                                         │
│                                                                  │
│ Status: PARTIALLY_VERIFIED                                       │
│ Reason: Repo implements weighting from Eq.7 but default          │
│         clipping threshold differs from paper.                   │
│                                                                  │
│ Discrepancy: β = 0.1 (paper) vs β = 0.05 (repo, train.yaml:43)   │
│             Severity: MEDIUM                                     │
└──────────────────────────────────────────────────────────────────┘
```

That's the shape every claim card in v1 should conform to.