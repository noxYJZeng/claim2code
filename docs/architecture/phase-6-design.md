# Phase 6 — Verification Design

> Goal (from `phase-roadmap.md:115-129`): every claim gets a
> verdict + paper evidence + code evidence + discrepancies.
>
> Acceptance (verbatim): every verification has paper evidence
> AND code evidence, or `UNVERIFIED`.

## 1. What ships

| Layer | Change |
|---|---|
| `apps/api/claim2code_api/verification/__init__.py` | New: module docstring + `__all__` |
| `apps/api/claim2code_api/verification/exceptions.py` | New: `VerificationError` + `VerificationLLMError` + `VerificationConfigError` + `VerificationNotFoundError` |
| `apps/api/claim2code_api/verification/schemas.py` | New: `VerificationStatus`, `DiscrepancyType`, `DiscrepancySeverity` (StrEnums); `CodeEvidence`; `Discrepancy`; `Verification`; `VerificationRunRequest`, `VerificationRunResponse`, `VerificationListResponse` |
| `apps/api/claim2code_api/verification/normalizers.py` | New: `ModelNameNormalizer`, `DatasetNormalizer`, `MetricNameComparator` (each loads `verification/aliases/<thing>.json`) |
| `apps/api/claim2code_api/verification/aliases/{models,datasets,metrics}.json` | New: static alias dicts (e.g. `"qwen2.5-7b" → ["Qwen/Qwen2.5-7B", "qwen2.5-7b", "Qwen2.5-7B"]`) |
| `apps/api/claim2code_api/verification/hyperparameter_checker.py` | New: regex extract + additive/multiplicative numeric tolerance; emits `HYPERPARAMETER_MISMATCH` (HIGH) |
| `apps/api/claim2code_api/verification/discrepancies.py` | New: `DiscrepancyClassifier` — merge deterministic + LLM discrepancies, apply severity rules |
| `apps/api/claim2code_api/verification/verifier.py` | New: `compose_verification(...)` pure function (no I/O) + `extract_verdict(...)` thin wrapper around `LLMClient.verdict_for_claim` |
| `apps/api/claim2code_api/verification/workflow.py` | New: `run_verification_workflow(*, db, analysis_id, claim_ids=None, top_k=5)` orchestrating the per-claim pipeline; returns `VerificationWorkflowResult` dataclass |
| `apps/api/claim2code_api/verification/router.py` | New: `POST /analyses/{analysis_id}/verifications` (run) + `GET /analyses/{analysis_id}/verifications` (read). `VerificationError` → `HTTPException` mapping |
| `apps/api/claim2code_api/llm/client.py` | Edit: add `verdict_for_claim(...)` method + `_VERIFY_SYSTEM_PROMPT` constant (same call shape as `extract_claims`) |
| `apps/api/claim2code_api/app/main.py` | Edit: `app.include_router(verification_router)` |
| `apps/api/tests/test_verification_schemas.py` | New: ~14 tests — Pydantic validation, `extra="forbid"`, StrEnum membership |
| `apps/api/tests/test_hyperparameter_checker.py` | New: ~12 tests — regex extract + additive + multiplicative tolerance |
| `apps/api/tests/test_normalizers.py` | New: ~18 tests — 3 normalizers × (alias hit / fuzzy hit / miss / round-trip) |
| `apps/api/tests/test_discrepancies.py` | New: ~10 tests — 8 types emitted correctly + LLM/deterministic merge |
| `apps/api/tests/test_verifier.py` | New: ~8 tests — LLM call shape + JSON schema + error mapping (mock LLM) |
| `apps/api/tests/test_verification_workflow.py` | New: ~8 tests — pipeline ordering + UNVERIFIED coercion + JSONB write-back + **acceptance test** |
| `apps/api/tests/integration/test_verification_e2e.py` | New: ~4 tests — live Postgres + mock LLM + 5-claim/5-symbol round-trip |
| `README.md` | Edit: phase roadmap table — Phase 6 row → ✅ |
| `evaluation/README.md` | Edit: Phase 6 status line updated |

**No new Python deps.** `pg_trgm` ships with the base `pgvector/pgvector:pg16` image. We use stdlib `difflib` for fuzzy name matching.

**No new env vars.** Phase 6 only reuses the existing OpenAI + Postgres settings.

**No new migration.** `AnalysisState.verifications: JSONB` was pre-staged in Phase 1 (`apps/api/claim2code_api/db/models.py:103`); Phase 6 only writes to it.

**No frontend task.** Phase 7 / 8 may bring a `VerificationList` panel; Phase 6 ships the API + JSONB persistence only.

## 2. Key decisions

### 2.1 JSONB stays — no dedicated `Verification` table

`AnalysisState.verifications: JSONB` was deliberately pre-staged in Phase 1:

```python
# apps/api/claim2code_api/db/models.py:75-77
The wide JSONB columns (claims, symbols, …) are deliberate
for Phase 1 — the schema of those nested objects will only stabilise
in Phase 2-6, and JSONB lets us evolve without DDL churn.
```

Phase 6 is where the schema finally stabilises (4 statuses, 8
discrepancy types, 3 severities — all defined by
`docs/specs/spec-summary.md:65-68`). So *could* we normalise
now? Yes, but:

- **Cost** — new migration + ORM model + relationship wiring +
  test fixtures + relationship assertions.
- **Phase 6 benefit** — nothing in this phase needs per-status
  aggregation or FK joins.
- **Phase 10 eval benefit** — yes, but Phase 10 is the natural
  place to revisit.

We stay in JSONB for Phase 6. Phase 10 eval can revisit.

The verifications list is sorted by `claim_id` on persist so the
GET response is deterministic across runs.

### 2.2 POST + GET endpoints (not GET-only)

The roadmap names `GET /analyses/{id}/verifications`. But without
a trigger, verifications never run. Three options:

  - **A.** Auto-run inside `papers/workflow.py` after claim
    extraction.
  - **B.** POST + GET (mirrors `POST /papers` / `GET /papers/{id}`).
  - **C.** Sync inside the GET (run-on-demand, cache).

Pick **B**. Reasoning:

  - **A** couples paper parsing to LLM availability (the papers
    workflow test fixtures would have to mock `LLMClient`); and
    Phase 2 papers workflow is already 200 lines.
  - **C** is semantically weird (GET should be idempotent) and
    has no precedent in this codebase.
  - **B** matches the existing convention: `POST /papers` →
    `GET /papers/{id}`, `POST /repositories` → `GET
    /repositories/{id}/retrieve`.

The POST body is optional:

```python
class VerificationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_ids: list[str] | None = None    # None = all claims
    top_k: int = Field(default=5, ge=1, le=20)
```

Empty body / no body → verify all claims, `top_k=5`. The frontend
can re-POST after editing to recompute.

### 2.3 Verifier LLM extracts structured hyperparameters

The spec doc (`spec-summary.md:57-58`) lists
`entities, metrics, baselines, hyperparameters` as `Claim` fields.
They don't exist on the Pydantic model today
(`papers/schemas.py:75-90`) and Phase 2's LLM isn't asked for
them.

Three options for getting structured hyperparameters:

  - **(a)** Extend `Claim` + Phase 2 prompt + retest 3 shipped
    paths. Expensive.
  - **(b)** Regex-only — extract from paper evidence quote.
    Misses values that aren't quoted verbatim.
  - **(c)** Verifier LLM extracts them as part of its own
    response (it has both paper evidence + candidate signatures
    in scope).

Pick **(c)**. The verifier's `Verification` response schema gains
a `hyperparameters: list[HyperparameterEvidence]` field; the
deterministic `HyperparameterChecker` cross-validates against the
LLM's numbers. Disagreement → `AMBIGUOUS` discrepancy.

Cost: one extra field in the LLM response schema. No Phase 2
change. If a future phase needs `Claim.hyperparameters`, it can
backfill from the persisted verification list.

### 2.4 Verifier LLM system prompt — the "no model knowledge" rule

`spec-summary.md:148-150`:

> The verifier **CANNOT** use model knowledge as evidence. Every
> statement must be supported by **paper evidence AND code
> evidence**. If insufficient: `UNVERIFIED`.

This is the **most important single constraint** in the phase.
Operationalised in the system prompt as four rules:

```
1. **No model knowledge.** Every piece of paper evidence you cite
   must be a verbatim substring of paper_evidence.quote. Every
   piece of code evidence you cite must reference a specific
   candidate from the list provided (its symbol_id,
   file:line, signature, or docstring). You MAY NOT use knowledge
   of the paper, the library, or the algorithm beyond what is
   explicitly given in the inputs.

2. **One of four verdicts:**
     VERIFIED, PARTIALLY_VERIFIED, CONTRADICTED, UNVERIFIED.
   - VERIFIED only when ≥1 candidate's signature/docstring
     directly implements the claim and no discrepancies are HIGH.
   - PARTIALLY_VERIFIED when ≥1 candidate is relevant but
     discrepancies (LOW/MEDIUM) exist.
   - CONTRADICTED when a candidate clearly contradicts the claim
     or a HIGH-severity discrepancy exists.
   - UNVERIFIED when no candidate is relevant.

3. **Extract structured hyperparameters** for any hyperparameter
   mentioned in the claim or quote (learning_rate, batch_size,
   temperature, beta, gamma, num_epochs, hidden_size). We use
   this for the deterministic checker.

4. **Cite deterministically.** Each code_evidence entry must
   include symbol_id (UUID string), quote (verbatim from
   candidate.signature or candidate.docstring), and file:line.
```

The prompt is committed in `llm/client.py` as `_VERIFY_SYSTEM_PROMPT`
next to the existing `_EXTRACT_SYSTEM_PROMPT`.

The response schema is enforced via OpenAI's Structured Outputs
(`response_format={"type": "json_schema", ...}`) — same mechanism
Phase 2 uses for `ClaimList`.

### 2.5 The 4-status rule (operational)

```python
if not verification.code_evidence:                  # structural guard
    verification.status = UNVERIFIED
    verification.explanation = "No code candidate cited."
```

| Status                  | paper_evidence | code_evidence (≥1) | Discrepancies |
|-------------------------|----------------|--------------------|---------------|
| `VERIFIED`              | verbatim quote | signature+file:line from top-5 | no HIGH-severity |
| `PARTIALLY_VERIFIED`    | verbatim quote | signature+file:line from top-5 | ≥1 LOW or MEDIUM |
| `CONTRADICTED`          | verbatim quote | signature+file:line from top-5 | ≥1 HIGH OR LLM=CONTRADICTED |
| `UNVERIFIED`            | any            | none / none of top-5 cited      | n/a            |

**The structural acceptance check is enforced at the
`compose_verification` layer** — a verification with no
`code_evidence` entries is auto-coerced to `UNVERIFIED`
regardless of what the LLM said. This makes the acceptance
criterion testable as a single assertion:

```python
def test_every_verification_has_evidence_or_unverified(verifications):
    for v in verifications:
        assert v.paper_evidence is not None
        assert v.code_evidence or v.status == VerificationStatus.UNVERIFIED
```

### 2.6 Deterministic first, LLM residual

```
for claim in claims:
  candidates = hybrid_ranker(db, repository_id, claim.text, top_k)   # Phase 5

  deterministic = []
  deterministic += HyperparameterChecker.compare(claim, candidates)
  deterministic += ModelNameNormalizer.compare(claim, candidates)
  deterministic += DatasetNormalizer.compare(claim, candidates)
  deterministic += MetricNameComparator.compare(claim, candidates)

  if any(d.severity == HIGH for d in deterministic):
      llm_verdict = None   # already decisive
  else:
      llm_verdict = llm.verdict_for_claim(claim, ..., candidates, deterministic)

  verification = compose_verification(claim, candidates, deterministic, llm_verdict)
  if not verification.code_evidence:
      verification.status = UNVERIFIED
  persist
```

Why deterministic first:

  - **Reproducibility** — deterministic output is unit-testable
    without LLM mocking.
  - **Bounded LLM usage** — for claims where the hyperparameter
    already contradicts, we don't burn an OpenAI call.
  - **Matches Phase 5 pattern** — Phase 5's hybrid_ranker fuses
    3 legs and the orchestrator is the unit of composition. Same
    shape, just a different number of "legs".

### 2.7 The 8 discrepancy types + 3 severities

Per `spec-summary.md:65-68`:

| Type                          | Emitted by                | Default severity |
|-------------------------------|---------------------------|------------------|
| `HYPERPARAMETER_MISMATCH`     | HyperparameterChecker     | HIGH             |
| `IMPLEMENTATION_MISSING`      | LLM (no candidate)        | MEDIUM           |
| `UNDOCUMENTED_IMPLEMENTATION` | LLM (candidate, no claim) | LOW              |
| `OBJECTIVE_MISMATCH`          | LLM                       | MEDIUM           |
| `DATASET_MISMATCH`            | DatasetNormalizer         | MEDIUM           |
| `METRIC_MISMATCH`             | MetricNameComparator      | LOW              |
| `CONFIGURATION_MISMATCH`      | ModelNameNormalizer       | LOW (alias hit) / MEDIUM (no hit) |
| `AMBIGUOUS`                   | HyperparameterChecker (LLM disagrees with regex) | LOW |

`Severity ∈ {LOW, MEDIUM, HIGH}` per `spec-summary.md:69-69`.

**Why no 9th `MODEL_NAME_MISMATCH`:** spec doc lists exactly 8
types. Adding a 9th would diverge from the spec.
`CONFIGURATION_MISMATCH` is the closest semantic fit for "paper
says model X, code uses model Y" — LOW when the alias normalizer
agrees they're aliases, MEDIUM when no alias is found.

### 2.8 Alias JSON config files (not Python dicts)

`verification/aliases/{models,datasets,metrics}.json` hold the
canonical → alias mappings:

```json
{
  "qwen2.5-7b":          ["Qwen/Qwen2.5-7B", "qwen2.5-7b", "Qwen2.5-7B"],
  "qwen2.5-7b-instruct": ["Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"],
  "bert-base-uncased":   ["google-bert/bert-base-uncased", "bert-base-uncased"],
  "gpt2":                ["openai-community/gpt2", "gpt2", "GPT-2"]
}
```

Why JSON, not a Python dict:

  - **Data, not code** — adding "Qwen2.5-7B-Instruct" or "DeepSeek-R1"
    doesn't require a code PR.
  - **Lintable** — JSON Schema validation in tests.
  - **Swappable per environment** — corporate users can override
    the file without touching the code.

Loaded at module import time via `json.loads(Path(__file__).parent
/ "aliases" / "<thing>.json").read_text())`.

Fuzzy match (trigram ratio via `difflib.SequenceMatcher`) is the
fallback when a token isn't in any alias list — same algorithm
`pg_trgm` would use, just in Python.

### 2.9 Hyperparameter numeric tolerance

Fixed taxonomy of 7 names:

```
learning_rate | lr | η
batch_size
temperature | τ
beta | β
gamma | γ
num_epochs | epochs
hidden_size | d_model | d
```

Two tolerance regimes:

  - **Additive** (`batch_size`, `num_epochs`, `hidden_size`):
    relative diff > 5% → mismatch. `32` vs `33` → match
    (3.1% diff); `32` vs `40` → mismatch.
  - **Multiplicative** (`learning_rate`, `temperature`, `beta`,
    `gamma`): ratio > 10× → mismatch. `5e-5` vs `1e-4` → match
    (2× ratio); `5e-5` vs `1e-3` → mismatch.

Why two regimes: learning rates vary by orders of magnitude across
methods (`1e-3` Adam vs `1e-5` fine-tuning); the *relative* log
ratio matters more than the absolute diff.

Extraction is regex over `paper_evidence.quote` (with fallback
to `claim.text`):

```
r"(?P<name>learning_rate|lr|...)\s*[=:]\s*(?P<value>[0-9.eE+-]+)"
```

Same regex applied to each candidate's `signature` and (if
present) `docstring`.

### 2.10 `LLMClient.verdict_for_claim(...)`

Mirrors `extract_claims()` exactly:

```python
class LLMClient:
    def verdict_for_claim(
        self, *,
        claim: Claim,
        paper_evidence: PaperEvidence,
        candidates: list[RetrieveMatch],
        deterministic_discrepancies: list[Discrepancy],
    ) -> Verification:
        # Uses self._client + self._settings.openai_model_chat
        # response_format = {"type": "json_schema", "strict": True,
        #                    "schema": Verification.model_json_schema()}
        # Maps errors to VerificationLLMError (NOT ClaimExtractionError)
```

The verifier's request to OpenAI includes the
`deterministic_discrepancies` as part of the user prompt — the
LLM is told "we already found these; you don't need to repeat
them, but feel free to add more". This avoids double-counting
and lets the LLM focus on residual judgement.

### 2.11 Endpoint contract

```python
class VerificationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)

class VerificationRunResponse(BaseModel):
    analysis_id: UUID
    n_verified: int
    n_partially_verified: int
    n_contradicted: int
    n_unverified: int
    verifications: list[Verification]

class VerificationListResponse(BaseModel):
    analysis_id: UUID
    verifications: list[Verification]
```

Routes:

  - `POST /analyses/{analysis_id}/verifications` — runs
    pipeline, persists to `AnalysisState.verifications`,
    returns summary.
  - `GET /analyses/{analysis_id}/verifications` — reads the
    persisted JSONB.

Status mapping:

  - 404 if analysis doesn't exist (mirrors
    `papers/router.py:165-174`).
  - 500 if LLM fails (mirrors `EmbeddingError` mapping in
    `repos/router.py:514-535`).
  - 422 if `claim_ids` references non-existent claim IDs.

## 3. CI considerations

- **No new migration.** Phase 1 already provisioned
  `AnalysisState.verifications: JSONB`.
- **Backend-test job** picks up new unit tests via
  `pytest apps/api/tests` — no YAML edit.
- **Integration tests** opt-in via `-m integration` (already
  configured in `apps/api/pyproject.toml:71-80`).
- **No new Python deps.** `pg_trgm` already ships with the
  base `pgvector/pgvector:pg16` image. We use stdlib `difflib`
  for fuzzy matching.
- **`evaluation/evaluate_verification.py`** is **NOT wired in
  Phase 6** — that's a Phase 10 eval concern. The
  `Makefile` target `eval-verification` is pre-declared but
  not yet functional.
- **No Dockerfile / dev-up.sh / workflow YAML edits.**
- **`apps/api/tests/conftest.py`** stays minimal. Integration
  fixtures (`db_session`, `seeded_session`) are module-local,
  matching the convention from Phase 3/5 tests.

## 4. Acceptance — what the doc specifies

**Hard acceptance** (from `phase-roadmap.md:128`, testable):

> Every verification has paper evidence AND code evidence, or
> `UNVERIFIED`.

Operationalised as a single test assertion in
`apps/api/tests/test_verification_workflow.py`:

```python
def test_every_verification_has_evidence_or_unverified(
    workflow_result, monkeypatch_llm,
):
    for v in workflow_result.verifications:
        assert v.paper_evidence is not None
        assert v.code_evidence or v.status == VerificationStatus.UNVERIFIED
```

**Soft acceptance** (testable but deferrable to Phase 10):

  - All 4 statuses appear in a typical 5-claim run (mix of
    paper-evidence-only and code-evidence-backed claims).
  - `HYPERPARAMETER_MISMATCH` is correctly emitted for
    `lr=5e-5` paper vs `lr=1e-3` code.
  - `ModelNameNormalizer` canonicalises
    `Qwen/Qwen2.5-7B` ↔ `Qwen2.5-7B`.

**Numerical acceptance** (Phase 10 eval, not Phase 6):

  - Verification accuracy on the 3 hand-labelled pairs.
  - Grounding rate (fraction of verifications whose paper
    evidence is a real verbatim quote).

## Files the Phase 6 implementation will touch (post-doc-approval)

### New (backend)
- `apps/api/claim2code_api/verification/__init__.py`
- `apps/api/claim2code_api/verification/exceptions.py`
- `apps/api/claim2code_api/verification/schemas.py`
- `apps/api/claim2code_api/verification/normalizers.py`
- `apps/api/claim2code_api/verification/hyperparameter_checker.py`
- `apps/api/claim2code_api/verification/discrepancies.py`
- `apps/api/claim2code_api/verification/verifier.py`
- `apps/api/claim2code_api/verification/workflow.py`
- `apps/api/claim2code_api/verification/router.py`
- `apps/api/claim2code_api/verification/aliases/models.json`
- `apps/api/claim2code_api/verification/aliases/datasets.json`
- `apps/api/claim2code_api/verification/aliases/metrics.json`

### Edited (backend)
- `apps/api/claim2code_api/llm/client.py` — add
  `verdict_for_claim(...)` + `_VERIFY_SYSTEM_PROMPT`
- `apps/api/claim2code_api/app/main.py` — mount
  `verification_router`

### Tests (new)
- `apps/api/tests/test_verification_schemas.py`
- `apps/api/tests/test_hyperparameter_checker.py`
- `apps/api/tests/test_normalizers.py`
- `apps/api/tests/test_discrepancies.py`
- `apps/api/tests/test_verifier.py`
- `apps/api/tests/test_verification_workflow.py`
- `apps/api/tests/integration/test_verification_e2e.py`

### Docs
- `docs/architecture/phase-6-design.md` — **the deliverable of
  this doc** (already in place)
- `README.md` — phase roadmap table updated
- `evaluation/README.md` — Phase 6 status line updated

### No change
- `apps/api/claim2code_api/db/models.py` — JSONB column exists
- `apps/api/claim2code_api/migrations/versions/` — no new
  migration
- `apps/api/claim2code_api/papers/schemas.py` — `Claim` not
  extended (verifier LLM extracts hyperparams)
- `apps/api/claim2code_api/papers/workflow.py` — no
  auto-verify hook (POST is explicit)
- `apps/api/claim2code_api/repos/parser.py` — no change (no
  `source/imports/calls`)
- `.github/workflows/ci.yml` — picks up new tests via
  `pytest`; no YAML edit
- `Makefile` — no new target
- `deployments/Dockerfile.api` — no edit
- `scripts/dev-up.sh` — no edit

## Commit plan (after doc is approved)

One branch `phase/6-verification`, 8 logical commits, `--no-ff`
merge to main:

1. `feat(verify): Verification + Discrepancy + CodeEvidence Pydantic schemas`
2. `feat(verify): alias JSON files (models, datasets, metrics) + normalizers`
3. `feat(verify): HyperparameterChecker (regex extract + numeric tolerance)`
4. `feat(verify): DiscrepancyClassifier (8 types × 3 severities, merge rules)`
5. `feat(verify): LLMClient.verdict_for_claim (Structured Outputs + no-model-knowledge rule)`
6. `feat(verify): verification workflow + POST/GET /analyses/{id}/verifications`
7. `test(verify): unit + integration tests (deterministic checkers, LLM mock, e2e round-trip)`
8. `docs(verify): phase-6 design + README status update`

## Verification (of the design doc itself)

- ✓ Doc lives at `docs/architecture/phase-6-design.md`
- ✓ Sections §1-§4 present, in the order above
- ✓ Tone + structure mirror `phase-5-design.md`
- ✓ Every "we will X" has a corresponding file in the
  post-doc-approval list (no orphan decisions)
- ✓ Every "we will NOT do X" cites the phase that owns it
- ✓ Three live decisions each have a rationale + trade-off
  note (§2.1 JSONB stay, §2.2 POST + GET, §2.3 verifier
  LLM extracts hyperparams)
- ✓ Recon findings (no `Claim.hyperparameters`, no
  `CodeSymbol.source`, no existing normalizers) are reflected
  in the design (§2.3, §2.4, §2.8)
- ✓ "No model knowledge" rule from `spec-summary.md:148-150`
  is operationalised in the system prompt (§2.4) AND the
  structural acceptance rule (§2.5)
- ✓ Acceptance criterion (`paper_evidence AND code_evidence
  OR UNVERIFIED`) is testable as a single assertion on the
  response (§4)

After the doc is merged to main (separate doc-only commit on
`docs/phase-6-design`), the implementation branch opens and
work proceeds against the spec.

## Why these decisions specifically

- **JSONB stay vs dedicated table** — Phase 1 explicitly
  designed the JSONB columns to evolve until Phase 6
  (`db/models.py:75-77`). The shape IS stabilising now, but
  the cost of normalisation (migration + ORM + relationship
  wiring + test fixtures) doesn't buy anything Phase 6 needs.
  Phase 10 eval is the natural place to revisit if we want
  per-status aggregation.

- **POST + GET vs GET-only** — the roadmap names only GET, but
  without a POST the verifications never run. Auto-running
  inside `papers/workflow.py` (Phase 2) couples paper parsing
  to LLM availability and explodes `papers/workflow.py` test
  mocking. POST is one extra endpoint, matches the existing
  convention (`POST /papers` → `GET /papers/{id}`), and lets
  the frontend trigger re-verification after edits.

- **Verifier LLM extracts hyperparams vs Phase 2 model change**
  — the Phase 2 LLM is not asked for structured hyperparameters
  today. Extending Phase 2 requires a new Pydantic field, a
  Phase 2 system-prompt change, and re-testing 3 already-shipped
  extraction paths. Pulling it into Phase 6's verifier response
  costs one extra field in the verifier JSON schema and keeps
  Phase 2 untouched. If a future phase needs `Claim.hyperparameters`
  for filtering, it can backfill from the persisted verification
  list.

- **Aliases in JSON config files vs Python dict** — JSON files
  are data, not code. Easy to extend (no PR for adding
  "Qwen2.5-7B-Instruct" or "DeepSeek-R1"). Easy to lint, easy
  to test (load + assert structure). Python dicts would require
  code changes for every alias.

- **Deterministic first, LLM residual** — matches the Phase 5
  retrieval pattern (3 legs → orchestrator). Deterministic
  output is reproducible; LLM usage stays bounded.

- **No 9th discrepancy type (`MODEL_NAME_MISMATCH`)** — spec
  doc lists exactly 8 types. Adding a 9th would diverge from
  the spec. `CONFIGURATION_MISMATCH` is the closest semantic
  fit (LOW severity when the alias normalizer agrees they're
  aliases, MEDIUM when they don't).

- **Two tolerance regimes (additive / multiplicative)** —
  learning rates vary by orders of magnitude across methods
  (`1e-3` Adam vs `1e-5` fine-tuning); the *log* ratio matters
  more than the absolute diff. Batch sizes / epochs / hidden
  dims are absolute counts and a 5% relative diff is the right
  boundary.

- **No frontend in Phase 6** — the roadmap does not list a
  Phase 6 web task. Phases 7-8 may bring a `VerificationList`
  panel; for now the API + JSONB persistence is the deliverable.
  The frontend can `GET /analyses/{id}/verifications` and
  render however it wants.