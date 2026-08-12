# Phase 7 — Related Work Design

> Goal (from `phase-roadmap.md:132-143`): selected claim → up to 5
> related papers with structured comparison.
>
> Acceptance (verbatim): Returns up to 5 papers, each with
> Problem / Method / Dataset / Metric / Difference / Agreement fields.

## 1. What ships

| Layer | Change |
|---|---|
| `apps/api/claim2code_api/literature/__init__.py` | Edit: module docstring + `__all__` |
| `apps/api/claim2code_api/literature/exceptions.py` | New: `LiteratureError` + `LiteratureSearchError` + `LiteratureConfigError` + `LiteratureNotFoundError` |
| `apps/api/claim2code_api/literature/schemas.py` | New: `ScholarField` (StrEnum); `ScholarPaper` (S2/OpenAlex paper shape); `Comparison` (6-field structured extract); `RankedPaper` (ScholarPaper + score + breakdown + comparison); `RelatedWorkEntry` (per-claim wrapper); `RelatedWorkRunRequest`/`Response`/`ListResponse` |
| `apps/api/claim2code_api/literature/scholar_client.py` | New: `ScholarClient` Protocol + `SemanticScholarClient` (primary) + `OpenAlexClient` (fallback). Both httpx-based with `proxy=None, trust_env=False` workaround + 429 retry. |
| `apps/api/claim2code_api/literature/ranker.py` | New: `RankingWeights` dataclass + `rank_papers()` — linear combination `score = w_sem·sim + w_rec·recency + w_cite·citation_norm + w_method·method_overlap` |
| `apps/api/claim2code_api/literature/comparator.py` | New: `extract_comparison()` thin wrapper around `LLMClient.extract_comparison(...)` |
| `apps/api/claim2code_api/literature/workflow.py` | New: `run_related_work_workflow(*, db, analysis_id, claim_id, top_k=5, force_refresh=False)` + `RelatedWorkWorkflowResult` dataclass |
| `apps/api/claim2code_api/literature/router.py` | New: `POST /analyses/{analysis_id}/related-work` (run) + `GET /analyses/{analysis_id}/related-work` (read). `LiteratureError` → HTTPException mapping. |
| `apps/api/claim2code_api/llm/client.py` | Edit: add `extract_comparison(*, claim, paper)` method + `_COMPARE_SYSTEM_PROMPT` constant (same call shape as `verdict_for_claim`) |
| `apps/api/claim2code_api/db/models.py` | Edit: `AnalysisState` adds `related_work: JSONB = []` column |
| `apps/api/migrations/versions/0006_analysis_state_related_work.py` | New: adds `related_work: JSONB` column with default `[]` |
| `apps/api/claim2code_api/app/config.py` | Edit: `Settings` adds `semantic_scholar_api_key: str \| None = None` |
| `apps/api/claim2code_api/app/main.py` | Edit: `app.include_router(literature_router)` |
| `.env.example` | Edit: add `SEMANTIC_SCHOLAR_API_KEY=` |
| `apps/api/tests/test_literature_schemas.py` | New: ~14 tests — Pydantic validation, `extra="forbid"`, StrEnum membership |
| `apps/api/tests/test_scholar_client.py` | New: ~12 tests — mocked httpx; S2 happy path, OpenAlex fallback, 429 retry, missing API key, abstract reconstruction |
| `apps/api/tests/test_ranker.py` | New: ~10 tests — ranking monotonicity, weight sensitivity, edge cases (zero citations, missing year) |
| `apps/api/tests/test_comparator.py` | New: ~10 tests — LLM call shape, JSON schema, error mapping, per-paper isolation (mock LLM) |
| `apps/api/tests/test_literature_workflow.py` | New: ~10 tests — pipeline ordering, per-paper isolation, JSONB write-back, **acceptance test** |
| `apps/api/tests/integration/test_literature_e2e.py` | New: ~4 tests — live Postgres + mock LLM + mocked S2 + 1-claim round-trip |
| `README.md` | Edit: phase roadmap table — Phase 7 row → ✅ |
| `evaluation/README.md` | Edit: Phase 7 status line |

**No new Python deps.** `httpx` is already in `pyproject.toml`. We use stdlib + httpx for everything.

**One new env var.** `SEMANTIC_SCHOLAR_API_KEY` (optional; raises rate limit from 1 req/s unauthenticated to 100 req/5 min with key).

**One new migration.** `0006_analysis_state_related_work.py` adds `related_work: JSONB` column. (`verifications` was pre-staged in Phase 1; `related_work` was not — unavoidable new migration.)

**No frontend task.** The roadmap does not list a Phase 7 web task. Phase 7 ships the API + JSONB persistence only.

## 2. Key decisions

### 2.1 Semantic Scholar primary + OpenAlex fallback (no arXiv)

Three candidates evaluated: **Semantic Scholar**, **OpenAlex**, **arXiv**.

| Criterion | Semantic Scholar | OpenAlex | arXiv |
|---|---|---|---|
| Free / API key | Yes, optional key | Yes, anonymous OK | Yes, no key |
| Rate limit | 1 req/s unauth → 100/5min with key | ~10 req/s polite pool | 1 req / 3 sec |
| Metadata richness | ★★★★★ — TLDR, citationCount, fieldsOfStudy, SPECTER embeddings | ★★★★ — authorships, primary_location, concepts (no TLDR) | ★★ — title, abstract, authors, year (no TLDR, citation, venue) |
| Relevance search | Built-in (SPECTER) | Built-in (lexical + full-text) | Lexical only |
| Python SDK | `semanticscholar` PyPI + httpx | `pyalex` + httpx | `arxiv` PyPI + httpx |

**Pick S2 primary.** SPECTER embeddings give the best relevance out of the box; the TLDR field is a pre-trained model summary that the LLM comparison step can cite directly; citationCount + year + venue + externalIds are all directly available. arXiv is dropped because the 3-second throttle makes a 5-paper batch take 15+ seconds, and the metadata gaps (no TLDR, no citation count) mean we'd embed every abstract ourselves anyway.

**OpenAlex as fallback.** When S2 is rate-limited (429) or returns 5xx, fall back to OpenAlex for breadth. OpenAlex stores abstracts as inverted indexes that need reconstruction (a known pattern), and uses cursor pagination. We over-fetch from S2 (limit=20), so falling back to OpenAlex with the same over-fetch is symmetric.

### 2.2 POST + GET endpoints (mirrors Phase 6)

The roadmap names `GET /analyses/{id}/related-work?claim_id=…`. Without a trigger the related-work never runs. Three candidates:

- **(A)** Auto-run inside `verification/workflow.py` after claim verification.
- **(B)** POST + GET (mirrors Phase 6 `POST /analyses/{id}/verifications`).
- **(C)** Sync inside the GET (run-on-demand, cache).

Pick **(B)**. Reasoning (same as Phase 6 §2.2):

  - **(A)** couples verification to a secondary feature and inflates `verification/workflow.py` (already 270+ lines).
  - **(C)** makes GET non-idempotent — the same URL gives different responses at different times depending on cache TTL.
  - **(B)** matches `POST /verifications` → `GET /verifications`. The roadmap's GET becomes the read endpoint; POST adds the trigger with optional `force_refresh=true` to bypass cache.

```python
class RelatedWorkRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    force_refresh: bool = False
```

### 2.3 One LLM call per paper (per-paper isolation, mirrors Phase 6)

Phase 6's `VerificationWorkflowResult` records per-claim errors and never aborts the run. Phase 7 mirrors: one LLM failure on paper #3 doesn't abort papers #1, #2, #4, #5. The error is recorded on `state.errors`, that paper's `Comparison` is `None`, the rest complete.

Three candidates:

  - **(a)** One batched LLM call for all 5 papers — cheaper but a schema error fails all 5.
  - **(b)** Per-paper LLM calls — ~5× cost, per-paper isolation, matches Phase 6.
  - **(c)** Per-claim, batched across papers — requires a different shape; rejected for surface size.

Pick **(b)**. Cost is bounded: paper abstract is ~2 KB, response is ~1 KB, so per-call cost is small. The isolation is the win: in 5 calls, one malformed response fails one paper, not five.

### 2.4 `comparison` allowed to be `None` (structural guard)

Mirrors Phase 6's "no `code_evidence` → UNVERIFIED" rule. If the LLM call fails for a paper, the paper's `comparison` field is `None`; the paper still ships with metadata + score + breakdown. The frontend can render "comparison unavailable" without hiding the whole paper.

The structural acceptance criterion is:

> Every `RankedPaper` has `title/year/abstract` non-empty AND `score ∈ [0, 1]` AND `comparison IS None OR all 6 fields populated`.

This is a single assertion on the response, testable as a unit test (the acceptance test in `test_literature_workflow.py`).

### 2.5 JSONB stay — `related_work` column over a table

Same Phase 6 §2.1 reasoning: shape may still evolve in Phase 10 eval; no FK constraint needed (`claim_id` references the same row's `claims`); Phase 10 can revisit if per-source aggregation (S2 vs OpenAlex) becomes a query.

The trade-off: no FK constraint means a `RelatedWorkEntry` with a `claim_id` not in `state.claims` would persist silently. Mitigation: validate `claim_id` at workflow entry (raises `LiteratureNotFoundError` → 422) before persisting. The validation is the structural guard against dangling references.

### 2.6 Four-leg ranker mirroring Phase 5's HybridWeights

Phase 5's `hybrid_ranker` fuses three legs (semantic / lexical / structural) with RRF. Phase 7's `rank_papers` fuses four legs (semantic / recency / citation / method-overlap) with a linear combination. Different fusion (linear vs RRF) because Phase 7's legs are continuous scores that compose naturally; Phase 5's legs are ranks where RRF's rank-based fusion matters.

Default weights: `w_sem=0.55, w_rec=0.15, w_cite=0.20, w_method=0.10`. The semantic leg dominates (it's the only leg that knows the user's actual query); the other three are tie-breakers. The `method_overlap` leg is `0.0` in v1 — populating it would need a method taxonomy + extractor (deferred to Phase 7.5). The `score_breakdown` exposes each leg's contribution for the UI tooltip ("why this paper?").

## 3. Pipeline (per claim)

```
1. Load AnalysisState.claims → resolve claim.text + claim.paper_evidence.quote

2. If not force_refresh AND state.related_work already has this claim_id →
   return the cached entry (idempotent POST semantics).

3. SEARCH:
   candidates = SemanticScholarClient.search(
       query=claim.text, limit=20,
       fields=[title, abstract, year, authors, citationCount, venue,
               publicationVenue, externalIds, openAccessPdf, tldr,
               fieldsOfStudy],
       api_key=settings.semantic_scholar_api_key,
   )
   On LiteratureSearchError → fall back to OpenAlexClient.search.
   On both failing → return empty entry, record error on state.errors.

4. RANK:
   query_embedding = EmbeddingClient.embed([claim.text])[0]   # 1536-dim
   paper_embeddings = EmbeddingClient.embed([p.abstract for p in candidates])
   sims = cosine_similarity(query_embedding, paper_embeddings)
   ranked = rank_papers(candidates, sims=sims, weights=RankingWeights())[:5]

5. COMPARE (one LLM call per paper, per-paper isolation):
   for paper in ranked:
     try:
       comparison = llm.extract_comparison(claim=claim, paper=paper)
       paper.comparison = comparison
     except LiteratureComparisonError as e:
       state.errors.append({...})
       paper.comparison = None    # structural guard

6. PERSIST:
   entry = RelatedWorkEntry(
       claim_id=claim_id, papers=ranked,
       generated_at=now(),
       source="semantic_scholar",   # or "openalex" if S2 failed
   )
   state.related_work = replace_where_match(state.related_work, entry)
   db.flush()

7. Return RelatedWorkWorkflowResult(entry, n_papers=len(ranked))
```

`run_related_work_workflow` is the entry point; the router calls it inside `POST`.

## 4. `LLMClient.extract_comparison(...)`

Mirrors `verdict_for_claim(...)` exactly — same `chat.completions.create` + `response_format={"type": "json_schema", "strict": True, "schema": Comparison.model_json_schema()}` shape. New method, new system prompt.

```python
class LLMClient:
    def extract_comparison(
        self, *, claim: Claim, paper: ScholarPaper,
    ) -> Comparison:
        # Reuses self._client + self._settings.openai_model_chat
        # Maps errors to LiteratureComparisonError (NOT
        # VerificationLLMError — siblings in different stages).
```

**System prompt** (`_COMPARE_SYSTEM_PROMPT` constant):

```
You are a meticulous research-paper comparison analyst. Given
(1) a technical claim extracted from a paper, and (2) a related
paper's metadata (title, authors, year, venue, abstract, optional
TLDR), produce a structured 6-field comparison.

RULES (non-negotiable):

  1. **No model knowledge.** Every comparison field must cite the
     verbatim abstract or TLDR provided. You MAY NOT use knowledge
     of the paper, the related work, or the field beyond what is
     explicitly given in the inputs.

  2. **Six fields, all required:**
       problem:  what problem does the related paper address?
       method:   what method does it use?
       dataset:  what dataset does it evaluate on? (or null)
       metric:   what metric does it report? (or null)
       difference: how does it DIFFER from the claim's paper?
       agreement: where does it AGREE with the claim's paper?

  3. **Cite the abstract.** Each field's text should be a 1–3
     sentence summary that an evaluator can trace back to the
     verbatim input. If the abstract doesn't address a field, set
     it to null (dataset/metric) or to "(abstract does not
     address)" (problem/method/difference/agreement).

  4. **Be specific.** "Uses a transformer" is unhelpful;
     "Uses BART-base fine-tuned on CNN/DailyMail" is the goal.

Output schema is enforced via Structured Outputs (response_format).
Any deviation fails the call.
```

The `Comparison` schema carries the same `extra="forbid"` + StrEnum conventions as `Verification`. The "no model knowledge" rule from `spec-summary.md:148-150` is operationalised verbatim.

## 5. Endpoint contract

```python
class RelatedWorkEntry(BaseModel):
    claim_id: str
    papers: list[RankedPaper]
    generated_at: datetime
    source: Literal["semantic_scholar", "openalex"]

class RelatedWorkRunResponse(BaseModel):
    analysis_id: UUID
    entry: RelatedWorkEntry
    n_papers: int
    n_with_comparison: int                            # how many had LLM success
    errors: list[str] = []

class RelatedWorkListResponse(BaseModel):
    analysis_id: UUID
    entries: list[RelatedWorkEntry]
```

- `POST /analyses/{analysis_id}/related-work` — body `RelatedWorkRunRequest`; runs pipeline; persists; returns `RelatedWorkRunResponse`. `claim_id` not found in `state.claims` → 422. `LiteratureSearchError` after both S2 and OpenAlex fail → 500. `LiteratureConfigError` (e.g. malformed settings) → 500.
- `GET /analyses/{analysis_id}/related-work?claim_id=…` (query param optional; omit → return all entries) — reads `state.related_work: JSONB`, validates each entry through `RelatedWorkEntry.model_validate`, skips malformed entries with a log warning (mirrors Phase 6 GET behavior).

## 6. Tests

| File | Test count target | Coverage |
|---|---|---|
| `test_literature_schemas.py` | 14 | Pydantic validation, `extra="forbid"`, StrEnum membership, score_breakdown shape |
| `test_scholar_client.py` | 12 | S2 happy path with mocked httpx; OpenAlex fallback; 429 retry; missing API key; inverted-index reconstruction |
| `test_ranker.py` | 10 | ranking monotonicity, weight sensitivity, edge cases (zero citations, missing year, single paper) |
| `test_comparator.py` | 10 | LLM call shape + JSON schema + error mapping + per-paper isolation (mock LLM) |
| `test_literature_workflow.py` | 10 | pipeline ordering, JSONB write-back, cache-on-repeat-POST, force_refresh bypass, **acceptance test** |
| `integration/test_literature_e2e.py` | 4 | live Postgres + mock LLM + mocked S2 + 1-claim round-trip |

**Acceptance test** (in `test_literature_workflow.py`):
Given a fixture analysis with 1 claim and a stub S2 client that returns 5 papers, assert every returned `RankedPaper` satisfies:
- `paper.title`, `paper.year`, `paper.abstract` are non-empty (ScholarClient contract)
- `paper.score` ∈ [0, 1] (ranker normalized)
- `paper.comparison` is `None` OR has all 6 fields populated (LLM contract)

## 7. CI / infra

- `backend-test` job picks up new tests via `pytest apps/api/tests` — no YAML change.
- `backend-lint` job picks up new source files via `ruff check .` + `mypy .` — no YAML change. New `httpx.Client` calls mirror Phase 4/6; the `# type: ignore[arg-type]` pattern (from Phase 6's CI green) covers the httpx/httpx2 mismatch.
- No new Python deps. `httpx` is already in `pyproject.toml`.
- `.env.example` updated to document `SEMANTIC_SCHOLAR_API_KEY=`. CI doesn't set it (tests use mocks); dev runs without it get the unauthenticated 1 req/s rate.

## 8. What we explicitly do NOT do in Phase 7

- ❌ Frontend panel for related work (roadmap does not list a Phase 7 web task)
- ❌ Persisting papers we cite to a separate `papers` table for cross-analysis reuse (Phase 10 eval owns this)
- ❌ arXiv as third fallback (deferred to Phase 7.5 if preprints matter — keep Phase 7 surface small)
- ❌ Cross-paper claim matching (claims from multiple papers → "is paper X's claim c3 the same as paper Y's claim c7?") — different feature
- ❌ Async related-work (sync, like Phase 5/6; one paper at a time sequentially)
- ❌ Method-overlap deterministic signal (`method_overlap=0.0` in v1; future phase can compute from abstract)
- ❌ Force-refresh TTL semantics (explicit `force_refresh=true` flag in POST body is enough for v1)
- ❌ Per-paper confidence calibration (Phase 10 owns calibration)
- ❌ Re-ranking from user feedback (Phase 9 Observability owns feedback loop)
- ❌ Normalizing `related_work` into a `related_papers` table (defer to Phase 10 if eval needs per-source aggregation)

## 9. Files Phase 7 will touch (post-doc-approval)

### New (backend)
- `apps/api/claim2code_api/literature/exceptions.py`
- `apps/api/claim2code_api/literature/schemas.py`
- `apps/api/claim2code_api/literature/scholar_client.py`
- `apps/api/claim2code_api/literature/ranker.py`
- `apps/api/claim2code_api/literature/comparator.py`
- `apps/api/claim2code_api/literature/workflow.py`
- `apps/api/claim2code_api/literature/router.py`
- `apps/api/migrations/versions/0006_analysis_state_related_work.py`

### Edited (backend)
- `apps/api/claim2code_api/literature/__init__.py`
- `apps/api/claim2code_api/llm/client.py`
- `apps/api/claim2code_api/db/models.py`
- `apps/api/claim2code_api/app/config.py`
- `apps/api/claim2code_api/app/main.py`
- `.env.example`

### Tests (new)
- 6 new test files under `apps/api/tests/` + `apps/api/tests/integration/`

### Docs
- `README.md` — phase roadmap table: Phase 7 row → ✅
- `evaluation/README.md` — Phase 7 status line

### No change
- `apps/api/claim2code_api/papers/schemas.py` — `Claim` not extended
- `apps/api/claim2code_api/verification/*` — no cross-package import
- `.github/workflows/ci.yml` — picks up new tests + sources; no YAML edit
- `Makefile` — no new target (eval-literature lives in Phase 10)

## 10. Commit plan

One branch `phase/7-related-work`, 7 logical commits, `--no-ff` merge to main:

1. `feat(literature): RelatedWork + Comparison + ScholarPaper + RankedPaper Pydantic schemas`
2. `feat(literature): SemanticScholarClient + OpenAlexClient (httpx + proxy workaround + 429 retry)`
3. `feat(literature): RankingWeights + rank_papers (semantic × recency × citation × method_overlap)`
4. `feat(literature): LLMClient.extract_comparison (Structured Outputs + no-model-knowledge rule)`
5. `feat(literature): related-work workflow + POST/GET /analyses/{id}/related-work + migration + settings`
6. `test(literature): unit + integration tests (scholar mock, ranker, comparator, workflow, e2e)`
7. `docs(literature): README + evaluation status`

(7 commits. This doc itself ships first on a separate `docs/phase-7-design` branch, merged `--no-ff` before the implementation branch opens.)

## 11. Verification (of the design doc itself)

- ✓ Doc lives at `docs/architecture/phase-7-design.md`
- ✓ Sections §1-§10 present, in the order above
- ✓ Tone + structure mirror `phase-6-design.md` (POST + GET + per-paper isolation match the Phase 6 pattern)
- ✓ Every "we will X" has a corresponding file in the post-doc-approval list (no orphan decisions)
- ✓ Every "we will NOT do X" cites the phase that owns it
- ✓ Acceptance criterion (every paper has metadata + score + comparison or comparison is None) is testable as a single assertion on the response

After this doc is merged to main (separate doc-only commit on `docs/phase-7-design`), the implementation branch opens and work proceeds against the spec above.