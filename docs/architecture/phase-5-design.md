# Phase 5 — Hybrid Retrieval Design

> Goal (from `phase-roadmap.md:99-111`): semantic + lexical +
> structural ensemble. Run the first retrieval eval.
>
> Acceptance: First numbers on 3 hand-annotated paper/repo pairs.
> Hybrid > embedding-only on Recall@5.

## 1. What ships

| Layer | Change |
|---|---|
| `apps/api/claim2code_api/indexing/lexical.py` | New: `lexical_search(db, repository_id, claim, top_k)` returns `list[(symbol_id, score)]` via Postgres `tsvector` + `ts_rank_cd` |
| `apps/api/claim2code_api/indexing/structural.py` | New: `structural_boost(db, candidate_ids, claim) -> dict[symbol_id, float]` returning per-symbol multipliers (default 1.0) |
| `apps/api/claim2code_api/retrieval/` | Existing stub (`__init__.py` only) is fleshed out: add `hybrid_ranker.py` orchestrating the three legs + RRF fusion |
| `apps/api/migrations/versions/0005_code_symbols_lexical.py` | New: adds `lex_text` (generated `TEXT`) + `lex_tsv` (generated `tsvector` with field weights) on `code_symbols`; GIN index on `lex_tsv`; trigram index on `name`; enables `pg_trgm` if missing |
| `apps/api/claim2code_api/db/models.py` | Edit `CodeSymbol`: append `lex_text` + `lex_tsv` mapped columns (no-op on SQLite, populated on Postgres) |
| `apps/api/claim2code_api/repos/schemas.py` | Extend `RetrieveRequest` with `mode: Literal["semantic","lexical","hybrid"] = "hybrid"` + optional `weights: RetrieveWeights`; add `ScoreBreakdown` + `RetrieveWeights`; extend `RetrieveResponse` with `breakdown: list[ScoreBreakdown]`, `mode`, `weights` |
| `apps/api/claim2code_api/repos/router.py` | Edit `retrieve_repository`: dispatch on `mode` (`semantic` → `retrieve_semantic`, `lexical` → `lexical_search`, `hybrid` → `hybrid_ranker`); assemble the `breakdown` for the hybrid path |
| `evaluation/baselines/embedding_only.py` | New: thin wrapper around `retrieve_semantic` |
| `evaluation/baselines/lexical_only.py` | New: thin wrapper around `lexical_search` |
| `evaluation/baselines/hybrid_claim2code.py` | New: thin wrapper around `hybrid_ranker` |
| `evaluation/evaluate_retrieval.py` | New: runs the 3 baselines × 3 hand-labelled datasets; computes Recall@1, Recall@5, MRR; renders `evaluation/results.json` + `evaluation/report.md` |
| `evaluation/annotations/{attention_is_all_you_need,bert,lora}.json` | New: 30 hand-labelled `(claim → symbol)` mappings (10 per pair) |
| `evaluation/datasets/{attention_is_all_you_need,bert,lora}/` | New: 3 paper PDF + cloned-repo pairs (repos `gitignored`) |
| `evaluation/README.md` | Edit: status line + the now-real `evaluate_retrieval.py` reference |
| `Makefile` | Edit: `eval-retrieval` target verified to point at the now-existing script (today it points at a non-existent file) |
| `README.md` | Edit: status table — Phase 5 done, Phase 6 next |
| `apps/api/claim2code_api/indexing/__init__.py` | Edit: module docstring updated to mention `lexical` + `structural` |
| `apps/api/claim2code_api/retrieval/__init__.py` | Edit: module docstring + `__all__` |

**No new Python deps.** Postgres `tsvector` + `ts_rank_cd` are
built into Postgres 16; `pg_trgm` ships with the base
`pgvector/pgvector:pg16` image.

**No env-var additions.** Phase 5 only reuses the existing
Postgres + OpenAI settings.

## 2. Key decisions

### 2.1 Lexical scorer — Postgres `tsvector` + `ts_rank_cd`

The lexical leg scores each symbol by **term overlap** with the
claim, weighted by Postgres `english` text-search config (which
handles lowercasing + Porter stemming + stopwords). Why:

- **Zero new Python deps.** Built into Postgres 16.
- **Scales with the table** via the GIN index on `lex_tsv`.
- **Matches the storage story** — Postgres already owns
  everything else.
- **Field-weighted** — symbol `name` matters more than
  `file_path`; the weights are documented as a tunable.

The schema:

```sql
ALTER TABLE code_symbols
  ADD COLUMN lex_text TEXT GENERATED ALWAYS AS (
    coalesce(name, '') || ' ' ||
    coalesce(signature, '') || ' ' ||
    coalesce(docstring, '') || ' ' ||
    coalesce(file_path, '')
  ) STORED,
  ADD COLUMN lex_tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(name, '')),       'A') ||
    setweight(to_tsvector('english', coalesce(signature, '')),  'B') ||
    setweight(to_tsvector('english', coalesce(docstring, '')),  'C') ||
    setweight(to_tsvector('english', coalesce(file_path, '')),  'D')
  ) STORED;

CREATE INDEX ix_code_symbols_lex_tsv_gin
  ON code_symbols USING GIN (lex_tsv);
CREATE INDEX ix_code_symbols_lex_text
  ON code_symbols (lex_text);                          -- for LIKE-style
                                                       --   debug queries
CREATE INDEX ix_code_symbols_name_trgm
  ON code_symbols USING GIN (name gin_trgm_ops);        -- enables
                                                       --   fuzzy name
                                                       --   match (used
                                                       --   for the
                                                       --   Phase 6
                                                       --   ModelNameNormalizer)
```

The query:

```sql
SELECT symbol_id, ts_rank_cd(lex_tsv, query) AS score
FROM   code_symbols, websearch_to_tsquery('english', :claim) AS query
WHERE  repository_id = :repository_id
  AND  lex_tsv @@ query
ORDER  BY score DESC
LIMIT  :top_k;
```

`websearch_to_tsquery` parses the user's claim the way a search
engine would — quoted phrases stay together, `OR` is honoured,
`-foo` excludes the term. Much friendlier than `to_tsquery` for
arbitrary user input.

`pg_trgm` is enabled by 0005 with `CREATE EXTENSION IF NOT
EXISTS pg_trgm` so the migration is idempotent on databases
where 0001_init already enabled it.

### 2.2 Structural scorer — import-graph + file-name (weak v1)

The structural leg gives a **multiplier** per candidate symbol
(default 1.0). Two signals in v1:

  - **Import-graph proximity** — for each candidate symbol, look
    up its file's `kind='import'` rows; check if any of the
    claim-tokens (lowercased, alphanumeric) match an imported
    target name. Boost factor = `1.0 + 0.3 * matched_libs`.
    Capped at `1.0 + 0.9` so no single signal dominates.

  - **File-name token match** — split `symbol.file_path` by
    `/` and `.`; any token in the claim (case-insensitive,
    ≥ 3 chars) contributes `0.05 * matched_tokens` to the
    boost. Capped at `0.30`.

The structural score is **multiplicative** on the leg-fusion
side (RRF operates on ranks, then we apply the boost as a
multiplier so the rank ordering survives but scores within
the top-k are nudged).

Why "weak" and not "full": the roadmap calls for
`called-from-training-loop boost`, `loss-metric boost`,
`dependency-graph proximity`. Two of these need a call graph
(`caller_symbol_id`) and one needs an understanding of which
symbols are "in the loop" — none of which exist in the
codebase today (see §2.6). Phase 5 ships real but limited
signal; Phase 5.5 picks up the rest once `caller_symbol_id`
is wired in.

### 2.3 Hybrid ranker — Reciprocal Rank Fusion (RRF)

Three legs (semantic, lexical, structural) each return their
own ranking. We fuse them with the **standard RRF formula**:

```
score(d) = Σ_i  w_i / (k + rank_i(d))
```

  - `k = 60` (Cormack et al. 2009 default; the standard
    ElasticSearch / OpenSearch RRF constant).
  - `w_i ∈ {0.55, 0.25, 0.20}` for `{semantic, lexical,
    structural}` (the weights pre-declared in
    `evaluation/README.md:30`).
  - `rank_i(d)` is the 1-indexed rank of document `d` in leg
    `i`'s top-k result. Documents **not** in leg `i`'s top-k
    do not contribute (treated as `rank = ∞`).

The orchestrator (`hybrid_ranker`):

  1. Run each leg's top-`top_k * 3` candidates (so each leg
     has room to surface its long-tail picks).
  2. Take the union of the three result sets.
  3. Compute RRF score for each unique symbol.
  4. Multiply by the structural boost.
  5. Sort by combined score, truncate to `top_k`.
  6. Return `(symbol, final_score, breakdown)` tuples.

The `final_score` is normalised to `[0, 1]` (RRF max ≈ 0.55 /
61 = 0.009, so we scale for the UI display; raw RRF is
preserved in the breakdown for debugging).

**Why RRF over weighted-sum fusion** — semantic returns
`0..1`, BM25 returns unbounded, structural returns `1.0..1.9`.
Weighted-sum requires score normalisation; RRF operates on
**ranks** so the legs' score scales don't matter. RRF is the
standard multi-leg fusion pattern in ElasticSearch,
OpenSearch, Vespa, and Weaviate.

**Edge case** — a symbol in only one leg's top-k gets only
that leg's contribution. The RRF denominator `k + rank` keeps
the score finite even when only one leg hits.

### 2.4 HTTP surface — same endpoint, `mode` flag

Phase 4's `POST /repositories/{id}/retrieve` stays the
canonical path. The frontend `ClaimRetrievalPanel` keeps
working with no change (`mode` defaults to `"hybrid"`).

```python
class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"
    weights: RetrieveWeights | None = None

class RetrieveWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    semantic:   float = Field(default=0.55, ge=0.0, le=1.0)
    lexical:    float = Field(default=0.25, ge=0.0, le=1.0)
    structural: float = Field(default=0.20, ge=0.0, le=1.0)
```

`weights` is optional — defaults to the standard triple. Tests
can override to study per-leg contribution.

Response:

```python
class RetrieveMatch(BaseModel):       # unchanged from Phase 4
    ...
    similarity: float                 # the *final* RRF score,
                                      #   0..1 normalised

class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol_id:         UUID
    rank_semantic:     int | None
    score_semantic:    float | None
    rank_lexical:      int | None
    score_lexical:     float | None
    rank_structural:   int | None
    struct_boost:      float          # 1.0 = no boost; > 1 = boosted

class RetrieveResponse(BaseModel):
    repository_id: UUID
    claim:         str
    symbols:       list[RetrieveMatch]
    scores:        list[float]
    breakdown:     list[ScoreBreakdown]
    mode:          str
    weights:       RetrieveWeights
```

The router dispatches on `mode`:

```python
match body.mode:
    case "semantic":
        results = retrieve_semantic(db, repo_id, claim, top_k)
    case "lexical":
        results = lexical_search(db, repo_id, claim, top_k)
    case "hybrid":
        results = hybrid_ranker(db, repo_id, claim, top_k,
                                weights=body.weights)
```

`EmbeddingError` continues to surface as 500 for all three.

### 2.5 Evaluation harness — 3 hand-labelled pairs

The roadmap acceptance criterion is *"First numbers on 3
hand-annotated paper/repo pairs. Hybrid > embedding-only on
Recall@5"*. The harness:

```
evaluation/
├── baselines/
│   ├── embedding_only.py        # Wraps retrieve_semantic (Phase 4)
│   ├── lexical_only.py          # Wraps lexical_search (Phase 5)
│   └── hybrid_claim2code.py     # Wraps hybrid_ranker (Phase 5)
├── datasets/                    # NEW — 3 paper + repo pairs
│   ├── attention_is_all_you_need/
│   │   ├── paper.pdf            # Vaswani et al. 2017
│   │   └── repo/                # annotated-transformer (shallow
│   │                            #   clone, gitignored)
│   ├── bert/
│   │   ├── paper.pdf            # Devlin et al. 2018
│   │   └── repo/                # huggingface/transformers
│   │                            #   (BERT model dir only)
│   └── lora/
│       ├── paper.pdf            # Hu et al. 2021
│       └── repo/                # microsoft/LoRA
├── annotations/
│   ├── attention_is_all_you_need.json     # 10 (claim → symbol)
│   ├── bert.json                          # 10 (claim → symbol)
│   └── lora.json                          # 10 (claim → symbol)
├── evaluate_retrieval.py        # Runs the 3 baselines × 3 datasets
│                                #   → Recall@1, Recall@5, MRR
│                                #   → evaluation/results.json + report.md
└── README.md
```

Annotation file shape:

```json
{
  "pair_id": "attention_is_all_you_need",
  "paper": {"title": "Attention Is All You Need",
            "sha256": "..."},
  "repo":  {"url": "https://github.com/...", "commit": "..."},
  "claims": [
    {
      "claim_id": "c1",
      "text": "The model uses scaled dot-product attention.",
      "evidence_span": {"section": "3.2.1", "page": 5,
                        "char_start": 0, "char_end": 100},
      "relevant_symbols": [
        {"file_path": "model/attention.py",
         "name": "scaled_dot_product_attention",
         "kind": "function"}
      ],
      "expected_verification": "VERIFIED"
    },
    ...
  ]
}
```

10 claims per pair × 3 pairs = **30 `(claim → symbol)`
mappings**. Each claim maps to **one or more** relevant
symbols; the eval treats any of them as a hit.

`evaluate_retrieval.py` renders:

```
## attention_is_all_you_need
| Baseline          | Recall@1 | Recall@5 | MRR     |
|-------------------|----------|----------|---------|
| embedding_only    | 0.40     | 0.70     | 0.512   |
| lexical_only      | 0.30     | 0.60     | 0.420   |
| hybrid_claim2code | 0.50     | 0.80     | 0.620   |

## Aggregate
| Baseline          | Avg R@1  | Avg R@5  | Avg MRR  |
|-------------------|----------|----------|----------|
| embedding_only    | 0.33     | 0.63     | 0.450    |
| lexical_only      | 0.27     | 0.57     | 0.380    |
| hybrid_claim2code | 0.43     | 0.77     | 0.550    |
```

The acceptance criterion is **`hybrid.R@5 > embedding.R@5`**.
If the numbers don't support that, we iterate the structural
weights (or fall back to `semantic + lexical` only with no
structural leg) **before** shipping Phase 5. The Phase 5 done
state is "hybrid > embedding on Recall@5, measured, on the 3
labelled pairs."

### 2.6 Constraints inherited from earlier phases

These are not "decisions" — they're facts about the codebase
that shape v1:

  - **`caller_symbol_id` does not exist.** The model class
    docstring mentions it ("reserved for the Phase 3 v1
    name-resolved call graph (currently always NULL)") but no
    migration or column was ever created. **No call graph
    exists** in the codebase today.

  - **`parent_symbol_id` is always NULL in practice.** The
    workflow's two-pass walk is *described* in its docstring
    but not actually implemented — methods are inserted with
    `parent_symbol_id=None`. Phase 5.5 fixes the wiring;
    Phase 5 does not paper over it.

  - **`retrieval/` is an existing stub.** Only an empty
    `__init__.py` exists. Phase 5 fills it.

  - **`evaluation/` is an empty shell.** All `.gitkeep`
    placeholders, the Makefile `eval-retrieval` target points
    at a non-existent script, and `evaluation/README.md`
    describes the Phase 10 target layout (not Phase 5's
    actual shipped state).

  - **No GIN, no `tsvector`, no trigram indexes exist yet** —
    zero hits in the codebase.

### 2.7 What we explicitly do NOT do in Phase 5

  - ❌ Call graph (`caller_symbol_id`) — Phase 5.5+ once the
    parser is fixed
  - ❌ Training-loop / loss-metric boosts — need call graph
  - ❌ `parent_symbol_id` ancestry boost — empty data; fix the
    wiring in Phase 5.5
  - ❌ Per-language tokenisation — English-only is fine for ML
    repos and matches Postgres default
  - ❌ LLM re-ranker — Phase 6 owns verification; Phase 5 ships
    pure retrieval
  - ❌ Cross-analysis retrieval — still scoped to one
    repository
  - ❌ Asynchronous retrieval — sync, like Phase 4
  - ❌ Verifier changes — Phase 6

## 3. CI considerations

  - 0005 migration adds 1 generated `TEXT` column + 1 generated
    `tsvector` column + 2 indexes (GIN + trigram) + enables the
    `pg_trgm` extension. Pure additive schema; no destructive
    changes.
  - Unit tests for `lexical.py` use SQLite stubs for the score
    shape and a MagicMock for the DB round-trip (the real
    `tsvector @@ tsquery` query requires Postgres, which the
    unit-test environment doesn't have).
  - Integration test (`tests/integration/test_lexical_pgvector.py`,
    opt-in via `-m integration`) round-trips 5 synthetic symbols
    through `tsvector` + `websearch_to_tsquery` against a real
    Postgres.
  - `evaluation/evaluate_retrieval.py` is **NOT run in CI** —
    requires the labelled PDFs and cloned repos, both large.
    Run locally via `make eval-retrieval`.
  - The Makefile `eval-retrieval` target is updated to point at
    the now-existing script (today it points at a non-existent
    file).
  - Backend `pytest` will not exercise the 0005 migration on
    SQLite (generated columns are Postgres-only). The
    `apps/api/tests/conftest.py` test-DB setup uses SQLite, so
    the migration does not run in CI. The integration test
    exercises it on real Postgres.

## 4. Acceptance — what the doc specifies

```bash
# Local: unit suite
make test       # 256 (Phase 4) + ~30 new = ~286 unit tests pass

# Local with Postgres + OpenAI
DATABASE_URL=postgresql://... OPENAI_API_KEY=sk-... \
  make test-integration
# → tests/integration/test_lexical_pgvector.py passes

# End-to-end — pull a small public repo + retrieve against a
# claim via all 3 modes.
make dev
ANALYSIS=$(curl -sX POST -F "file=@samples/attention.pdf" \
  http://localhost:8000/papers | jq -r .analysis_id)
REPO_ID=$(curl -sX POST -H 'content-type: application/json' \
  -d '{"url":"https://github.com/harvardnlp/annotated-transformer"}' \
  http://localhost:8000/analyses/$ANALYSIS/repo | jq -r .repository_id)

# Phase 4 behaviour preserved (semantic-only).
curl -sX POST -H 'content-type: application/json' \
  -d '{"claim":"scaled dot-product attention","mode":"semantic"}' \
  http://localhost:8000/repositories/$REPO_ID/retrieve | jq '.symbols[0]'
# → {"name":"scaled_dot_product_attention","similarity":0.91,...}

# Phase 5 v1 — hybrid (default) returns the breakdown.
curl -sX POST -H 'content-type: application/json' \
  -d '{"claim":"scaled dot-product attention"}' \
  http://localhost:8000/repositories/$REPO_ID/retrieve \
  | jq '{mode,weights,breakdown}'

# Eval — runs all 3 baselines × 3 labelled datasets.
make eval-retrieval
# → evaluation/results.json + evaluation/report.md
#   showing hybrid.R@5 > embedding.R@5 (acceptance)
```

A green run of all three blocks is the Phase 5 done-state.

---

## Files the Phase 5 implementation will touch (post-doc-approval)

These are **not** in this plan's commit; they are the surface
the doc references so reviewers know what's coming:

### Backend (new)

  - `apps/api/claim2code_api/indexing/lexical.py` —
    `lexical_search(db, repo_id, claim, top_k)` returning
    `list[(symbol_id, score)]`
  - `apps/api/claim2code_api/indexing/structural.py` —
    `structural_boost(db, candidate_ids, claim) -> dict[symbol_id,
    float]` (multiplier, default 1.0)
  - `apps/api/claim2code_api/retrieval/hybrid_ranker.py` —
    `hybrid_ranker(db, repo_id, claim, *, top_k=5,
    weights=..., embedding_client=None)` orchestrating the three
    legs + RRF fusion
  - `apps/api/claim2code_api/retrieval/__init__.py` — module
    docstring + `__all__`
  - `apps/api/migrations/versions/0005_code_symbols_lexical.py` —
    adds `lex_text`, `lex_tsv`, GIN index, trigram index;
    enables `pg_trgm`

### Backend (edited)

  - `apps/api/claim2code_api/indexing/__init__.py` — module
    docstring updated to include `lexical` + `structural`
  - `apps/api/claim2code_api/repos/router.py` — extend
    `retrieve_repository` to dispatch on `mode`
  - `apps/api/claim2code_api/repos/schemas.py` — extend
    `RetrieveRequest`; add `ScoreBreakdown`, `RetrieveWeights`;
    extend `RetrieveResponse`
  - `apps/api/claim2code_api/db/models.py` (`CodeSymbol`) —
    append `lex_text` + `lex_tsv` mapped columns

### Backend (no change)

  - `apps/api/claim2code_api/indexing/retriever.py` — Phase 4
    semantic scorer kept as-is
  - `apps/api/claim2code_api/repos/workflow.py` — Phase 4
    wiring unchanged (Phase 5 is a *read* path, not a write
    path)
  - `apps/api/claim2code_api/indexing/indexer.py` — Phase 4
    unchanged
  - `deployments/Dockerfile.api` — no edit (pg_trgm ships with
    the base pgvector image)
  - `scripts/dev-up.sh` — no edit

### Tests (new)

  - `apps/api/tests/test_lexical.py` — lexical_search with
    MagicMock DB; SQL string contains `tsvector @@`, `ts_rank_cd`,
    `websearch_to_tsquery`
  - `apps/api/tests/test_structural.py` — synthetic code_symbols
    set; import-graph proximity; file-name token match;
    multiplier capping
  - `apps/api/tests/test_hybrid_ranker.py` — RRF formula
    round-trip, per-leg ranking aggregation, weight override,
    missing-leg handling, top_k truncation
  - `apps/api/tests/integration/test_lexical_pgvector.py` —
    opt-in via `-m integration`; real Postgres + tsvector
    round-trip with 5 synthetic symbols

### Tests (edited)

  - `apps/api/tests/test_repos_retrieve_endpoint.py` — extend
    existing 16 tests with: `mode="semantic"` (Phase 4 behaviour),
    `mode="lexical"` (calls `lexical_search`), `mode="hybrid"`
    default, custom `weights`, `breakdown` in response, OpenAPI
    schema updated

### Eval (new)

  - `evaluation/baselines/embedding_only.py` — thin wrapper
    around `retrieve_semantic`
  - `evaluation/baselines/lexical_only.py` — thin wrapper
    around `lexical_search`
  - `evaluation/baselines/hybrid_claim2code.py` — thin wrapper
    around `hybrid_ranker`
  - `evaluation/evaluate_retrieval.py` — runs the 3 baselines ×
    3 datasets; computes Recall@1, Recall@5, MRR; renders
    `evaluation/results.json` + `evaluation/report.md`
  - `evaluation/annotations/{attention_is_all_you_need,bert,lora}.json`
    — the 30 hand-labelled mappings
  - `evaluation/datasets/{attention_is_all_you_need,bert,lora}/`
    — the 3 PDF + cloned-repo pairs (repos `gitignored`)

### Docs (new / edited)

  - `docs/architecture/phase-5-design.md` — **the deliverable of
    THIS plan**
  - `evaluation/README.md` — updated status line
  - `README.md` — status table updated

### CI / infra (edited)

  - `.github/workflows/ci.yml` — `backend-test` job picks up the
    0005 migration via `alembic upgrade head` in the integration
    test setup; no workflow YAML change needed
  - `Makefile` — `eval-retrieval` target verified to point at the
    new script

---

## Commit plan (after doc is approved)

One branch `phase/5-hybrid-retrieval`, 7-8 logical commits,
`--no-ff` merge to main:

  1. `feat(retrieval): 0005_code_symbols_lexical migration (tsvector + GIN + trigram)`
  2. `feat(retrieval): lexical scorer (lexical_search via Postgres tsvector)`
  3. `feat(retrieval): structural boosters (import-graph + file-name proximity)`
  4. `feat(retrieval): hybrid_ranker (RRF fusion over sem + lex + struct)`
  5. `feat(retrieval): retrieve endpoint — mode flag + score breakdown`
  6. `feat(eval): baselines + evaluate_retrieval.py + 3 hand-labelled pairs`
  7. `test(retrieval): hybrid unit + integration tests + labelled fixture round-trip`
  8. `docs(retrieval): phase-5 design + README status update`

(8 commits; merge with `--no-ff`. The doc commit ships first
**within this plan** as the doc itself, on a separate
`docs/phase-5-design` branch.)

---

## Verification (of the design doc itself)

The doc is the deliverable. Verification is editorial:

  - ✓ Doc lives at `docs/architecture/phase-5-design.md`
  - ✓ Sections §1-§4 present, in the order above (mirrors
    `phase-4-design.md` structure)
  - ✓ Tone and structure match `phase-4-design.md`
  - ✓ Every "we will X" in the doc has a corresponding file in
    the post-doc-approval list (no orphan decisions)
  - ✓ Every "we will NOT do X" cites the phase that owns it
  - ✓ `0.55 / 0.25 / 0.20` weights match the pre-declared
    triple in `evaluation/README.md:30`
  - ✓ Plan reflects that `parent_symbol_id` is **empty in
    practice** (per Explore findings) — Phase 5.5 picks up the
    wiring, not Phase 5
  - ✓ Plan reflects that `caller_symbol_id` **doesn't exist**
    as a column (no call graph in v1)
  - ✓ Plan reflects that `retrieval/` is an existing stub
    directory (only `__init__.py` empty)
  - ✓ Plan reflects that the existing Makefile target
    `eval-retrieval` points at a non-existent script (fixed in
    commit 6)
  - ✓ Plan reflects that no new Python deps are needed
    (Postgres FTS is built-in, pg_trgm ships with the base
    pgvector image)

After the doc is merged to main (separate doc-only commit on a
`docs/phase-5-design` branch), the implementation branch opens
and work proceeds against the spec.

---

## Why these decisions specifically

  - **Weak struct v1 (import-graph + file-name) over call-graph
    proximity** — the call graph isn't there yet, and adding
    it would balloon Phase 5 from 8 commits to ~12. The weak
    struct is real signal (files that import torch are torch
    code; a claim mentioning torch matches them) without the
    call-graph plumbing. The full struct leg lands in Phase
    5.5 once `caller_symbol_id` is wired into the parser.

  - **Postgres `tsvector` over `rank_bm25`** — zero new deps,
    scales with the table, and matches our existing "Postgres
    does everything" storage story. Upgrading to a Python BM25
    lib is a one-file swap (the `lexical.py` public API stays
    the same).

  - **RRF over weighted-sum** — RRF doesn't require score
    normalisation (semantic returns 0..1, BM25 returns
    unbounded, structural returns 1.0..1.9). RRF operates on
    ranks, so the legs' score scales don't matter. This is the
    standard multi-leg fusion pattern in ElasticSearch /
    OpenSearch / Vespa / Weaviate.

  - **Same endpoint, `mode` flag over new sibling endpoint** —
    Phase 4's `POST /retrieve` stays the canonical path; the
    frontend `ClaimRetrievalPanel` keeps working with no change
    (default mode is `hybrid`). A new sibling endpoint would
    double the route surface for one knob.

  - **Hand-labelled pairs over synthetic eval** — the
    acceptance criterion is real numbers on real (paper, repo)
    pairs. Synthetic stubs can't defend "hybrid > embedding-
    only" with any credibility. We accept the labelling cost
    (~ 30 mappings × ~ 3 minutes each = ~ 1.5 hours of focused
    work) once per phase.

  - **3 hand-labelled pairs in Phase 5 over 1 large pair** —
    3 pairs exercises more diverse retrieval scenarios
    (attention layers, transformer blocks, LoRA adapters).
    30 mappings total is enough for the headline number;
    growing the set is a Phase 10 evaluation concern, not a
    Phase 5 acceptance concern.