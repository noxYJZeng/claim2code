# Phase 4 — Semantic Indexing Design

> Goal (from `phase-roadmap.md:84-95`): embed each symbol's
> structured representation → retrieve Top-5 candidate symbols
> for any claim. Acceptance: given a claim string, return 5
> candidate symbols ranked by cosine similarity.

## 1. What ships

| Layer | Change |
|---|---|
| `apps/api/claim2code_api/indexing/` | New package: `__init__.py`, `embedding_client.py` (provider-agnostic interface + OpenAI adapter), `symbol_repr.py` (turns a `CodeSymbol` into a single string for embedding), `indexer.py` (batched embed + persist into `code_embeddings`), `retriever.py` (`retrieve_semantic(claim, analysis_id, top_k=5)`) |
| `apps/api/claim2code_api/db/models.py` | Append 1 ORM class: `CodeEmbedding` (one row per embedded symbol; FK → `code_symbols.symbol_id`, `repositories.repository_id`, `analysis_states.analysis_id` with `ON DELETE CASCADE`) |
| `apps/api/migrations/versions/0004_code_embeddings.py` | New: `code_embeddings` table + HNSW index on the `embedding` column + pgvector `<=>` cosine operator. Depends on `0003_repositories` (needs `code_symbols`) |
| `apps/api/claim2code_api/repos/workflow.py` | **Extend** `run_repo_workflow`: after symbols are persisted, call `indexer.embed_and_persist_symbols(db, repository_id)` so a repo is searchable the moment `POST /repositories` returns |
| `apps/api/claim2code_api/repos/router.py` | Add `POST /repositories/{repository_id}/retrieve` (body `{claim, top_k}`) → returns `{symbols, scores}` |
| `apps/api/claim2code_api/app/config.py` | Add 2 new Settings fields: `embed_batch_size` (default 128), `embed_max_retries` (default 3). The pre-existing `openai_api_key`, `openai_base_url`, `openai_model_embed`, `openai_embed_dim` are reused as-is (declared in Phase 1 polish) |
| `apps/api/pyproject.toml` | Add `pgvector>=0.3` runtime dep (psycopg's pgvector adapter — pure Python, no system lib needed) |
| `apps/api/tests/` | `test_embedding_client.py` (mocked OpenAI), `test_symbol_repr.py` (string shape), `test_indexer.py` (batched embed + persist), `test_retriever.py` (cosine search round-trip), `test_repos_retrieve_endpoint.py` (HTTP); `tests/integration/test_code_embeddings_pgvector.py` (real Postgres + pgvector, opt-in) |
| `apps/web/src/api.ts` | Add `retrieveCode(repositoryId, claim, topK)` |
| `apps/web/src/ClaimCard.tsx` (or new `<ClaimRetrievalButton>`) | New: "Find matching code" button on each claim card → opens a side-panel with the top-5 retrieved symbols + similarity scores |
| `apps/web/src/App.test.tsx` | Add 1 test: a click on the retrieval button opens the panel and renders the symbols |

## 2. Key decisions

### 2.1 Embedding provider — OpenAI `text-embedding-3-small` (1536 dims)

Pre-decided in Phase 1 polish (`openai_model_embed`, `openai_embed_dim`).
Rationale:

  - **Same vendor as the LLM.** Phase 2 already uses OpenAI for claim
    extraction; reusing the SDK + base URL + key means no new auth
    surface.
  - **1536 dims is the sweet spot.** `text-embedding-3-large` (3072
    dims) gives ~5 % better Recall@5 on academic benchmarks but
    costs 10× more. v1 prefers cost.
  - **Provider-agnostic interface.** `EmbeddingClient` is an abstract
    base class; the OpenAI adapter is one implementation. A
    `LocalEmbeddingClient` (sentence-transformers) or `CohereClient`
    can be added later without touching the rest of the code.

The interface:

```python
class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed N strings → N vectors, each of dim openai_embed_dim."""
```

`retrieve_semantic` doesn't care which provider.

### 2.2 Per-symbol embedding (1 row per `CodeSymbol`)

The design doc says "embed each symbol's structured representation".
That means **one embedding row per `CodeSymbol` row**, not per file,
not per chunk. Justification:

  - A typical ML repo produces 5-50 k symbols. At ~200 tokens each,
    10 k symbols × 200 = 2 M tokens → ~$0.04 on OpenAI. Cheap.
  - The Phase 6 verifier wants the *exact* symbol that matches a
    claim, not a "this file mentions it" hit. Per-symbol embeddings
    give that for free.
  - Per-file embeddings (or per-chunk) would force Phase 5-6 to
    re-rank chunks against symbols, doubling the work.

**Chunking deferred.** A 500-line class that doesn't fit in one
embedding context is rare in ML repos; we accept the edge case and
log a warning when a symbol's representation is > 8 000 tokens (the
OpenAI embedding input limit). Phase 4.5+ can add hierarchical
chunking if needed.

### 2.3 `code_embeddings` schema (pgvector + HNSW)

```sql
CREATE TABLE code_embeddings (
    embedding_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol_id       UUID NOT NULL REFERENCES code_symbols(symbol_id)
                     ON DELETE CASCADE,
    repository_id   UUID NOT NULL REFERENCES repositories(repository_id)
                     ON DELETE CASCADE,
    analysis_id     UUID NOT NULL REFERENCES analysis_states(analysis_id)
                     ON DELETE CASCADE,
    embedding       VECTOR(1536) NOT NULL,
    model_name      TEXT NOT NULL,                -- e.g. text-embedding-3-small@2024-01
    embedded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (symbol_id, model_name)                -- re-embed = replace row
);

CREATE INDEX ix_code_embeddings_repository_id ON code_embeddings (repository_id);
CREATE INDEX ix_code_embeddings_analysis_id   ON code_embeddings (analysis_id);
CREATE INDEX ix_code_embeddings_hnsw
    ON code_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Notes:

  - **`UNIQUE (symbol_id, model_name)`** lets us upgrade the
    embedding model without orphaning rows: re-embed everything
    under the new model name; old rows stay (rebuildable
    from symbols) but become unreferenced.
  - **Denormalised `repository_id` + `analysis_id`** so the
    retriever doesn't need a 2-hop join through `code_symbols`.
  - **HNSW with `m=16, ef_construction=64`** — HNSW is pgvector's
    default and beats IVFFlat for our scale (5-50 k vectors).
    `m=16` is the literature default; `ef_construction=64` trades
    index-build time for query Recall@5.

### 2.4 Symbol representation string format

A `CodeSymbol` becomes a single string via
`symbol_repr.format_symbol(sym)`. The format is deliberately
structured (key: value pairs) so the embedding captures structure,
not just prose:

```
File: src/pkg/core.py
Language: python
Kind: function
Name: add
Signature: (a: int, b: int) -> int
Docstring: Add two ints.
Imports: numpy as np, torch
Source:
def add(a: int, b: int) -> int:
    """Add two ints."""
    return a + b
```

  - **`Source`** is the symbol's text body (between start_line and
    end_line), truncated at 2 000 chars. The full body is rarely
    needed — the docstring + signature + name usually pin the
    intent.
  - **`Imports`** is the **symbol's** imports (not the file's),
    resolved against the file's `import` symbols we already
    captured in Phase 3.
  - **Format mirrors what the design doc says:** `File: …,
    Symbol: …, Signature: …, Docstring: …, Source: …,
    Dependencies: …`. `Dependencies` is renamed to `Imports` for
    clarity (the docs word "dependencies" overloads with Phase 3's
    lockfile dependencies).

### 2.5 Indexing workflow — embed inside `run_repo_workflow`

Phase 3's `run_repo_workflow` already returns once the rows are
persisted. Phase 4 extends it:

```python
def run_repo_workflow(db, analysis_id, url):
    repo = clone + walk + parse + persist           # Phase 3 logic
    embed_and_persist_symbols(db, repo.repository_id)  # Phase 4 add
    return repo
```

`embed_and_persist_symbols` uses OpenAI's batch endpoint
(up to 2 048 inputs per call) at `Settings.embed_batch_size`
(default 128, conservative). It writes the embeddings + clears the
"parsing" status to "indexed".

  - **Why inline, not a separate endpoint:** every successful
    `POST /repositories` should leave the repo immediately
    retrievable. A separate `POST /repositories/{id}/index` would
    mean a 2-call UX and a half-indexed repo in flight.
  - **Failure mode:** if OpenAI is down or the API key is invalid,
    the repo row stays in `index_status='error'` and an entry is
    added to `analysis_states.errors`. The repo is still browsable
    via `GET /repositories/{id}/symbols` (Phase 3 rows exist);
    retrieval will simply return `[]` until the user retries.

### 2.6 Retrieval API

`retrieve_semantic(claim, analysis_id, top_k=5)` returns
`list[tuple[CodeSymbol, float]]`:

```sql
SELECT
    symbol_id,
    (embedding <=> $1::vector) AS distance
FROM code_embeddings
WHERE analysis_id = $2
ORDER BY embedding <=> $1::vector
LIMIT $3;
```

`<=>` is pgvector's **cosine distance** operator (`1 - cosine_similarity`).
We surface `similarity = 1 - distance` to the caller (intuitive 0..1 range).

Steps in `retriever.retrieve_semantic`:

  1. Embed the claim (1 OpenAI call, `embed([claim])`).
  2. Run the SQL above with the claim's vector as `$1`.
  3. Bulk-fetch the `CodeSymbol` rows by `symbol_id`.
  4. Return `[(symbol, 1.0 - distance), ...]` in distance-ascending order.

HTTP endpoint: `POST /repositories/{repository_id}/retrieve` (body
`{claim: str, top_k: int = 5}`) → `{symbols: [...], scores: [...]}`.
We chose `/repositories/...` over `/analyses/...` because the
index lives per-repo, and an analysis with multiple repos wants to
retrieve against one specific repo at a time.

### 2.7 Re-embedding on model upgrade

If the user changes `openai_model_embed` from
`text-embedding-3-small` to `text-embedding-3-large` (or upgrades
the model version), existing embeddings become stale. Strategy:

  - The `model_name` column on `code_embeddings` records the exact
    model used (e.g. `text-embedding-3-small@2024-01`).
  - The `embedding_model` column on `repositories` records the
    current canonical model (set during `embed_and_persist_symbols`).
  - `retrieve_semantic` filters on the **analysis's** canonical model.
    If no rows match (because the user upgraded but didn't
    re-index), the API returns 503 with `detail: "repository not
    indexed under model X; re-index via POST /repositories/{id}/index"`.
  - The **re-index endpoint** (`POST /repositories/{id}/index`)
    is a Phase 4.5+ add; v1 ships the inline embed and assumes
    model changes are rare.

### 2.8 What we explicitly do NOT do in Phase 4

- ❌ Multi-vector per symbol (different chunks of the same class).
  One embedding per `CodeSymbol` row.
- ❌ Hybrid retrieval (semantic + lexical + structural). That's
  Phase 5.
- ❌ Re-ranking via an LLM ("here are 50 candidates, pick the best 5").
  Phase 6's verifier uses pure cosine distance in v1.
- ❌ Embedding the **paper claims** themselves. Phase 5 indexes
  claims separately if needed.
- ❌ Embedding **configs / lockfiles** as their own rows. They're
  retrievable by Phase 6's deterministic checkers, not by
  cosine search.
- ❌ Asynchronous embedding pipeline. Same argument as Phase 3 —
  no concurrent load yet.
- ❌ Local embeddings (sentence-transformers). Optional Phase 4.5+
  adapter; v1 is OpenAI only.

## 3. CI considerations

- `pgvector>=0.3` adds a pure-Python dep (psycopg's pgvector
  adapter). No system library needed; the Postgres image already
  has the extension.
- The existing `pgvector/pgvector:pg16` image already has the
  `vector` extension installed from 0001_init.
- Unit tests mock `EmbeddingClient` so they don't hit OpenAI.
- Integration test (`test_code_embeddings_pgvector.py`, opt-in
  via `-m integration`) requires both `OPENAI_API_KEY` and
  `DATABASE_URL`. It embeds 3 synthetic symbols and verifies the
  cosine search round-trips.
- The new endpoint test (`test_repos_retrieve_endpoint.py`)
  stubs the OpenAI client; CI-safe.
- One new endpoint (`POST /repositories/{id}/retrieve`) — minimal
  routing surface.

## 4. Acceptance — what the doc specifies

```bash
# Local: unit suite
make test       # 192 (Phase 3) + ~40 new = ~232 unit tests pass

# Local with OpenAI + Postgres
OPENAI_API_KEY=sk-... make test-integration

# End-to-end
make dev
ANALYSIS=$(curl -sX POST -F "file=@samples/attention.pdf" \
  http://localhost:8000/papers | jq -r .analysis_id)

# Bind a repo + let Phase 4 index it inline.
curl -sX POST -H 'content-type: application/json' \
  -d "{\"url\":\"https://github.com/pytorch/pytorch\",\"analysis_id\":\"$ANALYSIS\"}" \
  http://localhost:8000/repositories
# → {"repository_id":"…","n_files":…,"n_symbols":…,"index_status":"indexed"}

# Retrieve against a claim.
curl -sX POST -H 'content-type: application/json' \
  -d '{"claim":"attention uses softmax(QK^T / sqrt(d))","top_k":5}' \
  http://localhost:8000/repositories/$REPO/retrieve
# → {"symbols":[{"name":"scaled_dot_product_attention", ...}, ...],
#    "scores":[0.91, 0.83, 0.79, ...]}
```

A green run of all three blocks is the Phase 4 done-state.

---

## Files the Phase 4 implementation will touch (post-doc-approval)

These are **not** in this plan's commit; they are the surface
the doc references so reviewers know what's coming:

### Backend (new)
- `apps/api/claim2code_api/indexing/__init__.py`
- `apps/api/claim2code_api/indexing/embedding_client.py` —
  abstract `EmbeddingClient` + `OpenAIEmbeddingClient` adapter +
  `EmbeddingError` exception
- `apps/api/claim2code_api/indexing/symbol_repr.py` —
  `format_symbol(sym: CodeSymbol) -> str` builds the structured
  representation (deterministic — used by tests)
- `apps/api/claim2code_api/indexing/indexer.py` —
  `embed_and_persist_symbols(db, repository_id)` batches + persists
- `apps/api/claim2code_api/indexing/retriever.py` —
  `retrieve_semantic(db, repository_id, claim, top_k=5)`
- `apps/api/migrations/versions/0004_code_embeddings.py` —
  `code_embeddings` table + HNSW index + downgrade()

### Backend (edited)
- `apps/api/claim2code_api/db/models.py` — append `CodeEmbedding`
- `apps/api/claim2code_api/repos/workflow.py` — call
  `embed_and_persist_symbols` after parsing
- `apps/api/claim2code_api/repos/router.py` — add
  `POST /repositories/{id}/retrieve`
- `apps/api/claim2code_api/repos/schemas.py` — add
  `RetrieveRequest`, `RetrieveResponse`
- `apps/api/claim2code_api/db/models.py` (Repository) — add
  `embedding_model: str | None` column
- `apps/api/migrations/versions/0003_repositories.py` — add the
  `embedding_model` column (small backwards-compat addition)
- `apps/api/claim2code_api/app/config.py` — add 2 new Settings
  fields: `embed_batch_size`, `embed_max_retries`

### Backend (no change)
- `apps/api/claim2code_api/app/config.py` —
  `openai_api_key`, `openai_base_url`, `openai_model_embed`,
  `openai_embed_dim` are reused as-is (Phase 1 polish)
- `deployments/Dockerfile.api` — no edit (pgvector image is set
  in 0001_init)
- `scripts/dev-up.sh` — no edit

### Tests (new)
- `apps/api/tests/test_embedding_client.py` — mocked OpenAI SDK
  + retry / batch logic
- `apps/api/tests/test_symbol_repr.py` — exact string shape per
  symbol kind (function / class / import)
- `apps/api/tests/test_indexer.py` — `embed_and_persist_symbols`
  with mocked client
- `apps/api/tests/test_retriever.py` — embed + pgvector cosine
  round-trip against in-memory SQLite (note: requires a real
  Postgres for the pgvector `<=>` operator — so this lives in
  `tests/integration/test_retriever_pgvector.py`)
- `apps/api/tests/test_repos_retrieve_endpoint.py` — TestClient
  + dep override
- `apps/api/tests/integration/test_code_embeddings_pgvector.py`
  — opt-in; embeds 3 symbols via real OpenAI + Postgres

### Frontend (new / edited)
- `apps/web/src/api.ts` — add `retrieveCode(repositoryId, claim, topK)`
- `apps/web/src/ClaimRetrievalPanel.tsx` (new) — top-5 symbols +
  scores, opened from a button on each claim card
- `apps/web/src/ClaimsList.tsx` — add the "Find matching code"
  button per card
- `apps/web/src/App.test.tsx` — extend with retrieval-panel test

### Docs (new / edited)
- `docs/architecture/phase-4-design.md` — **the deliverable of
  THIS plan**

### CI / infra (edited)
- `.github/workflows/ci.yml` — `backend-test` job picks up
  `pgvector` via `pip install -e .`; no workflow YAML change needed
- `deployments/Dockerfile.api` — no edit

---

## Commit plan (after doc is approved)

One branch `phase/4-semantic-indexing`, 6-8 logical commits,
`--no-ff` merge to main:

1. `feat(index): EmbeddingClient interface + OpenAI adapter`
2. `feat(index): symbol_repr.format_symbol + tests`
3. `feat(index): 0004_code_embeddings migration + CodeEmbedding ORM`
4. `feat(index): batched embed_and_persist_symbols (indexer)`
5. `feat(index): retrieve_semantic (pgvector cosine)`
6. `feat(repo): POST /repositories/{id}/retrieve endpoint + workflow integration`
7. `test(index): unit + integration tests`
8. `feat(web): ClaimRetrievalPanel + "Find matching code" button`

(8 commits; merge with `--no-ff`. The doc commit ships first
**within this plan** as the doc itself, on a separate
`docs/phase-4-design` branch.)

---

## Verification (of the design doc itself)

The doc is the deliverable. Verification is editorial:

- ✓ Doc lives at `docs/architecture/phase-4-design.md`
- ✓ Sections §1-§4 present, in the order above
- ✓ Tone and structure match `phase-3-design.md`
- ✓ Every "we will X" in the doc has a corresponding file in the
  post-doc-approval list (no orphan decisions)
- ✓ Every "we will NOT do X" cites the phase that owns it
- ✓ Embedding provider + dimensions + cost are verified against
  OpenAI's pricing page (Jan 2026): text-embedding-3-small =
  $0.02 / 1 M tokens, 1536 dims, 8 192 max input tokens
- ✓ Plan reflects that `openai_api_key`, `openai_base_url`,
  `openai_model_embed`, `openai_embed_dim` already exist in
  `config.py:60-64` (Phase 1 polish) — no env-var additions in
  Phase 4 beyond `embed_batch_size` + `embed_max_retries`
- ✓ Plan reflects that pgvector's `<=>` (cosine distance) is
  supported on Postgres 16 + the `vector` extension installed by
  0001_init
- ✓ Plan reflects that HNSW is pgvector's default and outperforms
  IVFFlat at our scale (5-50 k vectors)
- ✓ Plan reflects that 0003's `repositories` table is being lightly
  extended (1 column) rather than replaced — keeps the migration
  surface small

After the doc is merged to main (separate doc-only commit on a
`docs/phase-4-design` branch), the implementation branch opens and
work proceeds against the spec.

---

## Why these decisions specifically

- **OpenAI over local embeddings:** same vendor as Phase 2 LLM;
  zero new auth. Local sentence-transformers is a Phase 4.5+
  alternative for cost-sensitive users.
- **One embedding per `CodeSymbol`, not per chunk:** the Phase 6
  verifier wants the exact matching symbol, not "this file
  mentions it". Per-symbol gives that for free.
- **HNSW over IVFFlat:** at our scale (5-50 k vectors per repo),
  HNSW is both faster to query and simpler to maintain (no
  `lists` parameter to tune). IVFFlat's advantage kicks in at
  millions of vectors.
- **`UNIQUE (symbol_id, model_name)`:** lets us upgrade the
  embedding model without writing a destructive migration; the
  old rows just become unreferenced and a `VACUUM` reclaims the
  space.
- **Inline embedding in `run_repo_workflow`:** every successful
  clone leaves the repo immediately retrievable. A separate
  index endpoint would mean a 2-call UX and a half-indexed
  window where the user could try to retrieve.
- **`POST /repositories/{id}/retrieve` not `POST /analyses/...`:** the
  index lives per-repo; analyses with multiple repos need to
  retrieve against one specific repo at a time.
- **Cosine distance, not L2:** semantic similarity is naturally
  direction-based, not magnitude-based. Cosine matches the
  OpenAI API's notion of similarity directly.
- **No router in Phase 4 frontend:** still one screen, just
  longer. Phase 5+ lifts `<ClaimRetrievalPanel>` into its own
  route when the retrieval UX grows.