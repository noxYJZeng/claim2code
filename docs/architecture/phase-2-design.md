# Phase 2 — Paper Intelligence Design

> Goal (from `phase-roadmap.md:51-63`): upload a PDF → see 5–15 structured
> technical claims with paper-evidence spans. Acceptance: `POST /papers`
> accepts a PDF upload, parses it, and extracts claims; `GET
> /analyses/{id}/claims` returns the claim list; the frontend shows an
> upload control and the claims list.

## 1. What ships

| Layer | Change |
|---|---|
| `apps/api/claim2code_api/papers/` | New package: `parser.py` (PyMuPDF), `extractor.py` (OpenAI structured outputs), `storage.py` (disk + sha256), `router.py`, `schemas.py` |
| `apps/api/claim2code_api/llm/client.py` | New: lazy `LLMClient` singleton + `check_llm_ready()` probe + `reset_llm_for_tests()` |
| `apps/api/claim2code_api/db/models.py` | Append `Paper` ORM (`analysis_id` FK + sha256 + parse_status) |
| `apps/api/migrations/versions/0002_papers.py` | New: `papers` table with FK to `analysis_states` |
| `apps/api/claim2code_api/app/main.py` | First `include_router(papers_router, …)` |
| `apps/api/claim2code_api/app/deps.py` | Add `get_llm_client()` dependency |
| `apps/api/claim2code_api/app/config.py` | Add `max_pdf_size_mb`, `min_claim_count`, `max_claim_count`, `paper_section_char_budget` |
| `apps/api/pyproject.toml` | Add `pymupdf>=1.24`, `openai>=1.50` runtime deps |
| `apps/api/tests/` | `test_paper_parser.py`, `test_claim_extractor.py`, `test_papers_endpoints.py` (all unit); `tests/integration/test_paper_upload.py` (real LLM, opt-in) |
| `apps/web/src/App.tsx` | Replace welcome page with `<UploadSection />` + `<ClaimsList />` |
| `apps/web/src/api.ts` | Add `uploadPaper`, `getClaims`, `getPaperMetadata` (mirror `getHealth`'s discriminated-union style) |
| `apps/web/src/UploadSection.tsx` | New: native `<input type="file">` + submit button |
| `apps/web/src/ClaimsList.tsx` | New: render claims as Tailwind cards |

## 2. Key decisions

### 2.1 Synchronous workflow, no background jobs

`POST /papers` runs **parse + extract inline** and returns
`{analysis_id, paper_id, parse_status, n_claims}` when done. The request
is capped by `Settings.workflow_timeout_paper` (default `180`s); a
client that exceeds it gets 504.

Why sync: Phase 2 has no concurrent load. A 180-second sync request is
acceptable for "upload a paper and wait." Adding background processing
introduces a worker pool, a message broker, idempotency keys, and a
status-poll endpoint — none of which are justified by the v1 workload.

`analysis_states.paper_status` is updated at each transition so a
future async layer (Phase 5+) can resume mid-workflow without
re-architecting the schema.

### 2.2 LLM provider — OpenAI direct, Structured Outputs

Use the `openai` Python SDK (`openai>=1.50`). The `OPENAI_*` Settings
fields declared in Phase 1 polish (`api_key`, `base_url`, `model_chat`)
already cover everything we need.

**Structured Outputs** is the load-bearing decision. We send
`response_format={"type": "json_schema", "json_schema": {...}}` with
`strict: true`, where the JSON schema is derived from a `ClaimList`
Pydantic model via `model_json_schema()`. OpenAI guarantees the
response matches the schema or fails the call — no retry-on-parse-error
hot path, no regex post-processing.

`openai_base_url` lets us point at Azure OpenAI, a local llama.cpp
proxy, or any OpenAI-compatible endpoint with **zero code change** in
Phase 2.

### 2.3 PDF parser — PyMuPDF (`pymupdf` package, `fitz` import)

PyMuPDF is the fastest mature Python PDF parser; it handles text,
tables, and equations via `page.get_text("dict")` and `"xml"`. Two
guards in `parser.py`:

1. **Reject image-only PDFs**: if `len(page.get_text("text")) < N` for
   every page, the upload returns 422 with `{"error": "pdf_has_no_text_layer"}`.
   Scanned papers are deferred (downstream `UNVERIFIED` is a Phase 6
   concern).
2. **Section detection**: rely on font-size deltas in the `"dict"`
   output — paragraphs with `font.size > body_size + 1.5` become
   `PaperSection` boundaries. Equations and tables are parsed but not
   surfaced in Phase 2.

### 2.4 Storage — disk + new `papers` table

PDF bytes go to `data/uploads/<sha256>.pdf` (the directory is already
created by `scripts/dev-up.sh:18` with a `.gitkeep`). sha256 doubles as
the dedup key and the filename — uploading the same PDF twice returns
the existing `paper_id` rather than duplicating storage.

Metadata goes in a new `papers` table:

| Column | Type | Notes |
|---|---|---|
| `paper_id` | `UUID PK` | default `uuid.uuid4` |
| `analysis_id` | `UUID NOT NULL` | FK → `analysis_states.analysis_id`, `ON DELETE CASCADE` |
| `sha256` | `CHAR(64) NOT NULL UNIQUE` | dedup key |
| `filename` | `TEXT NOT NULL` | original upload filename |
| `size_bytes` | `BIGINT NOT NULL` | |
| `n_pages` | `INT` | populated after parse |
| `n_chars` | `INT` | populated after parse |
| `uploaded_at` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | |
| `parse_status` | `TEXT NOT NULL DEFAULT 'pending'` | `'pending' \| 'parsing' \| 'extracting' \| 'done' \| 'error'` |
| `parsed_at` | `TIMESTAMPTZ NULL` | |
| `error` | `TEXT NULL` | |

Indexes: `ix_papers_analysis_id`, `uq_papers_sha256` (UNIQUE).

The migration style mirrors `0001_init.py`: `revision = "0002_papers"`,
`down_revision = "0001_init"`, JSONB cast `'[]'::jsonb`, timezone
timestamps, `ix_<table>_<col>` naming.

### 2.5 API surface

| Method | Path | Returns |
|---|---|---|
| `POST` | `/papers` (multipart) | `{analysis_id, paper_id, parse_status, n_claims}` on success; 422 on bad PDF; 504 on timeout; 500 on parse/LLM failure |
| `GET` | `/papers/{paper_id}` | Paper metadata + parse status |
| `GET` | `/analyses/{analysis_id}/claims` | The extracted claims list |

`POST /papers` flow:

```
POST /papers (multipart)
  ├─ 1. Validate (size ≤ max_pdf_size_mb, magic bytes "%PDF-")
  ├─ 2. Compute sha256, write to data/uploads/<sha256>.pdf
  ├─ 3. INSERT paper row (parse_status='parsing')
  ├─ 4. UPDATE analysis_states SET paper_status='parsing'
  ├─ 5. PyMuPDF → Paper (sections, equations, tables)
  ├─ 6. UPDATE paper SET parse_status='extracting', n_pages, n_chars
  ├─ 7. UPDATE analysis_states SET paper_status='extracting'
  ├─ 8. LLMClient.extract_claims(paper) → list[Claim]
  ├─ 9. UPDATE analysis_states SET claims=..., paper_status='done',
  │      completed_at=now()
  ├─ 10. UPDATE paper SET parse_status='done', parsed_at=now()
  └─ 11. Return {analysis_id, paper_id, parse_status, n_claims}
```

Any step's exception → catch, append to `analysis_states.errors[]`,
set `paper_status='error'` + `parse_status='error'`, return 500 with a
JSON body pointing to `GET /analyses/{id}` for the failure detail.

### 2.6 LLM client — sync, mockable, with a readiness probe

`apps/api/claim2code_api/llm/client.py` exposes `LLMClient` with:

- `extract_claims(paper: Paper) -> list[Claim]` — the Structured Outputs
  call. Validates the response with the same `ClaimList` Pydantic
  model before returning.
- `check_ready() -> tuple[bool, str | None]` — mirrors
  `check_db_ready` shape: success returns `(True, None)`; failure
  returns `(False, "<ExceptionClass>: <truncated msg>")`.

Lazy init via module-level `_client` + `get_llm_client()` (mirrors
`db/session.py:11-27`). `reset_llm_for_tests()` parallels
`reset_for_tests()`.

**Phase 2 does not wire LLM readiness into `/health/ready`** — that
change is Phase 9 (Observability) when we also add OTel. The probe
function exists now and is unit-tested; future phases consume it.

### 2.7 Claim schema (Pydantic — source of the JSON schema sent to OpenAI)

```python
class ClaimType(str, Enum):
    METHOD, IMPLEMENTATION, PERFORMANCE, ABLATION, HYPERPARAMETER,
    DATASET, COMPLEXITY, TRAINING, EVALUATION = ...

class PaperEvidence(BaseModel):
    section_title: str
    page: int
    char_start: int       # offset into the section text
    char_end: int
    quote: str            # verbatim snippet the LLM cited

class Claim(BaseModel):
    claim_id: str         # UUID4 string, generated client-side
    text: str
    claim_type: ClaimType
    confidence: float     # 0..1
    paper_evidence: PaperEvidence

class ClaimList(BaseModel):
    claims: list[Claim]   # 5..15 items; OpenAI's `minItems`/`maxItems` enforce
```

`min_claim_count` and `max_claim_count` are Settings fields so the
Phase 10 evaluation can tune the range without code change.

### 2.8 Frontend — single page, native file input, no router

Replace the Phase 1 welcome page with `<UploadSection />` + `<ClaimsList />`
stacked vertically. **No `react-router-dom`** — Phase 2 is still one
screen; the upload *is* the analysis creation.

`<input type="file" accept="application/pdf" required>` styled with
Tailwind. Native HTML5 validation; no drop-zone library in v1.

Submit flow:

1. User picks a PDF → `<input type="file">` stores it in state.
2. Submit button `POST`s as `multipart/form-data` to `/papers`.
3. On 200, store `analysis_id` + `paper_id` in state, `fetch`
   `/analyses/{analysis_id}/claims`, render.
4. On 4xx/5xx, show the server's `error` field in a red banner.

Claim card mirrors `data-flow.md` §5: claim text, type, confidence,
paper section + page, status badge (`done` only in Phase 2 — verdicts
land in Phase 6).

## 3. What we explicitly do NOT do in Phase 2

- ❌ Auth (single-tenant dev mode)
- ❌ Background job processing / queue (Celery, RQ, FastAPI BackgroundTasks)
- ❌ Equation / table extraction surfaced in the UI (parsed but not extracted into claims)
- ❌ Multi-PDF batch upload
- ❌ Claim editing UI (read-only claims list)
- ❌ Distributed tracing / OpenTelemetry (Phase 9)
- ❌ Wiring LLM readiness into `/health/ready` (Phase 9)
- ❌ Token-cost / latency metrics (Phase 9)
- ❌ Streaming responses (chunked transfer not needed for ≤ 50 MB PDFs)
- ❌ Anthropic / local LLM providers (swap `openai_base_url` later, no code change)
- ❌ Secret redaction / prompt-injection defense (Phase 8 — PDF text enters the LLM prompt in Phase 2, so the safety gate is **added in Phase 8**, not deferred silently)

## 4. CI considerations

- **Unit tests are CI-safe.** `LLMClient.extract_claims` is mocked
  via `monkeypatch.setattr("claim2code_api.app.deps.get_llm_client",
  lambda: _stub_llm)`. PyMuPDF is invoked against a tiny synthetic PDF
  in `tests/fixtures/`.
- **The real-LLM integration test is opt-in:**
  `pytest -m "integration and paper"` requires `OPENAI_API_KEY`. The
  test is decorated
  `@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs LLM key")`,
  so CI's `backend-test` job collects and skips it (no secret needed
  in CI).
- **No new CI job.** Reuse Phase 1's 4-job split
  (`scaffold-smoke` / `backend-lint` / `backend-test` /
  `frontend-build`). The new test files join the existing unit suite.
- **Backend image stays slim.** `pymupdf` adds ~30 MB; `openai` adds
  ~5 MB. Total image growth ~35 MB. Acceptable.

## 5. Acceptance — the command list

```bash
# 1. Local static checks (no LLM key required)
cd apps/api && pip install -e ".[dev]"
ruff check apps/api && mypy apps/api
cd apps/web && npm ci && npm run lint && npm test && npm run build

# 2. Local unit + integration suite (integration skipped without key)
make test                  # 7/7 unit (4 existing + 3 new) pass
OPENAI_API_KEY=sk-... make test-integration   # adds 2 integration tests

# 3. Migration
make migrate               # applies 0001_init + 0002_papers
docker exec -it claim2code-postgres psql -U claim2code -d claim2code -c "\d papers"
# → shows paper_id, analysis_id (FK), sha256 UNIQUE, …

# 4. End-to-end
cp .env.example .env       # fill OPENAI_API_KEY + POSTGRES_PASSWORD
make dev
curl -X POST -F "file=@samples/attention.pdf" http://localhost:8000/papers
# → {"analysis_id":"…","paper_id":"…","parse_status":"done","n_claims":7}

curl -s http://localhost:8000/analyses/<analysis_id>/claims | jq '.claims | length'
# → 7

# Open http://localhost:5173 in a browser → upload page + claims list
```

A green run of all four blocks is the Phase 2 done-state.

---

## Why these decisions specifically

- **OpenAI Structured Outputs over regex/JSON parsing**: guarantees
  schema-valid responses, removes retry-on-parse-error from the hot
  path, costs nothing extra.
- **Disk storage over DB BLOB**: PDFs are 1-50 MB; `bytea` and
  JSONB are both bad fits at that scale; disk-volume backup is
  simpler than full pg_dump + restore.
- **Sync endpoint over BackgroundTasks**: v1 has no concurrent load.
  A 180 s sync request is acceptable for "upload a paper and wait."
  Background adds worker pool + broker + idempotency keys + status
  polling — none justified yet.
- **New `papers` table over extending `analysis_states` JSONB**:
  `Paper` is a distinct resource with its own lifecycle (upload,
  parse, archive). Co-locating it with the analysis result blurs
  boundaries; the FK keeps referential integrity.
- **No router in Phase 2 frontend**: every Phase 2 interaction is
  still one screen. Adding `react-router-dom` would be premature;
  revisit when Phase 3 introduces the repository page.
- **Equation/table deferral**: roadmap lists them as Phase 2 ships,
  but they aren't on the v1 acceptance path (no claim cites an
  equation or table in v1). Reduces surface area; we can extract
  them in a polish commit once we have a real paper to validate.
- **`paper_status` enum tracked even though v1 is sync**: cheap to
  write, expensive to retrofit. A future async layer (Phase 5+)
  resumes mid-workflow without schema migration.