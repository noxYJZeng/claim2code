# Phase 3 — Repository Intelligence Design

> Goal (from `phase-roadmap.md:67-80`): paste a GitHub URL → see
> functions/classes/configs with exact file:line locations. Acceptance:
> `POST /repositories` accepts a GitHub URL, clones it, parses symbols
> and configs; `GET /repositories/{id}/symbols` and
> `GET /repositories/{id}/configs` return the parsed shape; the
> frontend shows a repo section with the file-tree and symbol detail.

## 1. What ships

| Layer | Change |
|---|---|
| `apps/api/claim2code_api/repos/` | New package: `cloner.py` (`git clone --depth 1` + URL/size guards), `parser.py` (tree-sitter multi-language symbol extraction + Python `ast` fallback for argparse / dataclasses), `configs.py` (YAML/JSON/TOML/INI readers), `lockfiles.py` (requirements / poetry / npm / Pipfile extractors), `storage.py` (`data/repos/` path resolution + walk-ignore patterns), `router.py`, `schemas.py` |
| `apps/api/claim2code_api/db/models.py` | Append 4 ORM classes: `Repository`, `CodeSymbol`, `RepositoryConfig`, `RepositoryDependency` |
| `apps/api/migrations/versions/0003_repositories.py` | New: `repositories` + `code_symbols` + `repository_configs` + `repository_dependencies` tables, all FK → `analysis_states.analysis_id` with `ON DELETE CASCADE` |
| `apps/api/claim2code_api/app/main.py` | Second `include_router(repos_router, …)` |
| `apps/api/claim2code_api/app/config.py` | Add **3** new Settings fields (`repo_size_estimate_timeout_seconds`, `repo_parse_extensions`, `repo_parse_ignore_dirs`); the pre-existing `github_token`, `max_repo_mb_megs`, `max_repo_files`, `max_file_kb`, and `workflow_timeout_repo` from Phase 1 polish are reused |
| `apps/api/pyproject.toml` | Add `tree-sitter>=0.23`, `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`, `tree-sitter-java`, `tree-sitter-go`, `tree-sitter-rust`, `PyYAML>=6.0` runtime deps |
| `apps/api/tests/` | `test_repo_parser.py`, `test_repo_cloner.py`, `test_repo_configs.py`, `test_repo_lockfiles.py`, `test_repo_workflow.py`, `test_repos_endpoints.py` (all unit); `tests/integration/test_repo_clone.py` (real `git clone`, opt-in) |
| `apps/web/src/App.tsx` | Add third section: `<RepositorySection />` below `<ClaimsList />` |
| `apps/web/src/api.ts` | Add `cloneRepo`, `getRepository`, `getSymbols`, `getConfigs` (mirror `getClaims`'s discriminated-union style) |
| `apps/web/src/RepositorySection.tsx` | New: GitHub URL input + status badge |
| `apps/web/src/SymbolsList.tsx` | New: render symbols grouped by file path, paginated per-file |

## 2. Key decisions

### 2.1 Synchronous workflow, no background jobs (yet)

`POST /repositories` runs **clone + walk + parse + persist inline** and
returns `{repository_id, n_files, n_symbols}` when done. The request is
capped by `Settings.workflow_timeout_repo` (default `300`s, declared in
Phase 1 polish); a client that exceeds it gets 504.

Why sync: same argument as Phase 2 — no concurrent load yet. Cloning a
shallow copy of a typical ML repo takes 5-30 s; parsing 50k symbols
takes another 10-30 s. Adding background processing introduces a worker
pool + idempotency keys + status polling — none of which are justified
by the v1 workload. A future async layer (Phase 5+) can resume mid-
workflow because `analysis_states.repository_status` is updated at each
transition.

### 2.2 Parser — tree-sitter with six grammars

Use the `tree-sitter` Python binding with six grammars:
`tree-sitter-python`, `tree-sitter-javascript`,
`tree-sitter-typescript` (covers `.ts` + `.tsx`),
`tree-sitter-java`, `tree-sitter-go`, `tree-sitter-rust`. C/C++ is
deferred to a later phase (PHP/Ruby/Kotlin/Swift/Scala likewise) — see
§3.

For **Python only**, an additional pass over `argparse.ArgumentParser`
calls and `@dataclass` definitions (via the stdlib `ast` module) lifts
CLI / config schema into `CodeSymbol` records with
`kind="config_schema"`. This is the Python-specific extra beyond what
the roadmap's "argparse / dataclasses" callout covers; the rest of the
tree-sitter parsers don't need it.

Output is a normalised `list[CodeSymbol]`:

| Field | Source |
|---|---|
| `kind` | `function \| method \| class \| import \| constant \| config_schema` |
| `language` | `python \| javascript \| typescript \| java \| go \| rust` |
| `file_path` | repo-relative POSIX path |
| `start_line` / `start_col` / `end_line` / `end_col` | AST node byte ranges |
| `name` | identifier (function name, class name, import target) |
| `signature` | best-effort textual signature — parens arg list for funcs, full dotted import for imports |
| `parent_symbol_id` | FK to enclosing class/module (NULL for module-level) |
| `docstring` | first string literal in function/class body (best-effort, NULL if absent) |

**Symbol graph (who calls whom):** Phase 3 builds a **name-based**
call graph. For each function/method body, we record calls whose
callee name resolves to another `CodeSymbol.name` in the same repo.
Resolution is name-only (no type info) so collisions are possible; we
prefer the closest-by-line-range match. A full type-resolved call graph
needs a language server and is deferred — see §3.

Rationale over Python `ast`: tree-sitter survives malformed code (common
in half-finished research repos) and unifies Python / JS / TS / Java /
Go / Rust behind one normalised shape — Phase 4-6 don't need to know
the source language.

### 2.3 Repository acquisition — `git clone --depth 1` to local disk

`POST /repositories` runs `git clone --depth 1 <url>
data/repos/<owner>__<repo>/`. The clone path is derived from the URL
(`https://github.com/<owner>/<repo>.git` → `data/repos/<owner>__<repo>/`,
double-underscore joining owner/repo to avoid path ambiguity). The
directory is pre-provisioned by `scripts/dev-up.sh:18` with a
`.gitkeep`.

Three guards before the clone:

1. **URL is `https://github.com/<owner>/<repo>.git`** — no other hosts
   in v1 (GitLab / Bitbucket / self-hosted deferred — see §3).
2. **`max_repo_mb_megs` cap** enforced by checking the
   `Content-Length` header on a HEAD request first (best effort —
   bypassed if no header or if `Content-Length` is missing). With
   `GITHUB_TOKEN` (Phase 1 polish already declared the setting), the
   rate limit jumps from 60/h to 5000/h; the token is used only for
   the HEAD pre-flight, not the clone itself (`git clone` streams over
   HTTPS anonymously; rate-limit applies to the API).
3. **Per-file `max_file_kb` cap** during the walk so a single huge
   checkpoint blob doesn't blow the cap.

If a clone for this `(owner, name)` already exists at the target path,
the cloner refuses rather than re-cloning; the user must explicitly
`DELETE /repositories/{id}` first. (Phase 3 does not ship re-clone
semantics.)

### 2.4 Storage — disk + new `repositories` table

Repo bytes live at `data/repos/<owner>__<repo>/` (a sibling of
`data/uploads/`). sha256 (the Phase 2 dedup key) is replaced by
**clone path** as the identity — `data/repos/<owner>__<repo>/.git/HEAD`
is the canary that the clone is real.

`repositories` table mirrors `papers`:

| Column | Type | Notes |
|---|---|---|
| `repository_id` | `UUID PK` | default `uuid.uuid4` |
| `analysis_id` | `UUID NOT NULL` | FK → `analysis_states.analysis_id`, `ON DELETE CASCADE` |
| `url` | `TEXT NOT NULL` | canonical GitHub URL (deduped UNIQUE) |
| `owner` | `TEXT NOT NULL` | parsed from URL |
| `name` | `TEXT NOT NULL` | parsed from URL |
| `default_branch` | `TEXT NOT NULL DEFAULT 'main'` | from `HEAD` |
| `clone_path` | `TEXT NOT NULL` | relative to repo root, e.g. `data/repos/foo__bar/` |
| `n_files` | `INT` | post-walk count |
| `n_symbols` | `INT` | post-walk count |
| `cloned_at` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | |
| `index_status` | `TEXT NOT NULL DEFAULT 'pending'` | `'pending' \| 'cloning' \| 'parsing' \| 'done' \| 'error'` |
| `indexed_at` | `TIMESTAMPTZ NULL` | |
| `error` | `TEXT NULL` | |

Indexes: `ix_repositories_analysis_id`, `uq_repositories_url` (UNIQUE
on `(owner, name)` — see §2.4 rationale below).

The migration style mirrors `0002_papers.py` exactly: `revision =
"0003_repositories"`, `down_revision = "0002_papers"`, same header
rationale block, FK declared inline via `sa.ForeignKey(..., ondelete=
"CASCADE")`, indexes via separate `op.create_index(...)` calls,
`downgrade()` drops indexes in reverse then drops tables.

**Unique constraint rationale:** `uq_repositories_url` lives on
`(owner, name)` rather than the raw URL string, because GitHub URLs
have many equivalent forms (`https://github.com/foo/bar`,
`https://github.com/foo/bar.git`, `git@github.com:foo/bar.git`). The
clone step normalises to `https://github.com/<owner>/<repo>.git` before
inserting; two requests for the same repo land on the same row.

### 2.5 API surface

| Method | Path | Returns |
|---|---|---|
| `POST` | `/repositories` (JSON body `{url, analysis_id}`) | `{repository_id, n_files, n_symbols}` on success; 422 on bad URL; 413 on too-large; 504 on timeout; 500 on clone / parse failure |
| `GET` | `/repositories/{repository_id}` | Repo metadata + index status |
| `GET` | `/repositories/{repository_id}/symbols` | The extracted `list[CodeSymbol]` (paginated per-file in v1, default top 50 per file) |
| `GET` | `/repositories/{repository_id}/configs` | The parsed config snapshots (`list[RepositoryConfig]`) |
| `GET` | `/repositories/{repository_id}/dependencies` | The flattened dependency list (`list[RepositoryDependency]`) |
| `DELETE` | `/repositories/{repository_id}` | 204 on success; also removes the cloned directory |

`POST /repositories` flow:

```
POST /repositories ({url, analysis_id})
  ├─ 1. Validate URL (must be https://github.com/<owner>/<repo>.git)
  ├─ 2. HEAD GitHub for Content-Length (best-effort size guard)
  ├─ 3. INSERT repository row (index_status='cloning')
  ├─ 4. UPDATE analysis_states SET repository_status='cloning'
  ├─ 5. git clone --depth 1 <url> data/repos/<owner>__<repo>/
  ├─ 6. UPDATE repository SET index_status='parsing'
  ├─ 7. Walk tree (skip node_modules, .git, dist/, build/, target,
  │      __pycache__, .venv)
  ├─ 8. Per file: tree-sitter parse → CodeSymbol records
  ├─ 9. Python files only: ast parse → argparse/dataclass records
  ├─ 10. Config files (.yaml/.yml/.json/.toml/.ini) → parsed rows
  ├─ 11. Lockfiles (requirements*.txt / poetry.lock /
  │      package-lock.json / Pipfile.lock) → flattened dep rows
  ├─ 12. UPDATE repository SET index_status='done', n_files,
  │      n_symbols
  ├─ 13. UPDATE analysis_states SET repository_status='done',
  │      completed_at=now()
  └─ 14. Return {repository_id, n_files, n_symbols, ...}
```

Any step's exception → catch, append to `analysis_states.errors[]`,
set `repository_status='error'` + `index_status='error'`, return 500
with a JSON body pointing to `GET /repositories/{id}` for the failure
detail.

### 2.6 CodeSymbol storage — `code_symbols` table

Phase 3 needs the symbols accessible to Phase 4-6 for embedding and
retrieval. A separate table keeps the row count manageable (a 10k-file
repo produces ~50k symbols):

| Column | Type | Notes |
|---|---|---|
| `symbol_id` | `UUID PK` | default `uuid.uuid4` |
| `repository_id` | `UUID NOT NULL` | FK → `repositories.repository_id`, `ON DELETE CASCADE` |
| `kind` | `TEXT` | enum-as-string |
| `language` | `TEXT` | enum-as-string |
| `file_path` | `TEXT` | repo-relative POSIX |
| `start_line` / `start_col` / `end_line` / `end_col` | `INT` | |
| `name` | `TEXT` | |
| `signature` | `TEXT` | best-effort |
| `docstring` | `TEXT` | best-effort, NULL if absent |
| `parent_symbol_id` | `UUID NULL` | self-FK to enclosing class/module |
| `caller_symbol_id` | `UUID NULL` | self-FK to the called symbol (name-based resolution) |

A pgvector `embedding VECTOR(1536)` column is **added in Phase 4**
(one migration per phase keeps each diff reviewable).

Indexes: `ix_code_symbols_repository_id`, `ix_code_symbols_file_path`,
`ix_code_symbols_name`.

### 2.7 Config files + lockfiles

Config snapshots go into `repository_configs` (one row per file, JSONB
content):

| Column | Type | Notes |
|---|---|---|
| `config_id` | `UUID PK` | default `uuid.uuid4` |
| `repository_id` | `UUID FK` | ON DELETE CASCADE |
| `file_path` | `TEXT` | repo-relative |
| `format` | `TEXT` | `yaml \| json \| toml \| ini` |
| `content_json` | `JSONB` | parsed tree, or raw dict for unstructured files |
| `size_bytes` | `INT` | file size on disk |
| `extracted_at` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | |

Lockfile extractions go into `repository_dependencies` (flattened):

| Column | Type | Notes |
|---|---|---|
| `dep_id` | `UUID PK` | default `uuid.uuid4` |
| `repository_id` | `UUID FK` | ON DELETE CASCADE |
| `ecosystem` | `TEXT` | `pypi \| npm` |
| `name` | `TEXT` | package name |
| `version_spec` | `TEXT` | `==2.1.0`, `^1.0.0`, etc. |
| `source_file` | `TEXT` | e.g. `requirements.txt`, `package-lock.json` |

Parsers:
- `requirements*.txt` — one `name + version_spec` per non-comment line.
- `poetry.lock` — JSON; uses `[package[]]` array with `name` + `version`.
- `package-lock.json` — JSON; uses `packages` dict, joining `node_modules/<name>` keys with the `version` field.
- `Pipfile.lock` — JSON; uses `_meta.packages` for the resolved set.

Phase 4-5 use these rows directly to verify "paper says PyTorch 2.1,
code says `torch==2.1.0`."

### 2.8 Frontend — third section, no router

Three sections stacked: `<UploadSection />` → `<ClaimsList />` →
`<RepositorySection />`. After the user enters a GitHub URL, `POST
/repositories` clones + indexes; on success, the section reveals
`<SymbolsList />` grouped by file path, paginated to the top 50 symbols
per file for v1.

**No router in Phase 3** — still one screen, just longer. Phase 4
(embedding visualisation) is when a router finally pays for itself.

`<RepositorySection />` is a text input + submit button:
1. User pastes a GitHub URL → input stores it in state.
2. Submit button `POST`s as `application/json` to `/repositories`.
3. On 200, store `repository_id` in state, `fetch`
   `/repositories/{repository_id}/symbols`, render.
4. On 4xx/5xx, show the server's `error` field in a red banner.

Symbol card mirrors `ClaimsList`'s shape: file path header, list of
symbols with kind badges (function / class / method / import /
config_schema) and line ranges. README + lockfile content shows in a
separate collapsible `<ConfigsPanel />` to keep v1 scope tight.

## 3. What we explicitly do NOT do in Phase 3

- ❌ Private repositories (token support is there but Phase 3 only
  handles public repos)
- ❌ GitLab / Bitbucket / self-hosted (HTTPS clone works in theory
  but URL validation + rate-limit handling are GitHub-specific in v1)
- ❌ C / C++ / PHP / Ruby / Kotlin / Swift / Scala (grammars deferred)
- ❌ Authorship / git-log / blame history (Phase 3 walks the working
  tree only)
- ❌ Type-resolved call graphs (name-based only — language-server
  integration is a much larger lift)
- ❌ Repo diff (we don't track what the user changed locally vs origin)
- ❌ BackgroundTasks for clone (still sync — Phase 5+ revisits)
- ❌ Embedding generation (Phase 4)
- ❌ Retrieval / claim-to-symbol mapping (Phase 5)
- ❌ Verdict assignment (Phase 6)
- ❌ Secret redaction in source (Phase 8 — README + lockfiles are read
  in Phase 3, so the safety gate is **added in Phase 8**, not deferred
  silently)

## 4. CI considerations

- **tree-sitter + 6 grammars add ~25 MB** to the backend image. Each
  grammar is a small C-extension; total wheel footprint is moderate.
- **`git` is already installed** in `deployments/Dockerfile.api:17` with
  a comment that anticipates Phase 3 — no Dockerfile change.
- **Unit tests are CI-safe.** `tree-sitter` queries use synthetic code
  snippets; the cloner mocks `subprocess.run`; config tests write temp
  files. PyYAML is parsed in-process.
- **The real-`git-clone` integration test is opt-in:**
  `@pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"), reason="needs
  token")` so CI collects and skips it (no secret needed in CI). The
  test target is a tiny public repo, e.g. `octocat/Hello-World`.
- **No new CI job.** Reuse Phase 1's 4-job split (`scaffold-smoke` /
  `backend-lint` / `backend-test` / `frontend-build`). The new test
  files join the existing unit suite.

## 5. Acceptance — the command list

```bash
# 1. Local static checks (no LLM key, no GitHub token required)
cd apps/api && pip install -e ".[dev]"
ruff check apps/api && mypy apps/api
cd apps/web && npm ci && npm run lint && npm test && npm run build

# 2. Local unit + integration suite (integration skipped without token)
make test                  # unit suite passes (parser + cloner + configs
                           #   + lockfiles + endpoints)
GITHUB_TOKEN=ghp_... make test-integration   # adds the real-clone test

# 3. Migration
make migrate               # applies 0001_init + 0002_papers + 0003_repositories
docker exec -it claim2code-postgres psql -U claim2code -d claim2code \
  -c "\d repositories"
# → shows repository_id, analysis_id (FK), url, owner, name, …

# 4. End-to-end
cp .env.example .env       # fill OPENAI_API_KEY + POSTGRES_PASSWORD + (optional) GITHUB_TOKEN
make dev
ANALYSIS=$(curl -sX POST -F "file=@samples/attention.pdf" \
  http://localhost:8000/papers | jq -r .analysis_id)

# Bind a repo to the analysis.
curl -sX POST -H 'content-type: application/json' \
  -d "{\"url\":\"https://github.com/octocat/Hello-World\",\"analysis_id\":\"$ANALYSIS\"}" \
  http://localhost:8000/repositories
# → {"repository_id":"…","n_files":2,"n_symbols":3,…}

# Fetch symbols.
curl -s http://localhost:8000/repositories/<repo_id>/symbols | jq '.symbols | length'
# → 3

# Open http://localhost:5173 in a browser → upload page + claims list +
#   repo section (three stacked sections, one screen)
```

A green run of all four blocks is the Phase 3 done-state.

---

## Why these decisions specifically

- **tree-sitter over Python `ast`**: ML-paper repos are heterogeneous
  (research code often half-finished). tree-sitter survives malformed
  code and unifies Python / JS / TS / Java / Go / Rust behind one
  normalised shape — Phase 4-6 don't need to know the source language.
  The Python-only `ast` pass for argparse / dataclasses is the
  exception; it's narrow enough that one extra parse is cheap.
- **git clone over GitHub REST only**: AST walk needs every file
  locally; streaming the tarball and re-parsing per call is wasteful.
  Disk is cheap; `data/repos/<owner>__<repo>/` is trivial to back up
  with `tar`. The HEAD pre-flight is the only GitHub-API call we make
  per repo, so the 60/h unauthenticated limit is rarely a problem; with
  a token, the limit becomes a non-issue.
- **Multi-language from v1**: ML demo code is often JS / TS (HuggingFace
  Spaces, Gradio). Adding grammars incrementally later means re-running
  the index on every repo — much more expensive than adding them now.
- **`code_symbols` as a separate table**: a 50k-symbol repo doesn't
  fit the JSONB-in-`repositories` pattern. A proper table lets
  Phase 4-5 index / query efficiently and keeps `repositories` small.
- **Lockfile + config tables from v1**: Phase 6's verdict logic needs
  "what did the code actually declare?" — without these rows we'd
  re-parse on every verification, which is wasteful and risks drift
  between parsed-time and verify-time.
- **Name-based call graph in v1**: a type-resolved graph needs a
  language server (Pyright / TS server) per language — 100MB+ per
  install and slow. Name-based resolution gives Phase 6 "this method
  probably calls that method" which is enough for v1; we revisit in
  Phase 4 if claim-to-symbol retrieval needs more precision.
- **No router in Phase 3 frontend**: still one screen, just longer.
  Phase 4 (embedding visualisation) is when a router finally pays for
  itself.
- **Sync workflow, no BackgroundTasks**: same argument as Phase 2 —
  no concurrent load yet. Background adds worker pool + idempotency +
  status polling without a buyer. `repository_status` enum is
  tracked anyway, so a future async layer resumes mid-workflow without
  schema migration.