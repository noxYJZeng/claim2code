# Claim2Code — Phase 9 Design Document

## Context

Phases 0–8 shipped: paper intelligence → repo intelligence → semantic indexing → hybrid retrieval → verification → related work → responsible-AI safety. Every LLM call goes through `LLMClient`, every workflow has a `db: Session` parameter, every endpoint is a FastAPI router.

**The gap Phase 9 fills:** when a user runs an analysis, we have **zero** production observability. No way to see which workflow stage is slow. No way to attribute cost (tokens) per call. No way to debug a failed claim by walking a trace tree. Phase 9 ships:

- **OpenTelemetry spans** on the 3 workflows + 3 LLM SDK call sites + 3 retriever sub-calls + safety gate — every workflow call = one root span with a populated child tree
- **Per-step metrics** for LLM tokens, retrieval latency (per leg), safety violation rate — feeds Phase 10's evaluation harness
- **Env-driven exporter** (console default; OTLP when `OTEL_EXPORTER_OTLP_ENDPOINT` set) — dev-friendly, production-ready, zero new infra in CI
- **Self-contained CI tests** via `InMemorySpanExporter` — mirrors the Phase 8 safety-test pattern

**Why a doc first:** observability is **infrastructure for Phase 10's evaluator** + future debugging. Wrong decisions (e.g. coupling to a specific exporter, missing per-leg attribution, no trace correlation) will be expensive to back out. Four live decisions need user sign-off (already answered in plan mode):
1. **Exporter routing**: console default + OTLP optional (env-driven). Zero infra in dev.
2. **Retriever span shape**: orchestrator + 3 child spans (semantic / lexical / structural). Per-leg latency attributable.
3. **Metric scope**: LLM + retrieval + safety (3 families). Focused on what Phase 10 needs.
4. **CI exporter**: in-memory exporter in tests (self-contained). No collector sidecar.

**Hard constraints from recon (Explore agent findings):**

- Settings already pre-stages 3 OTel fields at `app/config.py:126-129`:
  ```python
  otel_service_name: str = "claim2code"
  otel_exporter_otlp_endpoint: str = ""
  otel_exporter_console: bool = True
  ```
  Phase 9 wires code that reads them; **no new Settings fields required**.
- 3 workflows: `papers/workflow.py:79 run_paper_workflow`, `verification/workflow.py:83 run_verification_workflow`, `literature/workflow.py:85 run_related_work_workflow`. Each is keyword-only, `db: Session`, returns frozen dataclass. Each is the natural root-span anchor.
- 3 LLM SDK call sites at `llm/client.py:240/342/461` (extract_claims / verdict_for_claim / extract_comparison). `check_ready()` at line 524 is **NOT** traced (readiness probe is telemetry-exempt — keeps probe logs clean).
- Retriever chain: `hybrid_ranker.py:175` (semantic, pgvector), `:184` (lexical, tsvector), `:209` (structural, boost), then RRF fusion at line 211. Three sub-call sites = 3 child spans.
- Safety gate: `safety/gate.py:47 SafetyGate.wrap_messages`. Invoked inside the LLM SDK method *before* the OpenAI call — **same call stack**, so a child span inside `wrap_messages` automatically parents under the LLM span via OTel's `contextvars`-based propagation. No explicit parent needed.
- Verification triple-loop: `verification/workflow.py:126 _verify_one_claim` iterates per-claim → calls `hybrid_ranker` + `llm.verdict_for_claim`. Each per-claim invocation needs `claim_id` attribute on the span to disambiguate parallel-ish children in the trace tree.
- Literature per-paper loop: `literature/workflow.py:411 _compare_papers` iterates per paper → calls `llm.extract_comparison`. Same need for `paper_id` attribute.
- FastAPI app: `app/main.py:33 create_app`, CORS middleware at line 45. `FastAPIInstrumentor.instrument_app(app)` is idempotent — safe to call in `lifespan`. Decision: call inside `lifespan` to keep `create_app` clean.
- Pytest setup: `pyproject.toml:78-81` markers (`integration`, `safety`); `addopts = "-m 'not integration and not safety'"` at line 77. Phase 9 adds `observability` marker + updates addopts. New test dir `tests/observability/` with auto-mark conftest (mirror `tests/safety/conftest.py:22`).
- CI: `backend-safety` job at `.github/workflows/ci.yml:112-155` is the template. New `backend-observability` job mirrors it.
- `apps/api/claim2code_api/observability/__init__.py` exists as an **empty** package pre-staged in Phase 0. Natural home for `tracer.py`, `meter.py`, `setup.py` modules.

---

## Four live decisions (user-confirmed in plan-mode AskUserQuestion)

1. **Console default + OTLP optional (env-driven)**: zero infra in dev, production-ready. The 3 pre-staged Settings fields cover this exactly. Both exporters can be active simultaneously; we route to whichever is configured.

2. **Orchestrator + 3 child spans (semantic / lexical / structural)**: per-leg latency is attributable in the trace tree. Matches the 3 sub-call sites already in `hybrid_ranker.py`. One orchestrator span + 3 children = 4 spans per retriever call. Cost: O(4N) spans where N is the number of retrieval calls (1 per verification claim + 1 per related-work). Bounded; not a cardinality bomb.

3. **LLM + retrieval + safety (3 metric families)**: focused on what Phase 10's evaluator needs (cost = tokens; latency = LLM + retrieval; safety = violation rate). HTTP latency via `FastAPIInstrumentor` is implicit (auto-instrumented) and doesn't count as a Phase 9 "metric family" we maintain by hand.

4. **In-memory exporter in tests (self-contained)**: `InMemorySpanExporter` + `InMemoryMetricReader`. Mirrors the Phase 8 safety test pattern (opt-in marker, self-contained). No external collector sidecar in CI = fast + simple + no infra drift.

---

## §1. What ships

| Layer | Change |
|---|---|
| `apps/api/claim2code_api/observability/__init__.py` | Edit: module docstring + `__all__` re-exporting `setup_tracing`, `setup_metrics`, `get_tracer`, `get_meter` |
| `apps/api/claim2code_api/observability/setup.py` | New: `setup_tracing(settings)` + `setup_metrics(settings)` — wires `TracerProvider` + `MeterProvider` from env-driven Settings; called once at app startup |
| `apps/api/claim2code_api/observability/tracer.py` | New: `get_tracer(name)` lazy-init helper + thin `start_span` re-export |
| `apps/api/claim2code_api/observability/meter.py` | New: 4 metric instruments — `llm_tokens_total` (counter), `llm_request_duration_seconds` (histogram), `retriever_request_duration_seconds` (histogram, `leg` label), `safety_violations_total` (counter, `pattern` label) |
| `apps/api/claim2code_api/llm/client.py` | Edit: each of 3 SDK methods opens `with tracer.start_as_current_span("llm.{kind}", attributes={...})` around `chat.completions.create(...)`; on success records token counter + duration histogram; reads `response.usage.prompt_tokens/completion_tokens` for token attribution; `check_ready()` unchanged |
| `apps/api/claim2code_api/papers/workflow.py` | Edit: `run_paper_workflow` opens root span `paper_workflow.run` |
| `apps/api/claim2code_api/verification/workflow.py` | Edit: `run_verification_workflow` opens root span `verification_workflow.run`; per-claim loop opens child span `verification.claim_verify` with `claim_id` attribute |
| `apps/api/claim2code_api/literature/workflow.py` | Edit: `run_related_work_workflow` opens root span `related_work_workflow.run`; per-paper loop opens child span `literature.paper_compare` with `paper_id` attribute |
| `apps/api/claim2code_api/retrieval/hybrid_ranker.py` | Edit: `hybrid_ranker` opens orchestrator span `retrieval.hybrid_ranker`; each sub-call (semantic line 175, lexical line 184, structural line 209) opens child span with `leg` attribute |
| `apps/api/claim2code_api/safety/gate.py` | Edit: `SafetyGate.wrap_messages` opens child span `safety.gate_wrap` (auto-parented under the LLM span via contextvars) with `kind` + `violations.{secret,injection}_count` attributes |
| `apps/api/claim2code_api/app/main.py` | Edit: `lifespan` calls `setup_tracing(settings)` + `setup_metrics(settings)` at startup, shuts down at teardown; `FastAPIInstrumentor.instrument_app(app)` inside `lifespan` (idempotent) |
| `apps/api/pyproject.toml` | Edit: add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, `opentelemetry-instrumentation-fastapi` to `[project.dependencies]`; `addopts = "-m 'not integration and not safety and not observability'"`; add `observability` marker |
| `apps/api/tests/observability/__init__.py` | New: empty marker package |
| `apps/api/tests/observability/conftest.py` | New: `pytest_collection_modifyitems` auto-marks every test under `tests/observability/` with `pytest.mark.observability`; `captured_spans` + `captured_metrics` fixtures using `InMemorySpanExporter` + `InMemoryMetricReader` |
| `apps/api/tests/observability/test_tracing.py` | New: ~10 tests — root span shape (1 root + N children), per-claim `claim_id` attribute on verification children, per-paper `paper_id` attribute on literature children, safety child span auto-parented under LLM span |
| `apps/api/tests/observability/test_metrics.py` | New: ~6 tests — token counter increments on successful LLM call, latency histogram observes duration, retrieval `leg` label distinguishes semantic/lexical/structural, safety violation counter by pattern |
| `apps/api/tests/observability/test_exporters.py` | New: ~4 tests — console exporter active when `otel_exporter_console=True`, OTLP exporter added when `otel_exporter_otlp_endpoint` non-empty, both exporters can coexist, no exporters = no-op tracer (CI default) |
| `apps/api/tests/observability/test_e2e.py` | New: ~4 tests — full pipeline (`run_paper_workflow` + `run_verification_workflow` + `run_related_work_workflow`) produces one root span each with the expected child tree |
| `.github/workflows/ci.yml` | Edit: new `backend-observability` job mirrors `backend-safety` shape — Postgres service + `pytest -m observability -v` |
| `README.md` | Edit: phase roadmap table — Phase 9 row → ✅ |
| `evaluation/README.md` | Edit: Phase 9 status line; mention that Phase 10's evaluator reads metrics via in-memory exporter |
| `docs/architecture/phase-9-design.md` | New: this doc, mirrored from `phase-8-design.md` |

**No new Settings fields** (3 already pre-staged at `app/config.py:126-129`).

**No new router** — observability is a cross-cutting concern; doesn't add endpoints.

**No new migration** — metrics live in process; traces don't persist to DB.

**No production infra change** — empty `OTEL_EXPORTER_OTLP_ENDPOINT` = console-only (dev). Set it in production = OTLP added.

---

## §2. Architecture

### 2.1 Trace tree (one full analysis = one tree)

```
POST /analyses/{id}/verifications        [FastAPIInstrumentor auto-span]
└── verification_workflow.run             [root, root span]
    ├── verification.claim_verify         [claim_id="c1"]
    │   ├── retrieval.hybrid_ranker       [top_k=5, weights={...}]
    │   │   ├── retrieval.semantic        [leg="semantic"]
    │   │   ├── retrieval.lexical         [leg="lexical"]
    │   │   └── retrieval.structural      [leg="structural"]
    │   ├── llm.verdict                   [model="gpt-4o", candidate_count=5]
    │   │   └── safety.gate_wrap          [kind="verdict", violations={secret:0, injection:0}]
    │   └── verification.compose          [deterministic + LLM verdict merge]
    ├── verification.claim_verify         [claim_id="c2"]
    │   └── ... (same shape)
    └── ...
```

### 2.2 Wiring (per LLM call)

```python
# llm/client.py — extract_claims / verdict_for_claim / extract_comparison
gate_result = self._gate.wrap_messages(messages, kind="<kind>")
with tracer.start_as_current_span(
    f"llm.{kind}",
    attributes={
        "llm.method": kind,
        "llm.model": self._settings.openai_model_chat,
        "llm.message_count": len(gate_result.messages),
        "llm.violations.secret_count": sum(
            v.span_count for v in gate_result.violations if v.kind == "secret"
        ),
        "llm.violations.injection_count": sum(
            v.span_count for v in gate_result.violations if v.kind == "injection"
        ),
    },
) as span:
    start = time.monotonic()
    response = self._client.chat.completions.create(...)
    duration = time.monotonic() - start

    # Token attribution (OpenAI returns usage on response)
    usage = getattr(response, "usage", None)
    if usage:
        span.set_attribute("llm.tokens.prompt", usage.prompt_tokens)
        span.set_attribute("llm.tokens.completion", usage.completion_tokens)
        span.set_attribute("llm.tokens.total", usage.total_tokens)
        llm_tokens_total.add(
            usage.total_tokens,
            {"model": self._settings.openai_model_chat, "kind": kind},
        )
    llm_request_duration_seconds.record(
        duration,
        {"model": self._settings.openai_model_chat, "kind": kind},
    )
```

### 2.3 Safety gate as auto-parented child

```python
# safety/gate.py — wrap_messages
def wrap_messages(self, messages, *, kind):
    with tracer.start_as_current_span(
        "safety.gate_wrap",
        attributes={
            "safety.kind": kind,
            "safety.violations.secret_count": ...,
            "safety.violations.injection_count": ...,
        },
    ):
        # existing redact + detect logic unchanged
        ...
```

OTel's `contextvars`-based propagation means this becomes a child of whatever span is current on entry — usually the LLM span (because `wrap_messages` is called immediately before `chat.completions.create` inside the LLM SDK method). **Zero explicit parent wiring required.**

### 2.4 Retriever span shape

```python
# retrieval/hybrid_ranker.py
def hybrid_ranker(db, repository_id, claim, *, top_k=5, weights=None, embedding_client=None):
    with tracer.start_as_current_span(
        "retrieval.hybrid_ranker",
        attributes={"retrieval.top_k": top_k, "retrieval.weights": str(weights)},
    ) as span:
        leg_start = time.monotonic()
        sem = _semantic(...)  # line 175
        leg_duration.labels(leg="semantic").observe(time.monotonic() - leg_start)

        leg_start = time.monotonic()
        lex = _lexical(...)  # line 184
        leg_duration.labels(leg="lexical").observe(time.monotonic() - leg_start)

        leg_start = time.monotonic()
        structural_boost(...)  # line 209 (no return value)
        leg_duration.labels(leg="structural").observe(time.monotonic() - leg_start)

        # ... RRF fusion, sort, top-k truncation (line 211+)
        span.set_attribute("retrieval.result_count", len(top))
        return top
```

### 2.5 Metric instruments (3 families)

```python
# observability/meter.py
llm_tokens_total = meter.create_counter(
    "llm_tokens_total",
    description="Total LLM tokens consumed",
    unit="tokens",
)
llm_request_duration_seconds = meter.create_histogram(
    "llm_request_duration_seconds",
    description="LLM SDK call duration",
    unit="s",
)
retriever_request_duration_seconds = meter.create_histogram(
    "retriever_request_duration_seconds",
    description="Hybrid retriever leg duration",
    unit="s",
)
safety_violations_total = meter.create_counter(
    "safety_violations_total",
    description="Safety violations by pattern",
)
```

---

## §3. Why this design

### Why SDK seam wrap, not method entry or auto-instrumentation

Method entry (per-method decorator) requires 3× boilerplate; SDK seam is one `with start_as_current_span(...)` per call site, all inside the 3 SDK methods. Auto-instrumentation for `openai` (via `opentelemetry-instrumentation-openai`) exists but couples trace shape to library internals — breaks when OpenAI updates the SDK. Explicit spans give stable attribute names + clear ownership.

### Why orchestrator + 3 child spans in retriever

The orchestrator-only shape (per-leg counters as attributes) buries latency in numeric fields that need a query to surface. Child spans let you click into a slow retrieval in the trace UI and see exactly which leg was slow. Cardinality is bounded: 1 orchestrator + 3 children = 4 spans per retriever call.

### Why 3 metric families, not everything

Phase 10's evaluator needs cost (tokens), latency (LLM + retrieval), safety rate (violations). HTTP latency is auto-instrumented via `FastAPIInstrumentor` — not a family we maintain by hand. Adding more metrics families expands scope + maintenance; we can add later when concrete need arises.

### Why console default + OTLP optional

Local dev (95% of usage): zero infra. Production: set `OTEL_EXPORTER_OTLP_ENDPOINT=https://...` and OTLP is added alongside console. Both can coexist (just two exporters on the same `TracerProvider`). Zero "no exporter" mode = no-op tracer (CI default — tests use in-memory instead).

### Why self-contained in-memory exporter in CI

Same trade-off as Phase 8's safety tests: opt-in marker, fast, self-contained. A real `otel-collector` sidecar adds ~30s to CI runtime + drift surface. We can revisit when production deployment needs CI to verify exporter wiring (probably never — the exporter wiring is just `provider.add_span_processor(...)`, easily unit-tested).

### Why pre-staged Settings fields stay as-is

`otel_service_name`, `otel_exporter_otlp_endpoint`, `otel_exporter_console` are already declared at `app/config.py:126-129`. Adding a 4th (e.g. `otel_sampler_ratio`) breaks the "pre-staged in Phase 0" contract. We can add later when concrete need arises.

---

## §4. Test plan

| File | Test count | Coverage |
|---|---|---|
| `apps/api/tests/observability/test_tracing.py` | 10 | Root span shape (1 root + N children); per-claim `claim_id` attribute on verification children; per-paper `paper_id` attribute on literature children; safety child span auto-parented under LLM span; attributes correct (model, kind, candidate_count, etc.); trace_id stable across root + children |
| `apps/api/tests/observability/test_metrics.py` | 6 | Token counter increments on successful LLM call (`response.usage.total_tokens`); latency histogram observes duration (≥1 sample); retrieval `leg` label distinguishes semantic/lexical/structural; safety violation counter increments per pattern label; metrics survive multiple calls (cumulative counter) |
| `apps/api/tests/observability/test_exporters.py` | 4 | Console exporter added when `otel_exporter_console=True`; OTLP exporter added when `otel_exporter_otlp_endpoint` non-empty; both exporters coexist; no exporters = tracer is `NoOpTracer` (no spans emitted) |
| `apps/api/tests/observability/test_e2e.py` | 4 | Full pipeline: `run_paper_workflow` produces 1 root span + 1 LLM child + 1 safety grandchild; `run_verification_workflow` produces N per-claim sub-trees; `run_related_work_workflow` produces M per-paper sub-trees; HTTP request span (via FastAPIInstrumentor) wraps workflow root |
| `apps/api/tests/observability/conftest.py` | — | `pytest_collection_modifyitems` auto-marks all tests under `tests/observability/` with `pytest.mark.observability`; `captured_spans` fixture (InMemorySpanExporter) + `captured_metrics` fixture (InMemoryMetricReader) |

**Acceptance test 1** (in `test_e2e.py::test_acceptance_one_root_span_per_workflow`):
Given a `run_paper_workflow(...)` call with mocked SDK, assert:
- `captured_spans` contains exactly 1 span with name `paper_workflow.run`
- That span has children: `llm.extract_claims` + `safety.gate_wrap`
- The root span's `trace_id` matches the children's `trace_id`
- The root span has attribute `paper.filename`

**Acceptance test 2** (in `test_e2e.py::test_acceptance_per_claim_spans`):
Given `run_verification_workflow(...)` with 3 claims, assert:
- 1 root span (`verification_workflow.run`)
- 3 `verification.claim_verify` child spans, each with `claim_id` attribute matching the input
- Each per-claim span has its own `retrieval.hybrid_ranker` + `llm.verdict` + `safety.gate_wrap` subtree

**Acceptance test 3** (in `test_metrics.py::test_token_counter_attributes`):
Given a successful `LLMClient.extract_claims(...)` with `response.usage.total_tokens=500`, assert:
- `llm_tokens_total` counter has value 500 with labels `{model: "gpt-4o", kind: "extract_claims"}`

---

## §5. CI / infra

- `backend-test` job: **unchanged**.
- `backend-safety` job: **unchanged**.
- **New job** `backend-observability` mirrors `backend-safety` shape:
  ```yaml
  backend-observability:
    name: Backend observability (Phase 9)
    runs-on: ubuntu-latest
    services:
      postgres:  # same as backend-safety
    env:
      DATABASE_URL: postgresql+psycopg://claim2code:testpw@localhost:5432/claim2code_test
      OPENAI_API_KEY: sk-test-mock
      OTEL_SERVICE_NAME: claim2code-otel-test
      OTEL_EXPORTER_CONSOLE: "false"  # in-memory exporter used in tests instead
      OTEL_EXPORTER_OTLP_ENDPOINT: ""  # explicitly empty
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: apps/api/pyproject.toml
      - run: pip install -e ".[dev]"
      - run: pytest -m observability -v
  ```
- `backend-lint` job: picks up new source files via `ruff check .` + `mypy .`. OTel API uses `ContextManager` + `trace.Span` types which mypy handles out of the box.
- `frontend-build` / `scaffold-smoke`: no change.

---

## §6. What we explicitly do NOT do in Phase 9

- ❌ Auto-instrumentation for `openai` SDK (`opentelemetry-instrumentation-openai`) — couples trace shape to library internals; explicit spans give stable attribute names
- ❌ Auto-instrumentation for SQLAlchemy / httpx — out of scope; can add in a future phase when concrete need arises
- ❌ OTLP gRPC exporter — HTTP is simpler; switch if production needs it
- ❌ Sampler configuration — `parentbased_always_on` is fine for v1; add ratio sampling in production when trace volume grows
- ❌ Log correlation (`opentelemetry-instrumentation-logging`) — different concern; Phase 9 ships traces + metrics
- ❌ Trace storage in DB — traces are ephemeral; Phase 10's eval reads them from in-memory exporter during test runs
- ❌ Sidecar `otel-collector` in CI — self-contained in-memory exporter is sufficient
- ❌ Production deployment changes (Prometheus scrape config, Grafana dashboard JSON) — deployment concern, not Phase 9; `deployments/prometheus/` and `deployments/grafana/` already scaffolded in Phase 0
- ❌ New Settings fields — 3 already pre-staged at `app/config.py:126-129`
- ❌ Per-claim metric cardinality explosion — we use `claim_id` as a span attribute (bounded by request size), NOT as a metric label (unbounded)
- ❌ Distributed tracing across services — single-process app; no remote parent context to propagate
- ❌ Custom OpenTelemetry Collector exporter config files — standard OTLP HTTP works with any collector

---

## §7. Files Phase 9 will touch

### New (backend)
- `apps/api/claim2code_api/observability/__init__.py` — edit: docstring + `__all__`
- `apps/api/claim2code_api/observability/setup.py` — `setup_tracing` + `setup_metrics`
- `apps/api/claim2code_api/observability/tracer.py` — `get_tracer` lazy-init helper
- `apps/api/claim2code_api/observability/meter.py` — 4 metric instruments
- `apps/api/tests/observability/__init__.py` — empty package
- `apps/api/tests/observability/conftest.py` — auto-mark + InMemorySpanExporter + InMemoryMetricReader fixtures
- `apps/api/tests/observability/test_tracing.py`
- `apps/api/tests/observability/test_metrics.py`
- `apps/api/tests/observability/test_exporters.py`
- `apps/api/tests/observability/test_e2e.py`
- `docs/architecture/phase-9-design.md` — this doc, merged to main as `docs/phase-9-design` first

### Edited (backend)
- `apps/api/claim2code_api/llm/client.py` — 3 SDK methods wrap `chat.completions.create(...)` in tracer span; record token counter + duration histogram from `response.usage`; `check_ready()` unchanged
- `apps/api/claim2code_api/papers/workflow.py` — `run_paper_workflow` opens root span
- `apps/api/claim2code_api/verification/workflow.py` — `run_verification_workflow` opens root span; per-claim loop opens child span
- `apps/api/claim2code_api/literature/workflow.py` — `run_related_work_workflow` opens root span; per-paper loop opens child span
- `apps/api/claim2code_api/retrieval/hybrid_ranker.py` — `hybrid_ranker` opens orchestrator span + 3 child spans
- `apps/api/claim2code_api/safety/gate.py` — `wrap_messages` opens child span
- `apps/api/claim2code_api/app/main.py` — `lifespan` calls `setup_tracing` + `setup_metrics` + `FastAPIInstrumentor.instrument_app`

### Infra / docs
- `apps/api/pyproject.toml` — add 4 OTel deps; update `addopts`; add `observability` marker
- `.github/workflows/ci.yml` — new `backend-observability` job
- `README.md` — phase roadmap table: Phase 9 row → ✅
- `evaluation/README.md` — Phase 9 status line

### No change
- `apps/api/claim2code_api/app/config.py` — 3 OTel fields already pre-staged at lines 126-129
- `apps/api/claim2code_api/db/models.py` — no schema change
- `apps/api/migrations/versions/` — no new migration
- Frontend (`apps/web`) — Phase 9 ships backend only; Phase 10 owns UI surfacing of metrics
- `deployments/prometheus/` + `deployments/grafana/` — already scaffolded in Phase 0; production dashboards are deployment work, not Phase 9

---

## §8. Commit plan (after doc approval)

One branch `phase/9-observability`, 6 logical commits, `--no-ff` merge to main. The design doc itself ships first on a separate `docs/phase-9-design` branch (mirrors Phase 6/7/8 rhythm):

1. `feat(otel): observability skeleton + setup_tracing + setup_metrics (3 Settings fields already pre-staged)`
2. `feat(otel): instrument 3 workflows + 3 SDK call sites (root spans + per-claim/paper attributes)`
3. `feat(otel): instrument retriever orchestrator + 3 child spans (semantic/lexical/structural)`
4. `feat(otel): safety.gate_wrap child span + FastAPIInstrumentor in lifespan`
5. `feat(otel): 4 metric instruments (LLM tokens + duration, retrieval duration per leg, safety violations)`
6. `test(otel): tracing + metrics + exporters + e2e (24 tests); chore(ci): backend-observability job; docs: README + evaluation status`

(6 commits. Each commit is independently green: lint + default test suite. The `backend-observability` job only needs to pass on commits 5-6.)

---

## §9. Verification (of the design doc itself)

- ✓ Doc lives at `docs/architecture/phase-9-design.md`
- ✓ Sections §1-§8 present, in the order above
- ✓ Tone + structure mirror `phase-8-design.md`
- ✓ Every "we will X" has a corresponding file in the post-doc-approval list (no orphan decisions)
- ✓ Every "we will NOT do X" cites the reason + the phase that owns it if material
- ✓ Four live decisions each have a rationale + trade-off note
- ✓ Recon findings (3 workflows, 3 SDK call sites, retriever sub-calls at lines 175/184/209, safety gate inside same call stack, FastAPI app at `app/main.py:33`, pre-staged Settings fields) reflected in design
- ✓ Acceptance criteria (1 root span per workflow, per-claim/paper disambiguation, token attribution, 3 child spans per retriever call) testable as assertions
- ✓ Composes with Phase 8's safety gate (`safety.gate_wrap` is a child of the LLM span via contextvars — no explicit parent wiring)
- ✓ Pre-staged Settings fields reused — no schema change, no new migration
- ✓ Production path is env-driven (zero code change to enable OTLP)

After the doc is merged to main (separate doc-only commit on `docs/phase-9-design`), the implementation branch opens and work proceeds against the spec above.