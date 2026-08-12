#!/usr/bin/env python3
"""Retrieval evaluator — Recall@1, Recall@5, MRR for each baseline.

Run with ``make eval-retrieval`` from the repo root. Loads
``evaluation/annotations/*.json`` (3 hand-labelled paper/repo
pairs), runs each baseline (embedding_only, lexical_only,
hybrid_claim2code) per dataset, and emits:

  - ``evaluation/results.json`` — structured per-(dataset, baseline)
    metrics for downstream consumers.
  - ``evaluation/report.md`` — a human-readable Markdown table
    mirroring the JSON.

The acceptance criterion for Phase 5 (per the design doc §5)
is ``hybrid.R@5 > embedding.R@5``; the report's *Aggregate*
table surfaces this comparison at a glance.

What this does NOT do
---------------------

  - **No reranking study.** The design doc §5 calls for
    measuring per-leg contribution; we leave that to
    ``hybrid_claim2code.run(weights=...)`` and a follow-up
    Phase 5.5 study.
  - **No verifier numbers.** Phase 6 owns
    ``evaluate_verification.py``; this script only measures
    retrieval quality.
  - **No synthetic pairs.** Real papers + real repos per the
    roadmap.

Run requirements
----------------

  - A live Postgres + pgvector reachable via the project's
    standard env (DATABASE_URL).
  - The OpenAI embedding client configured (so the semantic
    leg doesn't fail). Lexical-only doesn't need OpenAI but
    lives in the same comparison.
  - The 3 paper/repo pairs must already be cloned and
    indexed under their respective ``repository_id``s. The
    ``repository_id`` per pair is passed via the
    ``CLAIM2CODE_EVAL_REPO_IDS`` env var as a JSON dict (see
    :func:`_load_repo_ids`).
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

# Ensure ``apps/api`` is on sys.path before ``claim2code_api``
# resolves. Must happen before we import the baselines below.
_THIS = Path(__file__).resolve()
_APPS_API = _THIS.parents[1] / "apps" / "api"
if str(_APPS_API) not in sys.path:
    sys.path.insert(0, str(_APPS_API))

from claim2code_api.db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Baselines are imported AFTER sys.path setup.
from evaluation.baselines import (
    embedding_only,
    hybrid_claim2code,
    lexical_only,
)

logger = logging.getLogger("claim2code.eval.retrieval")

# Path constants. ``__file__`` is ``evaluation/evaluate_retrieval.py``.
EVAL_DIR = _THIS.parent
ANNOTATIONS_DIR = EVAL_DIR / "annotations"
RESULTS_JSON = EVAL_DIR / "results.json"
REPORT_MD = EVAL_DIR / "report.md"

BASELINES = [
    embedding_only,
    lexical_only,
    hybrid_claim2code,
]


# ============================================================
# Result types
# ============================================================


@dataclass(frozen=True)
class ClaimResult:
    """Per-claim retrieval result for one (dataset, baseline)."""

    claim_id: str
    relevant_symbol_ids: set[str] = field(default_factory=set)
    retrieved_symbol_ids: list[str] = field(default_factory=list)

    @property
    def recall_at_1(self) -> float:
        return 1.0 if self.relevant_symbol_ids & {self.retrieved_symbol_ids[0]} else 0.0

    @property
    def recall_at_5(self) -> float:
        return 1.0 if self.relevant_symbol_ids & set(self.retrieved_symbol_ids[:5]) else 0.0

    @property
    def mrr(self) -> float:
        for i, sid in enumerate(self.retrieved_symbol_ids, start=1):
            if sid in self.relevant_symbol_ids:
                return 1.0 / i
        return 0.0


@dataclass(frozen=True)
class BaselineMetrics:
    """Aggregate metrics for one (dataset, baseline)."""

    baseline: str
    dataset: str
    n_claims: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    error: str | None = None


# ============================================================
# Setup helpers
# ============================================================


def _build_session_factory() -> sessionmaker[Session]:
    """Build a SQLAlchemy session factory from ``DATABASE_URL``.

    Falls back to ``sqlite:///./data.db`` so the script can
    at least import + run sanity checks in environments where
    Postgres isn't available — but real evaluation requires
    the live database (the eval is opt-in via
    ``make eval-retrieval``).
    """
    import os

    url = os.environ.get("DATABASE_URL", "sqlite:///./data.db")
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _load_annotations() -> list[dict[str, Any]]:
    """Load every ``*.json`` annotation file. Sorted for determinism."""
    out: list[dict[str, Any]] = []
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        data["_path"] = str(path)
        out.append(data)
    return out


def _load_repo_ids() -> dict[str, str]:
    """Return ``{pair_id: repository_id}`` from env.

    Set via ``CLAIM2CODE_EVAL_REPO_IDS='{"attention_is_all_you_need": "uuid", ...}'``.
    Missing pairs raise at runtime (not at import) so the
    script can be inspected without a DB.
    """
    import os

    raw = os.environ.get("CLAIM2CODE_EVAL_REPO_IDS", "{}")
    return json.loads(raw)


def _resolve_symbol_ids(
    db: Session,
    repository_id: UUID,
    relevant_symbols: list[dict[str, str]],
) -> set[str]:
    """Map ``{file_path, name, kind}`` triplets to symbol_ids.

    A claim is considered "retrieved" if ANY of its relevant
    symbols appears in the top-k. If multiple relevant symbols
    are listed for a claim, the union is used.
    """
    ids: set[str] = set()
    for rs in relevant_symbols:
        # ``file_path`` may be relative or absolute; the
        # stored ``code_symbols.file_path`` is whatever the
        # parser emitted, so we don't normalise.
        rows = (
            db.query(Base.metadata.tables["code_symbols"])
            .filter_by(
                repository_id=str(repository_id),
                file_path=rs["file_path"],
                name=rs["name"],
                kind=rs["kind"],
            )
            .all()
        )
        for row in rows:
            ids.add(str(row.symbol_id))
    return ids


# ============================================================
# Evaluation loop
# ============================================================


def evaluate_baseline_on_dataset(
    *,
    db: Session,
    repository_id: UUID,
    annotation: dict[str, Any],
    baseline_mod: Any,
    top_k: int = 5,
) -> BaselineMetrics:
    """Run ``baseline_mod.run`` on every claim in ``annotation``."""
    name = baseline_mod.NAME
    dataset = annotation["pair_id"]
    per_claim: list[ClaimResult] = []

    try:
        for claim in annotation["claims"]:
            relevant = _resolve_symbol_ids(
                db,
                repository_id,
                claim.get("relevant_symbols", []),
            )
            try:
                rows = baseline_mod.run(
                    db,
                    repository_id,
                    claim["text"],
                    top_k=top_k,
                )
                retrieved_ids = [str(sid) for sid, _score in rows]
            except Exception:
                logger.exception(
                    "baseline=%s claim=%s failed",
                    name,
                    claim.get("claim_id"),
                )
                retrieved_ids = []

            per_claim.append(
                ClaimResult(
                    claim_id=claim.get("claim_id", "?"),
                    relevant_symbol_ids=relevant,
                    retrieved_symbol_ids=retrieved_ids,
                )
            )
    except Exception as exc:  # noqa: BLE001 — surface in metrics
        return BaselineMetrics(
            baseline=name,
            dataset=dataset,
            n_claims=len(per_claim),
            recall_at_1=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            error=str(exc),
        )

    n = len(per_claim)
    if n == 0:
        return BaselineMetrics(
            baseline=name,
            dataset=dataset,
            n_claims=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            mrr=0.0,
        )

    return BaselineMetrics(
        baseline=name,
        dataset=dataset,
        n_claims=n,
        recall_at_1=statistics.fmean(c.recall_at_1 for c in per_claim),
        recall_at_5=statistics.fmean(c.recall_at_5 for c in per_claim),
        mrr=statistics.fmean(c.mrr for c in per_claim),
    )


def evaluate_all(
    *,
    repo_ids: dict[str, str],
    top_k: int = 5,
) -> list[BaselineMetrics]:
    """Run all baselines × all annotations, return flat list of metrics."""
    annotations = _load_annotations()
    SessionLocal = _build_session_factory()
    out: list[BaselineMetrics] = []

    for annotation in annotations:
        pair_id = annotation["pair_id"]
        repo_id_str = repo_ids.get(pair_id)
        if not repo_id_str:
            logger.warning("skipping pair_id=%s (no repo id in CLAIM2CODE_EVAL_REPO_IDS)", pair_id)
            continue
        repo_id = UUID(repo_id_str)
        with SessionLocal() as db:
            for baseline in BASELINES:
                m = evaluate_baseline_on_dataset(
                    db=db,
                    repository_id=repo_id,
                    annotation=annotation,
                    baseline_mod=baseline,
                    top_k=top_k,
                )
                out.append(m)
                logger.info("%s / %s  R@1=%.3f  R@5=%.3f  MRR=%.3f",
                            pair_id, m.baseline, m.recall_at_1, m.recall_at_5, m.mrr)
    return out


# ============================================================
# Output rendering
# ============================================================


def _render_results_json(metrics: list[BaselineMetrics]) -> dict[str, Any]:
    """Aggregate by baseline (across datasets)."""
    by_baseline: dict[str, list[BaselineMetrics]] = {}
    for m in metrics:
        by_baseline.setdefault(m.baseline, []).append(m)

    aggregate: dict[str, dict[str, float]] = {}
    for baseline, ms in by_baseline.items():
        aggregate[baseline] = {
            "avg_recall_at_1": statistics.fmean(m.recall_at_1 for m in ms),
            "avg_recall_at_5": statistics.fmean(m.recall_at_5 for m in ms),
            "avg_mrr": statistics.fmean(m.mrr for m in ms),
        }

    return {
        "per_dataset": [
            {
                "dataset": m.dataset,
                "baseline": m.baseline,
                "n_claims": m.n_claims,
                "recall_at_1": m.recall_at_1,
                "recall_at_5": m.recall_at_5,
                "mrr": m.mrr,
                "error": m.error,
            }
            for m in metrics
        ],
        "aggregate_by_baseline": aggregate,
    }


def _render_report_md(metrics: list[BaselineMetrics]) -> str:
    """Markdown table report matching the design doc §5."""
    by_dataset: dict[str, dict[str, BaselineMetrics]] = {}
    for m in metrics:
        by_dataset.setdefault(m.dataset, {})[m.baseline] = m

    lines: list[str] = ["# Retrieval Evaluation Report", ""]

    for dataset, by_baseline in sorted(by_dataset.items()):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Baseline          | Recall@1 | Recall@5 | MRR     |")
        lines.append("|-------------------|----------|----------|---------|")
        for baseline in [b.NAME for b in BASELINES]:
            m = by_baseline.get(baseline)
            if m is None:
                lines.append(f"| {baseline:<17} | (no data) | (no data) | (no data) |")
                continue
            lines.append(
                f"| {baseline:<17} "
                f"| {m.recall_at_1:<8.2f} "
                f"| {m.recall_at_5:<8.2f} "
                f"| {m.mrr:<7.3f} |"
            )
        lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Baseline          | Avg R@1  | Avg R@5  | Avg MRR  |")
    lines.append("|-------------------|----------|----------|----------|")
    by_baseline: dict[str, list[BaselineMetrics]] = {}
    for m in metrics:
        by_baseline.setdefault(m.baseline, []).append(m)
    for baseline in [b.NAME for b in BASELINES]:
        ms = by_baseline.get(baseline, [])
        if not ms:
            lines.append(f"| {baseline:<17} | (no data) | (no data) | (no data) |")
            continue
        r1 = statistics.fmean(m.recall_at_1 for m in ms)
        r5 = statistics.fmean(m.recall_at_5 for m in ms)
        mrr = statistics.fmean(m.mrr for m in ms)
        lines.append(
            f"| {baseline:<17} | {r1:<8.2f} | {r5:<8.2f} | {mrr:<8.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Entry point. Returns process exit code (0 success)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    repo_ids = _load_repo_ids()
    if not repo_ids:
        logger.error(
            "CLAIM2CODE_EVAL_REPO_IDS is empty; set it to a JSON dict "
            "mapping pair_id → repository_id, e.g. "
            "'{\"attention_is_all_you_need\": \"<uuid>\", ...}'",
        )
        return 2

    metrics = evaluate_all(repo_ids=repo_ids)
    RESULTS_JSON.write_text(json.dumps(_render_results_json(metrics), indent=2))
    REPORT_MD.write_text(_render_report_md(metrics))
    logger.info("wrote %s and %s", RESULTS_JSON, REPORT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())