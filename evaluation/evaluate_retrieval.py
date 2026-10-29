#!/usr/bin/env python3
"""Retrieval evaluator — Recall@1, Recall@5, MRR for each baseline.

Run with ``make eval-retrieval`` from the repo root. Loads
``evaluation/annotations/*.json`` (3 hand-labelled paper/repo
pairs), runs each baseline (embedding_only, lexical_only,
hybrid_claim2code) per dataset, and emits:

  - ``evaluation/results/retrieval.json`` — structured
    per-(dataset, baseline) metrics. ``evaluation/report.py``
    folds it into ``evaluation/results.json`` + ``report.md``.

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
  - The paper/repo pairs must already be cloned and indexed.
    Each pair's ``repository_id`` is looked up by its annotated
    repo URL; ``CLAIM2CODE_EVAL_REPO_IDS`` (a JSON dict) overrides
    the lookup (see :func:`resolve_repo_ids`). Pairs whose repo is
    not indexed are skipped.
"""
# Imports below bootstrap the repository-root CLI entry point.
# ruff: noqa: E402
from __future__ import annotations

import argparse
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
_REPO_ROOT = _THIS.parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "apps" / "api"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from claim2code_api.db.models import Base

# Baselines are imported AFTER sys.path setup.
from evaluation.baselines import (
    embedding_only,
    hybrid_claim2code,
    lexical_only,
)
from evaluation.provenance import capture, finish, repository_snapshot, require_frozen

logger = logging.getLogger("claim2code.eval.retrieval")

# Path constants. ``__file__`` is ``evaluation/evaluate_retrieval.py``.
EVAL_DIR = _THIS.parent
ANNOTATIONS_DIR = EVAL_DIR / "annotations"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "retrieval.json"

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
    retrieved_scores: tuple[float, ...] = ()
    # One group per annotated symbol. A class and its methods are
    # alternative locations for that class, not extra relevant targets.
    # An empty group is unresolved and remains in the denominator.
    relevant_symbol_groups: tuple[frozenset[str], ...] | None = None
    relevant_symbols: tuple[dict[str, str], ...] = ()
    error: str | None = None

    def recall_at(self, k: int) -> float:
        groups = self.relevant_symbol_groups
        if groups is None:
            groups = tuple(frozenset({sid}) for sid in sorted(self.relevant_symbol_ids))
        if not groups:
            return 0.0
        found = set(self.retrieved_symbol_ids[:k])
        return sum(bool(group & found) for group in groups) / len(groups)

    @property
    def recall_at_1(self) -> float:
        return self.recall_at(1)

    @property
    def recall_at_5(self) -> float:
        return self.recall_at(5)

    @property
    def hit_at_1(self) -> float:
        return float(bool(self.relevant_symbol_ids & set(self.retrieved_symbol_ids[:1])))

    @property
    def hit_at_5(self) -> float:
        return float(bool(self.relevant_symbol_ids & set(self.retrieved_symbol_ids[:5])))

    @property
    def unresolved_symbols(self) -> list[dict[str, str]]:
        if self.relevant_symbol_groups is None:
            return []
        return [rs for rs, group in zip(self.relevant_symbols, self.relevant_symbol_groups, strict=True)
                if not group]

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
    hit_at_1: float | None = None
    hit_at_5: float | None = None
    claims: tuple[ClaimResult, ...] = ()


# ============================================================
# Setup helpers
# ============================================================


def _build_session_factory() -> sessionmaker[Session]:
    """Build a SQLAlchemy session factory.

    ``DATABASE_URL`` in the environment wins; otherwise the app's own
    settings (``apps/api/.env``) pick the database, as for the API.
    """
    import os

    url = os.environ.get("DATABASE_URL")
    if url is None:
        from claim2code_api.db.session import get_session_local

        return get_session_local()
    engine = create_engine(url, future=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _load_annotations(annotations_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load every ``*.json`` annotation file. Sorted for determinism."""
    out: list[dict[str, Any]] = []
    directory = ANNOTATIONS_DIR if annotations_dir is None else annotations_dir
    for path in sorted(directory.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        data["_path"] = str(path)
        out.append(data)
    return out


def _load_repo_ids() -> dict[str, str]:
    """Return the ``{pair_id: repository_id}`` override from env.

    Set via ``CLAIM2CODE_EVAL_REPO_IDS='{"attention_is_all_you_need": "uuid", ...}'``.
    """
    import os

    raw = os.environ.get("CLAIM2CODE_EVAL_REPO_IDS", "{}")
    return json.loads(raw)


def resolve_repo_ids(
    db: Session,
    annotations: list[dict[str, Any]],
) -> dict[str, str]:
    """Map each pair to an indexed ``repository_id``.

    The env override wins; otherwise the pair's ``repo.url`` is matched
    against ``repositories.url`` (ignoring a trailing ``/`` or ``.git``).
    Pairs with no indexed repository are left out.
    """
    overrides = _load_repo_ids()
    table = Base.metadata.tables["repositories"]
    indexed = {
        _normalise_url(row.url): str(row.repository_id)
        for row in db.query(table.c.url, table.c.repository_id)
    }
    out: dict[str, str] = {}
    for a in annotations:
        pair_id = a["pair_id"]
        repo_id = overrides.get(pair_id) or indexed.get(_normalise_url(a["repo"]["url"]))
        if repo_id:
            out[pair_id] = repo_id
    return out


def _normalise_url(url: str) -> str:
    url = url.strip().rstrip("/")
    return url.removesuffix(".git").lower()


def _resolve_symbol_ids(
    db: Session,
    repository_id: UUID,
    relevant_symbols: list[dict[str, str]],
) -> set[str]:
    """Map ``{file_path, name, kind}`` triplets to symbol_ids.

    A claim is considered "retrieved" if ANY of its relevant
    symbols appears in the top-k. If multiple relevant symbols
    are listed for a claim, the union is used.

    A class also stands for its methods: an annotation naming
    ``Linear`` is satisfied by retrieving ``Linear.forward``, since
    the class body lives in its methods.
    """
    return set().union(*_resolve_symbol_groups(db, repository_id, relevant_symbols))


def _resolve_symbol_groups(
    db: Session,
    repository_id: UUID,
    relevant_symbols: list[dict[str, str]],
) -> tuple[frozenset[str], ...]:
    """Resolve each annotation separately, retaining unresolved targets."""
    table = Base.metadata.tables["code_symbols"]
    groups: list[frozenset[str]] = []
    for rs in relevant_symbols:
        ids: set[str] = set()
        # ``file_path`` may be relative or absolute; the
        # stored ``code_symbols.file_path`` is whatever the
        # parser emitted, so we don't normalise.
        query = (
            db.query(table)
            .filter_by(
                repository_id=str(repository_id),
                file_path=rs["file_path"],
                name=rs["name"],
                kind=rs["kind"],
            )
        )
        if rs.get("parent_name"):
            parents = db.query(table.c.symbol_id).filter_by(
                repository_id=str(repository_id), file_path=rs["file_path"],
                name=rs["parent_name"], kind="class",
            ).all()
            query = query.filter(table.c.parent_symbol_id.in_([row.symbol_id for row in parents]))
        rows = query.all()
        for row in rows:
            ids.add(str(row.symbol_id))
            if rs["kind"] == "class":
                children = db.query(table.c.symbol_id).filter(
                    table.c.parent_symbol_id == row.symbol_id,
                )
                ids.update(str(child.symbol_id) for child in children)
        groups.append(frozenset(ids))
    return tuple(groups)


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
            relevant_symbols = claim.get("relevant_symbols", [])
            groups = _resolve_symbol_groups(
                db,
                repository_id,
                relevant_symbols,
            )
            relevant = set().union(*groups)
            error = None
            try:
                rows = baseline_mod.run(
                    db,
                    repository_id,
                    claim["text"],
                    top_k=top_k,
                )
                retrieved_ids = [str(sid) for sid, _score in rows]
                retrieved_scores = tuple(float(score) for _sid, score in rows)
            except Exception as exc:
                logger.exception(
                    "baseline=%s claim=%s failed",
                    name,
                    claim.get("claim_id"),
                )
                retrieved_ids = []
                retrieved_scores = ()
                error = f"{type(exc).__name__}: {exc}"

            per_claim.append(
                ClaimResult(
                    claim_id=claim.get("claim_id", "?"),
                    relevant_symbol_ids=relevant,
                    retrieved_symbol_ids=retrieved_ids,
                    retrieved_scores=retrieved_scores,
                    relevant_symbol_groups=groups,
                    relevant_symbols=tuple(relevant_symbols),
                    error=error,
                )
            )
    except Exception as exc:  # noqa: BLE001 — surface in metrics
        return BaselineMetrics(
            baseline=name,
            dataset=dataset,
            n_claims=len(annotation["claims"]),
            recall_at_1=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            hit_at_1=0.0,
            hit_at_5=0.0,
            error=str(exc),
            claims=tuple(per_claim),
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
            hit_at_1=0.0,
            hit_at_5=0.0,
        )

    return BaselineMetrics(
        baseline=name,
        dataset=dataset,
        n_claims=n,
        recall_at_1=statistics.fmean(c.recall_at_1 for c in per_claim),
        recall_at_5=statistics.fmean(c.recall_at_5 for c in per_claim),
        mrr=statistics.fmean(c.mrr for c in per_claim),
        hit_at_1=statistics.fmean(c.hit_at_1 for c in per_claim),
        hit_at_5=statistics.fmean(c.hit_at_5 for c in per_claim),
        claims=tuple(per_claim),
    )


def evaluate_all(
    *,
    repo_ids: dict[str, str] | None = None,
    top_k: int = 5,
    annotations: list[dict[str, Any]] | None = None,
) -> list[BaselineMetrics]:
    """Run all baselines × all annotations, return flat list of metrics."""
    annotations = _load_annotations() if annotations is None else annotations
    session_factory = _build_session_factory()
    if repo_ids is None:
        with session_factory() as db:
            repo_ids = resolve_repo_ids(db, annotations)
    out: list[BaselineMetrics] = []

    for annotation in annotations:
        pair_id = annotation["pair_id"]
        repo_id_str = repo_ids.get(pair_id)
        if not repo_id_str:
            logger.warning("skipping pair_id=%s (repository not indexed)", pair_id)
            continue
        repo_id = UUID(repo_id_str)
        with session_factory() as db:
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

    aggregate: dict[str, dict[str, float | None]] = {}
    for baseline, ms in by_baseline.items():
        aggregate[baseline] = {
            "avg_recall_at_1": statistics.fmean(m.recall_at_1 for m in ms),
            "avg_recall_at_5": statistics.fmean(m.recall_at_5 for m in ms),
            "avg_mrr": statistics.fmean(m.mrr for m in ms),
            "avg_hit_at_1": _optional_mean([m.hit_at_1 for m in ms]),
            "avg_hit_at_5": _optional_mean([m.hit_at_5 for m in ms]),
        }

    return {
        "schema_version": 2,
        "metric_semantics": "symbol_recall_v2",
        "metric_definitions": {
            "recall_at_k": "fraction of annotated relevant symbols retrieved in the first k results",
            "hit_at_k": "whether any annotated relevant symbol appears in the first k results",
            "class_symbols": "a class and its methods are alternatives for one annotated class target",
            "unresolved_symbols": "unresolved annotated targets remain in the recall denominator",
            "aggregation": "mean over claims per dataset, then unweighted mean over datasets",
        },
        "per_dataset": [
            {
                "dataset": m.dataset,
                "baseline": m.baseline,
                "n_claims": m.n_claims,
                "recall_at_1": m.recall_at_1,
                "recall_at_5": m.recall_at_5,
                "mrr": m.mrr,
                "hit_at_1": m.hit_at_1,
                "hit_at_5": m.hit_at_5,
                "n_unresolved_symbols": sum(len(c.unresolved_symbols) for c in m.claims),
                "n_claim_errors": sum(c.error is not None for c in m.claims),
                "error": m.error,
            }
            for m in metrics
        ],
        "aggregate_by_baseline": aggregate,
        "per_claim": [
            {
                "dataset": m.dataset,
                "baseline": m.baseline,
                "claim_id": c.claim_id,
                "relevant_symbols": list(c.relevant_symbols),
                "resolved_symbol_groups": [sorted(group) for group in c.relevant_symbol_groups or ()],
                "relevant_symbol_ids": sorted(c.relevant_symbol_ids),
                "unresolved_symbols": c.unresolved_symbols,
                "retrieved_symbol_ids": c.retrieved_symbol_ids,
                "retrieved_scores": list(c.retrieved_scores),
                "recall_at_1": c.recall_at_1,
                "recall_at_5": c.recall_at_5,
                "hit_at_1": c.hit_at_1,
                "hit_at_5": c.hit_at_5,
                "mrr": c.mrr,
                "error": c.error,
            }
            for m in metrics
            for c in m.claims
        ],
    }


def _optional_mean(values: list[float | None]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return statistics.fmean(v for v in values if v is not None)


def _display(value: float | None, places: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _render_report_md(
    metrics: list[BaselineMetrics], *, metric_semantics: str = "symbol_recall_v2",
) -> str:
    """Render metrics using their stored definition, including legacy hit rates."""
    by_dataset: dict[str, dict[str, BaselineMetrics]] = {}
    for m in metrics:
        by_dataset.setdefault(m.dataset, {})[m.baseline] = m

    lines: list[str] = ["# Retrieval Evaluation Report", ""]
    legacy = metric_semantics != "symbol_recall_v2"
    if legacy:
        lines += [
            "Archived `recall_at_1/5` values measure Hit@1/5 (any relevant symbol), not symbol recall.",
            "Symbol Recall@1/5 is unavailable from these aggregate-only artifacts; rerun the evaluator.",
            "",
        ]
    else:
        lines += [
            "Recall@k is the fraction of annotated relevant symbols found; Hit@k needs any one relevant symbol.",
            "Each annotated class counts once, accepting that class or a method as an alternative location.",
            "Unresolved annotated symbols remain in the recall denominator.",
            "",
        ]

    for dataset, by_baseline in sorted(by_dataset.items()):
        lines.append(f"## {dataset}")
        lines.append("")
        columns = "Hit@1 | Hit@5 | MRR" if legacy else "Recall@1 | Recall@5 | Hit@1 | Hit@5 | MRR"
        lines.append(f"| Baseline | {columns} |")
        lines.append("|---|" + "---|" * (3 if legacy else 5))
        for baseline in [b.NAME for b in BASELINES]:
            metric = by_baseline.get(baseline)
            if metric is None:
                lines.append(f"| {baseline} |" + " (no data) |" * (3 if legacy else 5))
                continue
            values = [_display(metric.recall_at_1), _display(metric.recall_at_5)]
            if not legacy:
                values += [_display(metric.hit_at_1), _display(metric.hit_at_5)]
            lines.append(f"| {baseline} | " + " | ".join([*values, _display(metric.mrr, 3)]) + " |")
        lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    columns = "Avg Hit@1 | Avg Hit@5 | Avg MRR" if legacy else "Avg R@1 | Avg R@5 | Avg Hit@1 | Avg Hit@5 | Avg MRR"
    lines.append(f"| Baseline | {columns} |")
    lines.append("|---|" + "---|" * (3 if legacy else 5))
    baseline_rows: dict[str, list[BaselineMetrics]] = {}
    for m in metrics:
        baseline_rows.setdefault(m.baseline, []).append(m)
    for baseline in [b.NAME for b in BASELINES]:
        ms = baseline_rows.get(baseline, [])
        if not ms:
            lines.append(f"| {baseline} |" + " (no data) |" * (3 if legacy else 5))
            continue
        r1 = statistics.fmean(m.recall_at_1 for m in ms)
        r5 = statistics.fmean(m.recall_at_5 for m in ms)
        mrr = statistics.fmean(m.mrr for m in ms)
        values = [_display(r1), _display(r5)]
        if not legacy:
            values += [_display(_optional_mean([m.hit_at_1 for m in ms])),
                       _display(_optional_mean([m.hit_at_5 for m in ms]))]
        lines.append(f"| {baseline} | " + " | ".join([*values, _display(mrr, 3)]) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--require-reproducible", action="store_true",
                        help="require frozen annotation/clone inputs; does not verify model weight revisions")
    parser.add_argument("--annotations-dir", type=Path,
                        help="annotation directory (default: evaluation/annotations)")
    parser.add_argument("--output-dir", type=Path,
                        help="directory for retrieval.json (default: evaluation/results)")
    parser.add_argument("--pairs", help="comma-separated annotation pair IDs")
    args = parser.parse_args(argv)
    annotations = _load_annotations() if args.annotations_dir is None else _load_annotations(args.annotations_dir)
    if args.pairs:
        selected = set(args.pairs.split(","))
        unknown = selected - {a["pair_id"] for a in annotations}
        if unknown:
            logger.error("unknown annotation pairs: %s", ", ".join(sorted(unknown)))
            return 2
        annotations = [a for a in annotations if a["pair_id"] in selected]
    if not annotations:
        logger.error("no annotations selected")
        return 2
    session_factory = _build_session_factory()
    try:
        with session_factory() as db:
            repo_ids = resolve_repo_ids(db, annotations)
            repositories = repository_snapshot(db, annotations, repo_ids)
        meta = capture(
            "retrieval", annotations=annotations, repositories=repositories,
            parameters={"top_k": 5, "baselines": [b.NAME for b in BASELINES],
                        "metric_semantics": "symbol_recall_v2"}, models=True,
        )
        if args.require_reproducible:
            require_frozen(meta)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    for issue in meta["reproducibility_issues"]:
        logger.warning("development evaluation: %s", issue)
    metrics = evaluate_all(repo_ids=repo_ids, annotations=annotations)
    if not metrics:
        logger.error("no annotated pair has an indexed repository")
        return 2
    output = RESULTS_JSON if args.output_dir is None else args.output_dir / "retrieval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        **_render_results_json(metrics), "provenance": finish(meta),
        "selected_pairs": [a["pair_id"] for a in annotations],
        "skipped_pairs": [a["pair_id"] for a in annotations if a["pair_id"] not in repo_ids],
    }, indent=2))
    logger.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
