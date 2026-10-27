#!/usr/bin/env python3
"""Verification evaluator — verdict accuracy, discrepancy detection, grounding.

Run with ``make eval-verification`` from the repo root. Every annotated
claim (``evaluation/annotations/*.json``) whose repository is indexed is
pushed through the production per-claim pipeline (hybrid retrieval at
``top_k=8`` → deterministic checkers → LLM verdict → composition), and
the verdict is scored against the claim's ``expected_verification``.

Metrics
-------

  - **status_accuracy** — exact match of the 4-way status.
  - **support_rate** — share of claims judged VERIFIED or
    PARTIALLY_VERIFIED. Every annotated claim is implemented in its
    repo, so anything else is a miss.
  - **discrepancy precision / recall** — claim level. A claim is
    labelled positive when ``expected_verification`` is
    PARTIALLY_VERIFIED (the annotation records a known paper/code
    mismatch in ``notes``) and predicted positive when the verdict is
    PARTIALLY_VERIFIED or CONTRADICTED.
  - **grounding_rate** — share of cited ``code_evidence`` entries whose
    quote occurs (whitespace-insensitively) in the cited file, i.e. the
    cited code is real and not fabricated.
  - **line_accuracy** — share of cited entries whose quote also sits at
    the cited lines (two lines of slack), i.e. the location is right.
  - **evidence_hit_rate** — share of supported verdicts that cite at
    least one annotated relevant symbol (or a line inside one).

Output
------

``evaluation/results/verification.json`` holds the metrics plus one
record per claim; ``evaluation/report.py`` renders it. Each claim takes
one LLM call (~1-2 min on a local model), so records are also appended
to ``evaluation/results/verification_claims.jsonl`` as they finish and
``--resume`` skips claims already there.

Usage::

    python evaluation/evaluate_verification.py [--pairs dpo,mae] [--resume | --regrade]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "apps" / "api"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluation.evaluate_retrieval import (
    _build_session_factory,
    _load_annotations,
    _resolve_symbol_ids,
    resolve_repo_ids,
)

logger = logging.getLogger("claim2code.eval.verification")

EVAL_DIR = _THIS.parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "verification.json"
CLAIMS_JSONL = RESULTS_DIR / "verification_claims.jsonl"

TOP_K = 8  # what the verification workflow uses in production
STATUSES = ("VERIFIED", "PARTIALLY_VERIFIED", "CONTRADICTED", "UNVERIFIED")
_SUPPORTED = {"VERIFIED", "PARTIALLY_VERIFIED"}
_FLAGGED = {"PARTIALLY_VERIFIED", "CONTRADICTED"}
_ELISION_RE = re.compile(r"\.{3,}|…")


# ============================================================
# Scoring (pure — unit-tested without a DB or LLM)
# ============================================================


def score(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-claim records into the metrics block.

    Each record carries ``expected``, ``predicted``, ``n_evidence``,
    ``n_grounded``, ``n_line_accurate``, ``evidence_hit`` and ``error``.
    """
    n = len(records)
    if n == 0:
        return {"n_claims": 0}
    exact = sum(r["predicted"] == r["expected"] for r in records)
    supported = [r for r in records if r["predicted"] in _SUPPORTED]
    tp = sum(r["expected"] == "PARTIALLY_VERIFIED" and r["predicted"] in _FLAGGED for r in records)
    pred_pos = sum(r["predicted"] in _FLAGGED for r in records)
    true_pos = sum(r["expected"] == "PARTIALLY_VERIFIED" for r in records)
    n_evidence = sum(r["n_evidence"] for r in records)
    confusion = {
        exp: {pred: 0 for pred in STATUSES} for exp in STATUSES
    }
    for r in records:
        confusion[r["expected"]][r["predicted"]] += 1
    return {
        "n_claims": n,
        "status_accuracy": exact / n,
        "support_rate": len(supported) / n,
        "discrepancy_precision": tp / pred_pos if pred_pos else None,
        "discrepancy_recall": tp / true_pos if true_pos else None,
        "n_expected_discrepancy": true_pos,
        "n_flagged": pred_pos,
        "grounding_rate": (
            sum(r["n_grounded"] for r in records) / n_evidence if n_evidence else None
        ),
        "line_accuracy": (
            sum(r.get("n_line_accurate", 0) for r in records) / n_evidence
            if n_evidence else None
        ),
        "n_code_evidence": n_evidence,
        "evidence_hit_rate": (
            sum(bool(r["evidence_hit"]) for r in supported) / len(supported)
            if supported else None
        ),
        "n_llm_errors": sum(r["error"] is not None for r in records),
        "predicted_counts": dict(Counter(r["predicted"] for r in records)),
        "confusion": {k: v for k, v in confusion.items() if any(v.values())},
    }


def quote_is_grounded(
    quote: str,
    file_text: str | None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> bool:
    """True when every fragment of ``quote`` occurs verbatim in ``file_text``.

    A quote is split into fragments at line breaks and at ``...``
    elisions, so a citation that skips a comment line or elides the
    middle of a long command still counts, while any altered code (a
    dropped comma, a renamed variable) does not. Whitespace is
    collapsed and shell/Python ``\\`` line continuations dropped on both
    sides, so re-indented or re-wrapped quotes match.

    With ``start_line``/``end_line`` the fragments must sit at those
    lines (two lines of slack each side for off-by-one citations).
    """
    fragments = [
        _normalise(f)
        for line in quote.splitlines()
        for f in _ELISION_RE.split(line)
    ]
    fragments = [f for f in fragments if f]
    if not fragments or file_text is None:
        return False
    lines = file_text.splitlines()
    if start_line is not None and end_line is not None:
        lines = lines[max(start_line - 3, 0):min(end_line + 2, len(lines))]
    window = _normalise(" ".join(lines))
    return all(f in window for f in fragments)


def _normalise(text: str) -> str:
    return " ".join(t for t in text.split() if t != "\\")


def grade_evidence(evidence: list[dict[str, Any]], read_file: Any) -> None:
    """Set ``grounded`` / ``line_accurate`` on each stored citation in place."""
    for e in evidence:
        text = read_file(e["file_path"])
        e["grounded"] = quote_is_grounded(e["quote"], text)
        e["line_accurate"] = quote_is_grounded(e["quote"], text, e["start_line"], e["end_line"])


# ============================================================
# Running the pipeline
# ============================================================


def _claim_from_annotation(ann_claim: dict[str, Any]) -> Any:
    """Build the ``Claim`` the workflow expects from an annotation entry.

    Annotations record a section/page, not the verbatim paper sentence,
    so the claim text doubles as the evidence quote.
    """
    from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence

    span = ann_claim.get("evidence_span", {})
    return Claim(
        claim_id=ann_claim["claim_id"],
        text=ann_claim["text"],
        claim_type=ClaimType.METHOD,
        confidence=1.0,
        paper_evidence=PaperEvidence(
            section_title=str(span.get("section", "")),
            page=int(span.get("page", 1)),
            char_start=int(span.get("char_start", 0)),
            char_end=max(int(span.get("char_end", 1)), int(span.get("char_start", 0)) + 1),
            quote=ann_claim["text"],
        ),
    )


def verify_pair(
    db: Any,
    annotation: dict[str, Any],
    repository_id: UUID,
    *,
    skip: set[str],
) -> Iterator[dict[str, Any]]:
    """Verify every claim of one pair, yielding a record as each finishes."""
    from claim2code_api.db.models import CodeSymbol, Repository
    from claim2code_api.verification.workflow import _verify_one_claim

    repo = db.get(Repository, repository_id)
    clone = Path(repo.clone_path)
    file_cache: dict[str, str | None] = {}

    def _file(rel: str) -> str | None:
        if rel not in file_cache:
            try:
                file_cache[rel] = (clone / rel).read_text(errors="replace")
            except OSError:
                file_cache[rel] = None
        return file_cache[rel]

    for ann_claim in annotation["claims"]:
        if ann_claim["claim_id"] in skip:
            continue
        relevant_ids = _resolve_symbol_ids(db, repository_id, ann_claim.get("relevant_symbols", []))
        relevant_rows = db.query(CodeSymbol).filter(
            CodeSymbol.symbol_id.in_([UUID(i) for i in relevant_ids])
        ).all() if relevant_ids else []
        started = time.monotonic()
        verification, llm_error = _verify_one_claim(
            db=db,
            analysis_id=repo.analysis_id,
            claim=_claim_from_annotation(ann_claim),
            repository_id=repository_id,
            top_k=TOP_K,
        )
        db.rollback()  # the eval never persists; drop anything the pipeline staged
        evidence = verification.code_evidence
        cited = [
            {"file_path": e.file_path, "start_line": e.start_line, "end_line": e.end_line,
             "quote": e.quote}
            for e in evidence
        ]
        grade_evidence(cited, _file)
        grounded = sum(c["grounded"] for c in cited)
        hit = any(
            str(e.symbol_id) in relevant_ids
            or any(
                e.file_path == r.file_path and r.start_line <= e.start_line <= r.end_line
                for r in relevant_rows
            )
            for e in evidence
        )
        record = {
            "pair_id": annotation["pair_id"],
            "claim_id": ann_claim["claim_id"],
            "expected": ann_claim["expected_verification"],
            "predicted": verification.status.value,
            "n_evidence": len(evidence),
            "n_grounded": grounded,
            "n_line_accurate": sum(c["line_accurate"] for c in cited),
            "evidence_hit": hit,
            "evidence": cited,
            "discrepancies": [
                {"type": d.type.value, "severity": d.severity.value, "detail": d.detail}
                for d in verification.discrepancies
            ],
            "error": llm_error,
            "seconds": round(time.monotonic() - started, 1),
        }
        logger.info(
            "%s %s expected=%s predicted=%s grounded=%d/%d hit=%s",
            annotation["pair_id"], ann_claim["claim_id"], ann_claim["expected_verification"],
            verification.status.value, grounded, len(evidence), hit,
        )
        yield record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(
    SessionLocal: Any,
    annotations: list[dict[str, Any]],
    repo_ids: dict[str, str],
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    """Verify every pending claim, streaming records to ``CLAIMS_JSONL``."""
    records = _read_jsonl(CLAIMS_JSONL) if resume else []
    if not resume:
        CLAIMS_JSONL.write_text("")
    finished = {(r["pair_id"], r["claim_id"]) for r in records}
    for annotation in annotations:
        pair_id = annotation["pair_id"]
        if pair_id not in repo_ids:
            continue
        skip = {cid for pid, cid in finished if pid == pair_id}
        with SessionLocal() as db, CLAIMS_JSONL.open("a") as sink:
            for record in verify_pair(db, annotation, UUID(repo_ids[pair_id]), skip=skip):
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                records.append(record)
    return records


def _regrade(
    SessionLocal: Any,
    repo_ids: dict[str, str],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute citation grounding for stored records against the clones."""
    from claim2code_api.db.models import Repository

    with SessionLocal() as db:
        clones = {
            pid: Path(db.get(Repository, UUID(rid)).clone_path) for pid, rid in repo_ids.items()
        }
    for r in records:
        clone = clones.get(r["pair_id"])
        if clone is None:
            continue

        def _file(rel: str, clone: Path = clone) -> str | None:
            try:
                return (clone / rel).read_text(errors="replace")
            except OSError:
                return None

        grade_evidence(r["evidence"], _file)
        r["n_grounded"] = sum(e["grounded"] for e in r["evidence"])
        r["n_line_accurate"] = sum(e["line_accurate"] for e in r["evidence"])
    return records


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pairs", help="comma-separated pair_ids (default: every indexed pair)")
    parser.add_argument("--resume", action="store_true", help="keep finished claims from the last run")
    parser.add_argument(
        "--regrade", action="store_true",
        help="re-score the last run's stored citations without calling the LLM",
    )
    args = parser.parse_args(argv)

    annotations = _load_annotations()
    if args.pairs:
        wanted = set(args.pairs.split(","))
        annotations = [a for a in annotations if a["pair_id"] in wanted]
    SessionLocal = _build_session_factory()
    with SessionLocal() as db:
        repo_ids = resolve_repo_ids(db, annotations)

    RESULTS_DIR.mkdir(exist_ok=True)
    skipped_pairs = [a["pair_id"] for a in annotations if a["pair_id"] not in repo_ids]
    for pair_id in skipped_pairs:
        logger.warning("skipping pair_id=%s (repository not indexed)", pair_id)
    if args.regrade:
        records = _regrade(SessionLocal, repo_ids, _read_jsonl(CLAIMS_JSONL))
        CLAIMS_JSONL.write_text("".join(json.dumps(r) + "\n" for r in records))
    else:
        records = _run(SessionLocal, annotations, repo_ids, resume=args.resume)

    if not records:
        logger.error("no annotated pair has an indexed repository")
        return 2
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_pair.setdefault(r["pair_id"], []).append(r)
    out = {
        "top_k": TOP_K,
        "aggregate": score(records),
        "per_pair": {pid: score(rs) for pid, rs in sorted(by_pair.items())},
        "skipped_pairs": skipped_pairs,
        "claims": records,
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", RESULTS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
