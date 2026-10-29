#!/usr/bin/env python3
"""Run every evaluator, then render ``results.json`` + ``report.md``.

Safety needs nothing external; retrieval needs the indexed repos and the
embedding model; verification additionally calls the chat model once per
annotated claim (the slow part). ``--skip-verification`` reuses the last
``results/verification.json``.
"""
# Imports below bootstrap the repository-root CLI entry point.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "apps" / "api"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluation import (
    evaluate_retrieval,
    evaluate_safety,
    evaluate_verification,
    report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--skip-verification", action="store_true")
    args = parser.parse_args()
    status = evaluate_safety.main([])
    status = evaluate_retrieval.main([]) or status
    if not args.skip_verification:
        status = evaluate_verification.main([]) or status
    report.main([])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
