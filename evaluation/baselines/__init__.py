"""Eval baselines — thin wrappers around the live retrieval scorers.

Each baseline exposes a uniform ``run(db, repository_id, claim,
top_k=5) -> list[(symbol_id, score)]`` API. The eval runner in
:file:`../evaluate_retrieval.py` is leg-agnostic; it only sees
this shape.
"""