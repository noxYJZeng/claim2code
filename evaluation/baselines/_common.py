"""Shared helpers for the eval baselines.

The baselines all want the same uniform result shape
``list[(symbol_id, score)]`` so the eval runner can iterate
leg-agnostically. This module also handles the
``import claim2code_api`` path manipulation — ``evaluate_retrieval.py``
calls :func:`setup_path` once at startup before importing the
baselines.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add ``apps/api`` to sys.path so ``claim2code_api`` resolves.
# (The package is set up under apps/api/pyproject.toml and not
# installed globally.)
_APPS_API = Path(__file__).resolve().parents[2] / "apps" / "api"


def setup_path() -> None:
    """Ensure :mod:`claim2code_api` resolves from ``apps/api``.

    Idempotent: a second call is a no-op.
    """
    p = str(_APPS_API)
    if p not in sys.path:
        sys.path.insert(0, p)


__all__ = ["setup_path"]