#!/usr/bin/env python3
"""Check that a local evaluation run can start, before it spends hours.

The verification evaluator calls the chat model once per claim (about two
hours with a local 27B model), and a missing model or unindexed pair only
surfaces part-way. This runs every cheap check up front:

  - the database answers and has the application tables;
  - every annotated pair has an indexed repository;
  - the embedding model answers with ``OPENAI_EMBED_DIM`` dimensions, the
    same as the stored code embeddings;
  - the active chat model answers a one-token request;
  - the measured code is committed (a dirty tree is recorded as such).

Exit status 0 when the run can start, 1 otherwise. Nothing is written.
"""
# Imports below bootstrap the repository-root CLI entry point.
# ruff: noqa: E402
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "apps" / "api"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sqlalchemy import text

from evaluation import provenance
from evaluation.evaluate_retrieval import (
    _build_session_factory,
    _load_annotations,
    resolve_repo_ids,
)

Check = Callable[[], tuple[bool, str]]


def _host(url: str) -> str:
    """``scheme://host:port`` only: never print credentials or paths."""
    parts = urlsplit(url)
    if not parts.hostname:
        return "the provider's default endpoint"
    return f"{parts.scheme}://{parts.hostname}{f':{parts.port}' if parts.port else ''}"


def _served_models(base_url: str, api_key: str) -> set[str] | None:
    """Model ids an OpenAI-compatible server lists, or None if it lists none."""
    from openai import OpenAI

    try:
        return {m.id for m in OpenAI(base_url=base_url, api_key=api_key, timeout=30.0).models.list()}
    except Exception:  # noqa: BLE001 — listing is optional for compatible servers
        return None


def _not_served(model: str, base_url: str, api_key: str) -> str | None:
    # Some local servers (LM Studio) answer for an unknown model id with
    # whatever is loaded, so the run would record the wrong model.
    served = _served_models(base_url, api_key)
    if served is None or model in served:
        return None
    return f"{model} is not served there; available: {', '.join(sorted(served)) or 'none'}"


def check_database() -> tuple[bool, str]:
    with _build_session_factory()() as db:
        db.execute(text("SELECT 1 FROM repositories LIMIT 1"))
    return True, "reachable, tables present"


def check_pairs() -> tuple[bool, str]:
    annotations = _load_annotations()
    with _build_session_factory()() as db:
        indexed = resolve_repo_ids(db, annotations)
    missing = [a["pair_id"] for a in annotations if a["pair_id"] not in indexed]
    if missing:
        return False, f"not indexed: {', '.join(missing)} (index them in the app first)"
    return True, f"{len(indexed)}/{len(annotations)} pairs indexed"


def check_embeddings() -> tuple[bool, str]:
    from claim2code_api.app.config import get_settings
    from claim2code_api.indexing.embedding_client import get_embedding_client

    settings = get_settings()
    where = f"{settings.openai_model_embed} at {_host(settings.openai_base_url)}"
    missing = _not_served(settings.openai_model_embed, settings.openai_base_url, settings.openai_api_key)
    if missing:
        return False, missing
    [vector] = get_embedding_client().embed(["preflight"])
    if len(vector) != settings.openai_embed_dim:
        return False, f"{where} returned {len(vector)} dims, OPENAI_EMBED_DIM is {settings.openai_embed_dim}"
    with _build_session_factory()() as db:
        stored = db.execute(text("SELECT vector_dims(embedding) FROM code_embeddings LIMIT 1")).scalar()
    if stored is not None and stored != len(vector):
        return False, f"{where} returns {len(vector)} dims but stored embeddings have {stored}; re-index"
    return True, f"{where}, {len(vector)} dims"


def check_chat() -> tuple[bool, str]:
    from claim2code_api.app.config import get_settings
    from claim2code_api.llm import providers
    from claim2code_api.llm.client import get_llm_client

    provider = providers.active(get_settings())
    where = f"{provider.model} ({provider.preset_id}) at {_host(provider.base_url)}"
    if provider.kind == "openai_compat":
        missing = _not_served(provider.model, provider.base_url, provider.api_key)
        if missing:
            return False, f"{missing} (load it, or fix OPENAI_MODEL_CHAT)"
    ok, error = get_llm_client().check_ready()
    return ok, where if ok else f"{where}: {error} (is the model loaded?)"


def check_tree() -> tuple[bool, str]:
    dirty = provenance.capture("preflight")["code"]["dirty"]
    if dirty:
        return True, "uncommitted changes: the run will be recorded as dirty"
    return True, "clean"


CHECKS: list[tuple[str, Check]] = [
    ("database", check_database),
    ("indexed pairs", check_pairs),
    ("embeddings", check_embeddings),
    ("chat model", check_chat),
    ("git tree", check_tree),
]


def run(checks: list[tuple[str, Check]] = CHECKS) -> bool:
    ready = True
    for name, check in checks:
        try:
            ok, detail = check()
        except Exception as exc:  # noqa: BLE001 — report every failure, keep checking
            ok, detail = False, f"{exc.__class__.__name__}: {' '.join(str(exc).split())[:200]}"
        print(f"  {'✓' if ok else '✗'} {name:14} {detail}")
        ready = ready and ok
    return ready


def main() -> int:
    print("Preflight:")
    return 0 if run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
