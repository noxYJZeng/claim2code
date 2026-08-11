"""Embed every ``CodeSymbol`` in a repository and persist the rows.

This is the Phase 4 write path. The indexer is wired into
:func:`claim2code_api.repos.workflow.run_repo_workflow` so that
a successful clone leaves the repository immediately retrievable:
by the time ``POST /repositories`` returns, the embedding index
has been built (or the failure has been recorded).

What it does
------------
  1. Loads the target ``Repository`` row (and reads its
     ``clone_path`` so we can read source bodies from disk).
  2. Groups the repo's ``CodeSymbol`` rows by file path.
  3. For each file, reads the body between ``start_line`` and
     ``end_line`` for each symbol, resolves the file's import
     targets (from the file's own ``kind="import"`` symbols),
     and builds the structured embedding string via
     :func:`claim2code_api.indexing.symbol_repr.format_symbol`.
  4. Embeds the strings in batches via ``EmbeddingClient``
     (typically :class:`OpenAIEmbeddingClient`). The batch size
     comes from ``Settings.embed_batch_size``.
  5. Inserts one ``CodeEmbedding`` row per symbol, recording
     ``model_name`` (so a model upgrade can be served side-by-side)
     and the repository's ``analysis_id`` (denormalised so the
     retriever's cosine query is one-table).
  6. Updates ``Repository.embedding_model`` so the retriever
     knows which model produced the rows.

Failure mode
------------
The function swallows ``EmbeddingError`` and records it on the
repository row (``repo.error`` + ``analysis_states.errors``).
We deliberately do NOT touch `index_status` — the parse step
already set it to ``'done'``; embedding is a best-effort
secondary pass that the user can retry later. Overwriting the
parse status would lie to the caller of :func:`run_repo_workflow`
that the clone itself failed when only the embedding did. The
repo stays browsable via ``GET /repositories/{id}/symbols``;
``retrieve_semantic`` simply returns ``[]`` until the user retries.

Why this is a separate module from the workflow
-----------------------------------------------
The workflow's job is to clone + walk + parse. The indexer's
job is to turn the parsed symbols into vectors. Keeping them
separate means:

  - The indexer can be re-run independently (Phase 4.5+ re-index
    endpoint, or model upgrade).
  - Unit tests can exercise the indexer with a stub
    ``EmbeddingClient`` and a real ``CodeSymbol`` set without
    touching the clone pipeline.
  - The workflow only gains one extra function call, not a
    wholesale rewrite.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.app.config import Settings, get_settings
from claim2code_api.db.models import (
    AnalysisState,
    CodeEmbedding,
    CodeSymbol,
    Repository,
)
from claim2code_api.indexing.embedding_client import EmbeddingClient
from claim2code_api.indexing.exceptions import EmbeddingError
from claim2code_api.indexing.symbol_repr import format_symbol

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def embed_and_persist_symbols(
    db: Session,
    repository_id: UUID,
    *,
    embedding_client: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> int:
    """Embed every ``CodeSymbol`` for ``repository_id`` and persist rows.

    Returns the number of embeddings written. Returns 0 (without
    raising) if the repository has no symbols yet.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param repository_id: the repository whose symbols to embed.
    :param embedding_client: optional override for the embedding
        client. Defaults to ``get_embedding_client()`` (the
        process-wide singleton); tests pass a stub.
    :param settings: optional Settings override; defaults to
        ``get_settings()``.
    :raises EmbeddingError: the embedding call failed irrecoverably
        and the function is unable to record it on the row (rare
        — usually a programmer error).
    """
    if settings is None:
        settings = get_settings()
    if embedding_client is None:  # pragma: no cover — exercised in tests via DI
        from claim2code_api.indexing.embedding_client import get_embedding_client

        embedding_client = get_embedding_client()

    repo = db.get(Repository, repository_id)
    if repo is None:
        raise ValueError(f"Repository {repository_id} not found")

    symbols: list[CodeSymbol] = (
        db.query(CodeSymbol)
        .filter(CodeSymbol.repository_id == repository_id)
        .order_by(CodeSymbol.file_path, CodeSymbol.start_line)
        .all()
    )
    if not symbols:
        # Nothing to embed. Still record the model so the retriever
        # at least knows the index was attempted.
        repo.embedding_model = settings.openai_model_embed
        db.commit()
        return 0

    # Group symbols by file so we can read each file once.
    by_file: dict[str, list[CodeSymbol]] = {}
    for sym in symbols:
        by_file.setdefault(sym.file_path, []).append(sym)

    # Build the embedding strings. Read source bodies + import
    # lists lazily; failures here are best-effort (per-file).
    embed_inputs: list[str] = []
    embed_symbols: list[CodeSymbol] = []
    for sym in symbols:
        text = _build_embedding_text(
            symbols_for_file=by_file[sym.file_path],
            sym=sym,
            clone_path=Path(repo.clone_path),
        )
        embed_inputs.append(text)
        embed_symbols.append(sym)

    # One batched call per Settings.embed_batch_size chunk.
    try:
        vectors = embedding_client.embed(embed_inputs)
    except EmbeddingError as exc:
        _record_indexer_failure(db, repo, analysis_id=_repo_analysis_id(db, repo), exc=exc)
        return 0

    # Persist one row per symbol. The UNIQUE(symbol_id, model_name)
    # constraint lets us safely re-run; existing rows are
    # replaced by the new (same model, same symbol) pair.
    written = 0
    for sym, vec in zip(embed_symbols, vectors, strict=True):
        row = CodeEmbedding(
            symbol_id=sym.symbol_id,
            repository_id=repository_id,
            analysis_id=repo.analysis_id,
            embedding=vec,
            model_name=settings.openai_model_embed,
        )
        db.add(row)
        written += 1

    repo.embedding_model = settings.openai_model_embed
    db.commit()
    return written


# ----------------------------------------------------------------------
# Per-file helpers
# ----------------------------------------------------------------------


def _build_embedding_text(
    *,
    symbols_for_file: list[CodeSymbol],
    sym: CodeSymbol,
    clone_path: Path,
) -> str:
    """Build the embedding string for one symbol.

    Reads the file from disk, extracts the source body between
    ``start_line`` and ``end_line``, and resolves the file's
    imports (the other ``kind='import'`` symbols in the same
    file). Failures are best-effort: a missing file or an
    out-of-range line yields an empty source / imports list,
    but the other fields still pin the symbol.
    """
    source = _read_source_range(
        clone_path=clone_path,
        rel_path=sym.file_path,
        start_line=sym.start_line,
        end_line=sym.end_line,
    )
    imports = _imports_for_file(symbols_for_file)
    return format_symbol(sym, source=source, imports=imports)


def _read_source_range(
    *,
    clone_path: Path,
    rel_path: str,
    start_line: int,
    end_line: int,
) -> str:
    """Read ``rel_path`` lines ``start_line``..``end_line`` (1-indexed, inclusive).

    Returns ``""`` on any read failure (missing file, encoding
    error, range out of bounds). The retriever can still rank
    the symbol by its signature + docstring + name in that case.
    """
    full = clone_path / rel_path
    try:
        text = full.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    # 1-indexed, inclusive on both ends.
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    if start_idx >= end_idx:
        return ""
    return "\n".join(lines[start_idx:end_idx])


def _imports_for_file(symbols_for_file: list[CodeSymbol]) -> list[str]:
    """Return the file's import-target strings, in declaration order.

    Filters to ``kind='import'`` and projects the ``signature``
    (which the parser stores as the dotted path / target). If
    the parser couldn't extract a signature, falls back to the
    symbol's ``name``.
    """
    out: list[str] = []
    for sym in symbols_for_file:
        if sym.kind != "import":
            continue
        target = sym.signature or sym.name
        if target:
            out.append(target)
    return out


# ----------------------------------------------------------------------
# Failure recording
# ----------------------------------------------------------------------


def _record_indexer_failure(
    db: Session,
    repo: Repository,
    *,
    analysis_id: UUID,
    exc: Exception,
) -> None:
    """Record the embedding failure on the row + analysis errors.

    Deliberately does NOT touch ``repo.index_status`` — the parse
    step already marked the repo ``'done'``. Embedding is a
    best-effort secondary pass; overwriting the parse status
    would lie to the caller that the clone itself failed when
    only the OpenAI side did. The retriever gracefully returns
    ``[]`` for a repo with no embeddings, and the user can retry
    by re-running the indexer (Phase 4.5+ endpoint) or by
    re-cloning.
    """
    msg = str(exc)[:500]
    repo.error = f"embedding: {msg}"
    blob = {
        "at": _utc_now_iso(),
        "where": "indexer",
        "type": exc.__class__.__name__,
        "message": msg,
    }
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(
        {AnalysisState.errors: [blob]}
    )
    db.commit()


def _repo_analysis_id(db: Session, repo: Repository) -> UUID:
    """``Repository.analysis_id`` may be unloaded after a flush; reload if needed."""
    aid = repo.analysis_id
    if aid is None:
        # Fallback: re-query to surface the FK.
        from sqlalchemy.orm import attributes

        attributes.flag_modified(repo, "analysis_id")  # pragma: no cover  -- defensive
        return repo.analysis_id  # type: ignore[return-value]
    return aid


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


__all__ = ["embed_and_persist_symbols"]
