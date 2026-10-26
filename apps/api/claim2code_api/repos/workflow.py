"""Synchronous repository-ingestion workflow.

Implements the steps from ``docs/architecture/phase-3-design.md``
§2.5. The whole workflow runs inside one request:

  1. Validate URL (GitHub-only in v1; see ``cloner.parse_github_url``)
  2. HEAD pre-flight size guard (best-effort; uses GITHUB_TOKEN)
  3. ``git clone --depth 1`` → ``data/repos/<owner>__<repo>/``
  4. INSERT Repository row (index_status='cloning')
  5. UPDATE AnalysisState.repository_status='cloning'
  6. Walk the tree (skip node_modules, .git, dist, build, …)
  7. Per file: tree-sitter parse → CodeSymbol records
  8. Config files (.yaml/.yml/.json/.toml/.ini/.cfg/.conf)
  9. Lockfiles (requirements*.txt / poetry.lock / Pipfile.lock /
     package-lock.json / yarn.lock / pnpm-lock.yaml)
 10. UPDATE Repository (index_status='done', n_files, n_symbols)
 11. UPDATE AnalysisState.repository_status='done', completed_at=now()
 12. Return summary

Every step's exception is caught at the workflow boundary, written into
``analysis_states.errors``, and re-raised mapped (so the router can
return the right HTTP status code).

Why synchronous and not BackgroundTasks?
-----------------------------------------
Same argument as the paper workflow (Phase 2). Phase 3 has no
concurrent load yet. Adding a worker pool + idempotency keys + status
polling is premature — Phase 5+ revisits when load demands it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from claim2code_api.db.models import (
    AnalysisState,
    CodeSymbol,
    Repository,
    RepositoryConfig,
    RepositoryDependency,
)
from claim2code_api.indexing.indexer import embed_and_persist_symbols
from claim2code_api.repos import cloner
from claim2code_api.repos.configs import parse_config_file
from claim2code_api.repos.exceptions import (  # noqa: F401  — referenced in docstrings
    RepoCloneError,
    RepoInvalidUrlError,
    RepoTooLargeError,
    UnsupportedLanguageError,
)
from claim2code_api.repos.lockfiles import parse_lockfile
from claim2code_api.repos.parser import parse_file
from claim2code_api.repos.schemas import CodeSymbol as CodeSymbolSchema
from claim2code_api.repos.storage import (
    WalkedFile,
    classify_file,
    walk_repo_files,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepoWorkflowResult:
    """The summary returned by :func:`run_repo_workflow`."""

    repository_id: UUID
    analysis_id: UUID
    index_status: str
    n_files: int
    n_symbols: int
    n_configs: int
    n_dependencies: int


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run_repo_workflow(
    *,
    db: Session,
    analysis_id: UUID,
    url: str,
    exclude_paths: list[str] | None = None,
) -> RepoWorkflowResult:
    """Execute the clone + walk + parse + persist pipeline for one repo.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param analysis_id: the analysis this repo is bound to.
    :param url: a GitHub URL (``https://github.com/<owner>/<repo>``).
    :param exclude_paths: repo-relative paths / globs to skip while
        walking (see :func:`storage.walk_repo_files`).

    :raises RepoInvalidUrlError: URL is malformed or non-GitHub.
    :raises RepoTooLargeError: GitHub-reported size exceeds the cap.
    :raises RepoCloneError: ``git clone`` failed.
    """
    # Validate URL — this raises RepoInvalidUrlError (mapped to 422).
    parsed = cloner.parse_github_url(url)

    # Insert a placeholder Repository row so we have an id to thread
    # through status updates. Use ``init`` to get the PK without a
    # separate SELECT round-trip.
    repo_row = Repository(
        analysis_id=analysis_id,
        url=parsed.url,
        owner=parsed.owner,
        name=parsed.name,
        default_branch="main",
        clone_path=str(cloner.clone_dir_for(parsed.owner, parsed.name)),
    )
    db.add(repo_row)
    db.flush()  # populate repository_id
    repository_id = repo_row.repository_id

    try:
        _set_repo_status(db, repository_id, "cloning")
        _set_analysis_repo_status(db, analysis_id, "cloning")
        db.commit()

        # 3. git clone (raises RepoCloneError / RepoTooLargeError).
        clone_result = cloner.clone_repository(parsed.url)
        repo_row.default_branch = clone_result.default_branch
        repo_row.clone_path = str(clone_result.clone_path)
        db.commit()

        # 6-9. Walk + parse.
        _set_repo_status(db, repository_id, "parsing")
        _set_analysis_repo_status(db, analysis_id, "parsing")
        db.commit()

        n_files, n_symbols, n_configs, n_deps = _walk_and_persist(
            db=db,
            repository_id=repository_id,
            clone_path=clone_result.clone_path,
            exclude_paths=exclude_paths,
        )

        # 10-11. Mark done.
        repo_row.n_files = n_files
        repo_row.n_symbols = n_symbols
        repo_row.indexed_at = datetime.now(UTC)
        _set_repo_status(db, repository_id, "done")
        _set_analysis_repo_status(db, analysis_id, "done", completed=True)
        db.commit()

        # Phase 4 — embed every CodeSymbol so the repo is immediately
        # retrievable. ``embed_and_persist_symbols`` swallows
        # EmbeddingError and records it on the row; the workflow
        # stays linear and the index_status='done' we just set
        # remains, so the repo is still browsable via
        # GET /repositories/{id}/symbols. Retrieval will simply
        # return [] until the user re-runs indexing.
        embed_and_persist_symbols(db, repository_id)

        return RepoWorkflowResult(
            repository_id=repository_id,
            analysis_id=analysis_id,
            index_status="done",
            n_files=n_files,
            n_symbols=n_symbols,
            n_configs=n_configs,
            n_dependencies=n_deps,
        )
    except UnsupportedLanguageError:
        # File-extension errors are non-fatal — they just skip the file.
        # Reaching here means a bug in the dispatch; surface it.
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        repo_row.index_status = "error"
        repo_row.error = str(exc)[:500]
        db.commit()
        _record_error(db, analysis_id, exc)
        raise


# ----------------------------------------------------------------------
# Walk + parse
# ----------------------------------------------------------------------


def _walk_and_persist(
    *,
    db: Session,
    repository_id: UUID,
    clone_path,
    exclude_paths: list[str] | None = None,
) -> tuple[int, int, int, int]:
    """Walk the clone, dispatch to extractors, return counts.

    Args:
        db: Active session.
        repository_id: The :class:`Repository` to associate rows with.
        clone_path: Absolute path to the cloned directory.

    Returns:
        ``(n_files, n_symbols, n_configs, n_dependencies)``.
    """
    from pathlib import Path  # local import to avoid circular import at module load

    files = walk_repo_files(Path(clone_path), exclude_paths=exclude_paths)
    n_files = len(files)
    n_symbols = 0
    n_configs = 0
    n_deps = 0

    # Build a name → symbol row mapping for parent linking (class → method).
    name_to_symbol_id: dict[tuple[str, str], UUID] = {}

    for wf in files:
        cls = classify_file(wf.absolute_path)
        if cls.mode == "source":
            n_symbols += _extract_symbols(db, repository_id, wf, name_to_symbol_id)
        elif cls.mode == "config":
            n_configs += _extract_config(db, repository_id, wf)
        elif cls.mode == "lockfile":
            n_deps += _extract_lockfile(db, repository_id, wf)

    db.commit()
    return n_files, n_symbols, n_configs, n_deps


def _extract_symbols(
    db: Session,
    repository_id: UUID,
    wf: WalkedFile,
    name_to_symbol_id: dict[tuple[str, str], UUID],
) -> int:
    """Parse one source file and persist CodeSymbol rows.

    Returns the number of symbols persisted.
    """
    try:
        symbols: list[CodeSymbolSchema] = parse_file(wf.absolute_path)
    except UnsupportedLanguageError:
        return 0
    except Exception as exc:  # noqa: BLE001 — best-effort, never abort the walk
        logger.warning("parse failed for %s: %s", wf.relative_path, exc)
        return 0

    # The parser records the on-disk path. Override with the
    # repo-relative POSIX path so the DB stores the canonical form
    # the frontend renders ("file:line:col" against the repo, not
    # against the absolute clone path on disk).
    for sym in symbols:
        sym.file_path = wf.relative_path

    inserted = 0
    class_rows: list[CodeSymbol] = []
    method_rows: list[CodeSymbol] = []
    for sym in symbols:
        row = CodeSymbol(
            repository_id=repository_id,
            kind=sym.kind.value,
            language=sym.language.value,
            file_path=sym.file_path,
            start_line=sym.start_line,
            start_col=sym.start_col,
            end_line=sym.end_line,
            end_col=sym.end_col,
            name=sym.name,
            signature=sym.signature,
            docstring=sym.docstring,
        )
        db.add(row)
        db.flush()  # populate symbol_id
        inserted += 1
        # Track class rows so methods can pick up parent_symbol_id later.
        if sym.kind.value == "class":
            name_to_symbol_id[(sym.file_path, sym.name)] = row.symbol_id
            class_rows.append(row)
        elif sym.kind.value == "method":
            method_rows.append(row)

    # Link each method to the innermost class in this file whose line
    # range encloses it. Line containment works for every parsed
    # language and handles nested / same-named classes, which a
    # name lookup cannot.
    for method in method_rows:
        enclosing = [
            c
            for c in class_rows
            if c.start_line <= method.start_line and method.end_line <= c.end_line
        ]
        if enclosing:
            parent = max(enclosing, key=lambda c: c.start_line)
            method.parent_symbol_id = parent.symbol_id
    return inserted


def _extract_config(
    db: Session,
    repository_id: UUID,
    wf: WalkedFile,
) -> int:
    """Parse one config file and persist a RepositoryConfig row."""
    try:
        cfg = parse_config_file(wf.absolute_path, wf.relative_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("config parse failed for %s: %s", wf.relative_path, exc)
        return 0
    row = RepositoryConfig(
        repository_id=repository_id,
        file_path=cfg.file_path,
        format=cfg.format.value,
        size_bytes=cfg.size_bytes,
        content_json=cfg.content_json,
        error=cfg.error,
    )
    db.add(row)
    db.flush()
    return 1


def _extract_lockfile(
    db: Session,
    repository_id: UUID,
    wf: WalkedFile,
) -> int:
    """Parse one lockfile and persist RepositoryDependency rows."""
    try:
        deps = parse_lockfile(wf.absolute_path, wf.relative_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("lockfile parse failed for %s: %s", wf.relative_path, exc)
        return 0
    inserted = 0
    for d in deps:
        row = RepositoryDependency(
            repository_id=repository_id,
            ecosystem=d.ecosystem.value,
            name=d.name,
            version_spec=d.version_spec,
            source_file=d.source_file,
        )
        db.add(row)
        inserted += 1
    db.flush()
    return inserted


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _set_repo_status(db: Session, repository_id: UUID, status: str) -> None:
    db.query(Repository).filter(Repository.repository_id == repository_id).update(
        {Repository.index_status: status}
    )


def _set_analysis_repo_status(
    db: Session,
    analysis_id: UUID,
    status: str,
    *,
    completed: bool = False,
) -> None:
    update: dict[object, object] = {AnalysisState.repository_status: status}
    if completed:
        update[AnalysisState.completed_at] = datetime.now(UTC)
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(
        update  # type: ignore[arg-type]
    )


def _record_error(db: Session, analysis_id: UUID, exc: Exception) -> None:
    """Append a structured error entry to AnalysisState.errors."""
    blob = {
        "at": datetime.now(UTC).isoformat(),
        "where": "repo_workflow",
        "type": exc.__class__.__name__,
        "message": str(exc)[:500],
    }
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(
        {AnalysisState.errors: [blob]}
    )
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(
        {AnalysisState.repository_status: "error"}
    )
    db.commit()


__all__ = ["RepoWorkflowResult", "run_repo_workflow"]

# Silence the "imported but unused" linter for func (used in migrations elsewhere).
_ = func
