#!/usr/bin/env python3
"""Prepare pinned benchmark repositories without executing downloaded code."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "apps" / "api"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sqlalchemy import and_, func  # noqa: E402

from claim2code_api.app.config import get_settings  # noqa: E402
from claim2code_api.db.models import AnalysisState, CodeEmbedding, CodeSymbol, Repository  # noqa: E402
from claim2code_api.indexing.indexer import embed_and_persist_symbols  # noqa: E402
from claim2code_api.repos.cloner import parse_github_url  # noqa: E402
from claim2code_api.repos.storage import resolve_repo_file  # noqa: E402
from claim2code_api.repos.workflow import _walk_and_persist  # noqa: E402
from evaluation.evaluate_retrieval import (  # noqa: E402
    _build_session_factory,
    _load_annotations,
    _resolve_symbol_groups,
)
from evaluation.evaluate_verification import quote_at_exact_lines  # noqa: E402
from evaluation.provenance import capture, finish  # noqa: E402


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"core.hooksPath={os.devnull}", "-C", str(directory), *args],
        check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    ).stdout.strip()


def check_clone(directory: Path, url: str, commit: str) -> None:
    """Refuse stale or modified clones; never reset a user's checkout."""
    if directory.is_symlink() or not (directory / ".git").exists():
        raise ValueError(f"{directory}: expected a non-symlink Git checkout")
    if _git(directory, "rev-parse", "HEAD") != commit.lower():
        raise ValueError(f"{directory}: checkout differs from the pinned revision; use a separate directory")
    if _git(directory, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"{directory}: checkout has local changes; preserve it and use a separate directory")
    remote = parse_github_url(_git(directory, "remote", "get-url", "origin"))
    if remote.url.lower() != parse_github_url(url).url.lower():
        raise ValueError(f"{directory}: origin differs from the annotated repository")


def ensure_clone(directory: Path, url: str, commit: str) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("Benchmark repositories require a full immutable commit SHA")
    canonical = parse_github_url(url).url
    if directory.exists() or directory.is_symlink():
        check_clone(directory, canonical, commit)
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
        staged = Path(temporary)
        _git(staged, "init", "--quiet")
        _git(staged, "remote", "add", "origin", canonical)
        _git(staged, "fetch", "--depth", "1", "origin", commit)
        _git(staged, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        check_clone(staged, canonical, commit)
        staged.rename(directory)
    return directory


def load_suite(manifest: Path, annotations_dir: Path) -> list[dict[str, Any]]:
    wanted = json.loads(manifest.read_text())["pair_ids"]
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("Suite pair_ids must be nonempty and unique")
    annotations = {a["pair_id"]: a for a in _load_annotations(annotations_dir)}
    selected = []
    for pair_id in wanted:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", pair_id) or pair_id not in annotations:
            raise ValueError(f"Unknown or invalid suite pair_id: {pair_id}")
        annotation = annotations[pair_id]
        parse_github_url(annotation["repo"]["url"])
        if not re.fullmatch(r"[0-9a-fA-F]{40}", annotation["repo"]["commit"]):
            raise ValueError(f"{pair_id}: repository revision is not a full immutable commit SHA")
        selected.append(annotation)
    return selected


def vector_coverage(db: Any, repository_id: Any, model: str) -> tuple[int, int, int]:
    symbols = db.query(func.count(CodeSymbol.symbol_id)).filter(
        CodeSymbol.repository_id == repository_id, CodeSymbol.kind != "import",
    ).scalar() or 0
    covered = db.query(func.count(func.distinct(CodeSymbol.symbol_id))).join(
        CodeEmbedding, and_(CodeEmbedding.symbol_id == CodeSymbol.symbol_id,
                            CodeEmbedding.repository_id == repository_id),
    ).filter(CodeSymbol.repository_id == repository_id, CodeSymbol.kind != "import",
             CodeEmbedding.model_name == model).scalar() or 0
    total = db.query(func.count(CodeEmbedding.embedding_id)).filter(
        CodeEmbedding.repository_id == repository_id,
    ).scalar() or 0
    return symbols, covered, total


def _check_targets(db: Any, repo: Repository, annotation: dict[str, Any]) -> int:
    targets = [symbol for claim in annotation["claims"] for symbol in claim["relevant_symbols"]]
    groups = _resolve_symbol_groups(db, repo.repository_id, targets)
    for symbol, group in zip(targets, groups, strict=True):
        if not group:
            raise ValueError(f"{annotation['pair_id']}: annotated symbol was not indexed: {symbol}")
    for claim in annotation["claims"]:
        for item in claim.get("source_evidence", []):
            path = resolve_repo_file(Path(repo.clone_path), item["file_path"])
            if path is None or not quote_at_exact_lines(
                item["quote"], path.read_text(errors="replace"), item["start_line"], item["end_line"],
            ):
                raise ValueError(f"{annotation['pair_id']}: annotated source evidence does not match its exact lines")
    if annotation["repo"].get("framework") == "pytorch":
        evidence = annotation["repo"].get("framework_evidence", [])
        if not evidence:
            raise ValueError(f"{annotation['pair_id']}: PyTorch framework evidence is missing")
        for item in evidence:
            path = resolve_repo_file(Path(repo.clone_path), item["file_path"])
            if path is None or item["quote"] not in path.read_text(errors="replace"):
                raise ValueError(f"{annotation['pair_id']}: framework evidence does not match pinned source")
            imports = ast.walk(ast.parse(path.read_text(errors="replace")))
            if not any((isinstance(node, ast.Import) and any(a.name.split(".")[0] == "torch" for a in node.names))
                       or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch")
                       for node in imports):
                raise ValueError(f"{annotation['pair_id']}: framework evidence file does not import PyTorch")
    return len(targets)


def prepare_pair(db: Any, annotation: dict[str, Any], repos_dir: Path, *, parse_only: bool) -> dict[str, Any]:
    parsed = parse_github_url(annotation["repo"]["url"])
    commit = annotation["repo"]["commit"]
    repo = db.query(Repository).filter_by(owner=parsed.owner, name=parsed.name).one_or_none()
    reused = repo is not None
    if repo is None:
        clone = ensure_clone(repos_dir / annotation["pair_id"], parsed.url, commit)
        state = AnalysisState(repository_status="parsing")
        db.add(state)
        db.flush()
        repo = Repository(analysis_id=state.analysis_id, owner=parsed.owner, name=parsed.name,
                          url=parsed.url, clone_path=str(clone), index_status="parsing")
        db.add(repo)
        db.flush()
        try:
            files, symbols, _configs, _deps = _walk_and_persist(
                db=db, repository_id=repo.repository_id, clone_path=clone,
            )
            repo.n_files, repo.n_symbols = files, symbols
            repo.index_status = state.repository_status = "done"
            repo.indexed_at = datetime.now(UTC)
            db.commit()
        except Exception:
            db.rollback()
            raise
    else:
        if parse_github_url(repo.url).url.lower() != parsed.url.lower():
            raise ValueError(f"{annotation['pair_id']}: existing repository URL is incompatible")
        check_clone(Path(repo.clone_path), parsed.url, commit)
        if repo.index_status != "done" or repo.error:
            raise ValueError(f"{annotation['pair_id']}: existing index is incomplete; use a separate benchmark database")
    targets = _check_targets(db, repo, annotation)
    model = get_settings().openai_model_embed
    symbols, covered, total = vector_coverage(db, repo.repository_id, model)
    if not symbols:
        raise ValueError(f"{annotation['pair_id']}: no implementation symbols were indexed")
    if not parse_only:
        if repo.embedding_model not in (None, model):
            raise ValueError(f"{annotation['pair_id']}: embedding model changed; use a separate benchmark database")
        if covered < symbols:
            if total:
                raise ValueError(f"{annotation['pair_id']}: partial/stale embeddings; use a separate benchmark database")
            written = embed_and_persist_symbols(db, repo.repository_id)
            symbols, covered, total = vector_coverage(db, repo.repository_id, model)
            if not written or repo.error or covered != symbols or repo.embedding_model != model:
                raise ValueError(f"{annotation['pair_id']}: embedding failed or left incomplete coverage")
        if db.get_bind().dialect.name == "postgresql":
            dimensions = {r[0] for r in db.query(func.vector_dims(CodeEmbedding.embedding)).filter(
                CodeEmbedding.repository_id == repo.repository_id, CodeEmbedding.model_name == model,
            ).distinct()}
            if dimensions != {get_settings().openai_embed_dim}:
                raise ValueError(f"{annotation['pair_id']}: stored embedding dimensions do not match current settings")
    return {"pair_id": annotation["pair_id"], "repository_id": str(repo.repository_id),
            "url": parsed.url, "commit": commit, "clone_path": repo.clone_path,
            "n_files": repo.n_files, "n_symbols": repo.n_symbols,
            "n_implementation_symbols": symbols, "n_embedded_implementation_symbols": covered, "n_embeddings": total,
            "n_resolved_annotation_symbols": targets, "reused": reused,
            "embedding_model": repo.embedding_model, "framework": annotation["repo"].get("framework")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation/suites/resume-targets.json")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "evaluation/annotations")
    parser.add_argument("--repos-dir", type=Path, default=ROOT / "data/benchmark/repos")
    parser.add_argument("--parse-only", action="store_true", help="index code without calling an embedding model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        annotations = load_suite(args.manifest, args.annotations_dir)
        factory = _build_session_factory()
        with factory() as db:
            rows = [prepare_pair(db, a, args.repos_dir, parse_only=args.parse_only) for a in annotations]
        provenance = capture("benchmark_preparation", annotations=annotations,
                             repositories={r["pair_id"]: {"commit": r["commit"], "dirty": False} for r in rows},
                             parameters={"parse_only": args.parse_only}, models=not args.parse_only)
        pairs = {r["pair_id"]: {**r, "source_checked": True, "indexed": True} for r in rows}
        result = {"schema_version": 1, "manifest": str(args.manifest), "parse_only": args.parse_only,
                  "n_repositories": len(rows), "n_pytorch_repositories": sum(r["framework"] == "pytorch" for r in rows),
                  "n_claims": sum(len(a["claims"]) for a in annotations), "repositories": rows,
                  "pairs": pairs, "provenance": finish(provenance)}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        else:
            print(json.dumps(result, indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Benchmark preparation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
