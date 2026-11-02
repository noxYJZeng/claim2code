"""Capture measurement inputs when an evaluator runs, never when it renders.

Only explicit non-secret settings are recorded. Legacy result files remain
usable, but their missing provenance must stay unknown rather than being
filled from the machine that happens to render them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
# What the evaluators and the report write. Running the documented
# sequence rewrites these, which must not mark the measured code dirty.
_OUTPUTS = ("evaluation/results", "evaluation/results.json", "evaluation/report.md")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def annotation_snapshot(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [{k: v for k, v in a.items() if not k.startswith("_")} for a in annotations]
    clean.sort(key=lambda a: a["pair_id"])
    pairs = {}
    issues = []
    for ann in clean:
        pid = ann["pair_id"]
        ref = str(ann.get("repo", {}).get("commit") or "")
        paper_hash = str(ann.get("paper", {}).get("sha256") or "")
        if not _SHA.fullmatch(ref):
            issues.append(f"{pid}: repository revision is not a full immutable commit SHA")
        if not _SHA256.fullmatch(paper_hash):
            issues.append(f"{pid}: paper SHA-256 is missing or a placeholder")
        pairs[pid] = {
            "annotation_sha256": digest(ann), "declared_repo_commit": ref,
            "declared_paper_sha256": paper_hash,
        }
    return {"annotations_sha256": digest(clean), "pairs": pairs, "issues": issues}


def repository_snapshot(
    db: Any, annotations: list[dict[str, Any]], repo_ids: dict[str, str],
) -> dict[str, Any]:
    """Record actual indexed clone revisions and reject pinned mismatches."""
    from claim2code_api.db.models import Repository

    out: dict[str, Any] = {}
    for ann in annotations:
        pid = ann["pair_id"]
        if pid not in repo_ids:
            continue
        row = db.get(Repository, UUID(repo_ids[pid]))
        if row is None:
            raise ValueError(f"{pid}: selected indexed repository does not exist")
        expected_url = ann["repo"]["url"].rstrip("/").removesuffix(".git").lower()
        actual_url = row.url.rstrip("/").removesuffix(".git").lower()
        if expected_url != actual_url:
            raise ValueError(f"{pid}: selected repository URL does not match the annotation")
        clone = Path(row.clone_path)
        actual = _git(clone, "rev-parse", "--verify", "HEAD^{commit}")
        dirty = _git(clone, "status", "--porcelain", "--untracked-files=no")
        expected = str(ann.get("repo", {}).get("commit") or "")
        if _SHA.fullmatch(expected) and actual != expected.lower():
            raise ValueError(f"{pid}: indexed clone does not match the pinned annotation revision")
        out[pid] = {"commit": actual, "dirty": dirty != "" if dirty is not None else None}
    return out


def capture(
    evaluator: str, *, annotations: list[dict[str, Any]] | None = None,
    repositories: dict[str, Any] | None = None, parameters: dict[str, Any] | None = None,
    models: bool = False, corpus: Any = None,
) -> dict[str, Any]:
    files = sorted((ROOT / "apps" / "api" / "claim2code_api").rglob("*.py"))
    files += sorted((ROOT / "apps" / "api" / "claim2code_api").rglob("*.json"))
    # Baseline implementations are measurement code too. Include nested
    # modules so an uncommitted scoring change invalidates resume.
    files += sorted((ROOT / "evaluation").rglob("*.py"))
    files += [ROOT / "apps" / "api" / "pyproject.toml"]
    source_digest = digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in files if p.is_file()})
    meta: dict[str, Any] = {
        "schema_version": 1, "evaluator": evaluator, "run_id": str(uuid4()),
        "started_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "code": {"git_commit": _git(ROOT, "rev-parse", "HEAD"),
                 "source_sha256": source_digest,
                 "dirty": _git(
                     ROOT, "status", "--porcelain", "--untracked-files=no", "--", ".",
                     *(f":(exclude){p}" for p in _OUTPUTS),
                 ) != ""},
        "parameters": parameters or {},
    }
    issues = []
    if annotations is not None:
        meta["dataset"] = annotation_snapshot(annotations)
        issues.extend(meta["dataset"]["issues"])
        meta["repositories"] = repositories or {}
        for pid, repo in meta["repositories"].items():
            if not repo.get("commit"):
                issues.append(f"{pid}: actual indexed clone revision could not be read")
            if repo.get("dirty") is not False:
                issues.append(f"{pid}: indexed clone has uncommitted or unknown changes")
    if corpus is not None:
        meta["corpus_sha256"] = digest(corpus)
    if models:
        import claim2code_api.llm.providers as providers
        from claim2code_api.app.config import get_settings

        settings = get_settings()
        provider = providers.active(settings)
        meta["models"] = {
            "chat_provider": provider.preset_id, "chat_model": provider.model,
            "embedding_model": settings.openai_model_embed,
            "embedding_dimensions": settings.openai_embed_dim,
            # Hash endpoints so a changed gateway invalidates resume
            # without recording userinfo, query strings or credentials.
            "chat_endpoint_sha256": hashlib.sha256(
                getattr(provider, "base_url", "").encode(),
            ).hexdigest(),
            "embedding_endpoint_sha256": hashlib.sha256(
                getattr(settings, "openai_base_url", "").encode(),
            ).hexdigest(),
            "chat_model_revision": None, "embedding_model_revision": None,
            "quantization": None,
        }
        meta["settings"] = {
            "llm_request_timeout_seconds": settings.llm_request_timeout_seconds,
            "embed_batch_size": settings.embed_batch_size,
            "embed_max_retries": settings.embed_max_retries,
        }
    meta["reproducibility_issues"] = issues
    return meta


def finish(meta: dict[str, Any]) -> dict[str, Any]:
    return {**meta, "completed_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


def require_frozen(meta: dict[str, Any]) -> None:
    if meta["reproducibility_issues"]:
        raise ValueError("Reproducibility check failed: " + "; ".join(meta["reproducibility_issues"]))


def measurement_identity(meta: dict[str, Any]) -> str:
    """Resume only the same measured code, models, parameters and inputs."""
    return digest({k: meta.get(k) for k in (
        "code", "models", "settings", "parameters", "dataset", "repositories", "corpus_sha256",
    )})


def comparison_warnings(sections: dict[str, dict[str, Any] | None]) -> list[str]:
    warnings = []
    provenance = {}
    for name, section in sections.items():
        if section is None:
            continue
        meta = section.get("provenance")
        if not meta or not meta.get("run_id") or not meta.get("code"):
            warnings.append(f"{name}: legacy output has no per-run measurement provenance")
            continue
        provenance[name] = meta
        warnings.extend(f"{name}: {issue}" for issue in meta.get("reproducibility_issues", []))
    code_hashes = {p["code"].get("source_sha256") for p in provenance.values()}
    if len(code_hashes) > 1:
        warnings.append("Results were measured with different source trees; do not compare as one run")
    model_sections = [provenance[n] for n in ("retrieval", "verification") if n in provenance]
    if len(model_sections) == 2:
        for field in ("models", "dataset", "repositories"):
            if model_sections[0].get(field) != model_sections[1].get(field):
                warnings.append(f"Retrieval and verification have different {field} provenance")
    return warnings
