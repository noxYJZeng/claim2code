#!/usr/bin/env python3
"""Measure a new checkout on a local model without overwriting archived results."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
for directory in (ROOT, ROOT / "apps" / "api"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def build_commands(
    *, manifest: Path, annotations_dir: Path, output_dir: Path,
    pairs: list[str], prepare: bool, resume: bool, require_reproducible: bool,
) -> list[list[str]]:
    """Keep stage order and every evaluator's output destination explicit."""
    python = sys.executable
    results = output_dir / "results"
    selection = ["--annotations-dir", str(annotations_dir), "--pairs", ",".join(pairs),
                 "--output-dir", str(results)]
    frozen = ["--require-reproducible"] if require_reproducible else []
    commands = []
    if prepare:
        commands.append([python, str(EVAL / "prepare_benchmark.py"), "--manifest", str(manifest),
                         "--annotations-dir", str(annotations_dir),
                         "--output", str(output_dir / "coverage.json")])
    commands += [
        [python, str(EVAL / "evaluate_retrieval.py"), *selection, *frozen],
        [python, str(EVAL / "evaluate_verification.py"), *selection, "--require-all-pairs",
         *frozen, *(["--resume"] if resume else [])],
        [python, str(EVAL / "evaluate_safety.py"), "--output-dir", str(results)],
        [python, str(EVAL / "report.py"), "--results-dir", str(results),
         "--output-dir", str(output_dir),
         *(["--require-provenance"] if require_reproducible else [])],
        [python, str(EVAL / "targets.py"), "--manifest", str(manifest),
         "--annotations-dir", str(annotations_dir), "--results-dir", str(results),
         "--coverage", str(output_dir / "coverage.json"), "--output-dir", str(output_dir)],
    ]
    return commands


def preflight(*, check_repositories: bool, pairs: list[str], annotations_dir: Path) -> None:
    """Check local dependencies before spending time on model inference."""
    import httpx
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    import claim2code_api.llm.providers as providers
    from claim2code_api.app.config import get_settings
    from evaluation.evaluate_retrieval import (
        _build_session_factory,
        _load_annotations,
        resolve_repo_ids,
    )
    from evaluation.provenance import repository_snapshot

    settings = get_settings()
    provider = providers.active(settings)
    if provider.preset_id != "env":
        raise ValueError("Local evaluation uses OPENAI_* settings; switch chat provider to env first")
    # Both requests are read-only model discovery, with no prompt or source upload.
    endpoints = [
        (settings.openai_base_url, settings.openai_api_key,
         [settings.openai_model_chat, settings.openai_model_embed]),
    ]
    for endpoint, key, wanted in endpoints:
        try:
            with httpx.Client(timeout=15, trust_env=False) as client:
                response = client.get(endpoint.rstrip("/") + "/models",
                                      headers={"Authorization": f"Bearer {key}"})
                response.raise_for_status()
                available = {model["id"] for model in response.json()["data"]}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            # Do not print request URLs or headers: they may contain credentials.
            raise ValueError("Cannot read the model list; check LM Studio and OPENAI_* settings") from exc
        missing = set(wanted) - available
        if missing:
            raise ValueError("Models not advertised by the local server: " + ", ".join(sorted(missing)))
    try:
        session_factory = _build_session_factory()
        with session_factory() as db:
            db.execute(text("SELECT 1"))
            if check_repositories:
                annotations = [a for a in _load_annotations(annotations_dir) if a["pair_id"] in pairs]
                ids = resolve_repo_ids(db, annotations)
                missing_pairs = set(pairs) - set(ids)
                if missing_pairs:
                    raise ValueError("Repositories not prepared: " + ", ".join(sorted(missing_pairs))
                                     + "; add --prepare")
                repository_snapshot(db, annotations, ids)
    except SQLAlchemyError as exc:
        raise ValueError("Cannot query the benchmark database; check DATABASE_URL and migrations") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=EVAL / "suites" / "resume-targets.json")
    parser.add_argument("--annotations-dir", type=Path, default=EVAL / "annotations")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare", action="store_true", help="clone pinned sources and index missing repositories")
    parser.add_argument("--resume", action="store_true", help="resume verification in an explicit --output-dir")
    parser.add_argument("--require-reproducible", action="store_true", help="also require frozen PDF hashes and provenance")
    parser.add_argument("--dry-run", action="store_true", help="show commands without database, network, or inference")
    args = parser.parse_args(argv)
    if args.resume and args.output_dir is None:
        parser.error("--resume requires the previous --output-dir")
    try:
        manifest = args.manifest.resolve()
        suite = json.loads(manifest.read_text())
        pairs = suite["pair_ids"]
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError("Manifest pair IDs must be nonempty and unique")
        annotations_dir = args.annotations_dir.resolve()
        for pair in pairs:
            # A manifest cannot escape the annotation directory.
            if Path(pair).name != pair or not (annotations_dir / f"{pair}.json").is_file():
                raise ValueError(f"Missing or invalid annotation pair: {pair}")
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (args.output_dir or EVAL / "runs" / f"{stamp}-{uuid4().hex[:8]}").resolve()
        if output_dir == EVAL or output_dir == EVAL / "results":
            raise ValueError("Choose a separate run directory to preserve the archived benchmark")
        if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
            raise ValueError("Output directory is nonempty; choose another one or use --resume")
        commands = build_commands(
            manifest=manifest, annotations_dir=annotations_dir, output_dir=output_dir,
            pairs=pairs, prepare=args.prepare, resume=args.resume,
            require_reproducible=args.require_reproducible,
        )
        if args.dry_run:
            for command in commands:
                print(json.dumps(command))
            return 0
        preflight(check_repositories=not args.prepare, pairs=pairs, annotations_dir=annotations_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for command in commands:
            stage = Path(command[1]).stem
            print(f"Running {stage}", flush=True)
            status = subprocess.run(command, cwd=ROOT, check=False).returncode
            if status:
                if stage == "targets" and status == 1:
                    print(f"Measurements saved; some resume targets remain unmet. See {output_dir / 'targets.md'}")
                else:
                    print(f"Stopped at {stage} (exit {status}); saved progress remains in {output_dir}", file=sys.stderr)
                return status
        print(f"Measurements and target checks saved in {output_dir}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
