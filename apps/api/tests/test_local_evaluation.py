"""Local benchmark orchestration and strict citation grading contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import run_local  # noqa: E402
from evaluation.evaluate_verification import quote_at_exact_lines, score  # noqa: E402


def test_local_commands_keep_stage_order_and_separate_archived_results(tmp_path) -> None:
    commands = run_local.build_commands(
        manifest=tmp_path / "suite.json", annotations_dir=tmp_path / "annotations",
        output_dir=tmp_path / "run", pairs=["a", "b"], prepare=True,
        resume=True, require_reproducible=True,
    )
    assert [Path(c[1]).stem for c in commands] == [
        "prepare_benchmark", "evaluate_retrieval", "evaluate_verification",
        "evaluate_safety", "report", "targets",
    ]
    assert "--require-all-pairs" in commands[2]
    assert "--resume" in commands[2]
    assert "--require-reproducible" in commands[1]
    assert "--require-provenance" in commands[4]
    assert all(str(run_local.EVAL / "results") not in c for c in commands)


@pytest.fixture
def local_suite(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "a.json").write_text('{"pair_id":"a"}')
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"pair_ids": ["a"]}))
    return ["--manifest", str(manifest), "--annotations-dir", str(annotations),
            "--output-dir", str(tmp_path / "run")]


def test_local_run_stops_before_further_inference_when_a_stage_fails(monkeypatch, local_suite) -> None:
    calls = []
    monkeypatch.setattr(run_local, "preflight", lambda **_: None)

    def execute(command, **_):
        calls.append(Path(command[1]).stem)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(run_local.subprocess, "run", execute)
    assert run_local.main(local_suite) == 3
    assert calls == ["evaluate_retrieval"]


def test_unmet_targets_preserve_all_completed_measurements(monkeypatch, local_suite) -> None:
    calls = []
    monkeypatch.setattr(run_local, "preflight", lambda **_: None)

    def execute(command, **_):
        stage = Path(command[1]).stem
        calls.append(stage)
        return SimpleNamespace(returncode=1 if stage == "targets" else 0)

    monkeypatch.setattr(run_local.subprocess, "run", execute)
    assert run_local.main(local_suite) == 1
    assert calls == ["evaluate_retrieval", "evaluate_verification", "evaluate_safety", "report", "targets"]


def test_dry_run_needs_no_server_and_creates_no_outputs(monkeypatch, local_suite, tmp_path) -> None:
    def unexpected(**_):
        pytest.fail("dry run must not contact models or the database")

    monkeypatch.setattr(run_local, "preflight", unexpected)
    assert run_local.main([*local_suite, "--dry-run"]) == 0
    assert not (tmp_path / "run").exists()


def test_resume_requires_explicit_previous_output() -> None:
    with pytest.raises(SystemExit) as exc:
        run_local.main(["--resume"])
    assert exc.value.code == 2


def test_local_run_rejects_overwriting_the_archived_results(local_suite) -> None:
    assert run_local.main([*local_suite, "--output-dir", str(run_local.EVAL / "results")]) == 2


def test_exact_line_grade_has_no_two_line_slack_or_elision() -> None:
    text = "x = 1\ny = 2\nreturn x + y\n"
    assert quote_at_exact_lines("return x + y", text, 3, 3)
    assert not quote_at_exact_lines("return x + y", text, 1, 1)
    assert not quote_at_exact_lines("x = 1 ... return x + y", text, 1, 3)
    assert not quote_at_exact_lines("return x + y", text, 3, 99)
    assert not quote_at_exact_lines("", text, 1, 1)
    assert not quote_at_exact_lines("x", text, 0, 1)
    assert quote_at_exact_lines("x=  1", "x= 1", 1, 1)


def test_legacy_records_do_not_invent_an_exact_line_measurement() -> None:
    record = {"expected": "VERIFIED", "predicted": "VERIFIED", "n_evidence": 1,
              "n_grounded": 1, "n_line_accurate": 1, "error": None, "evidence_hit": True}
    assert score([record])["exact_line_accuracy"] is None
    assert score([{**record, "n_line_exact": 0}])["exact_line_accuracy"] == 0
