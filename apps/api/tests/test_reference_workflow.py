"""Reference-check workflow and router, with the model and the database stubbed.

The sandbox is real: each test that gets past target selection runs
the harness on a throwaway repository.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from claim2code_api.app.config import get_settings
from claim2code_api.app.deps import get_db
from claim2code_api.app.main import create_app
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence
from claim2code_api.reference import workflow as workflow_mod
from claim2code_api.reference.exceptions import (
    ReferenceCheckLLMError,
    ReferenceCheckNotFoundError,
)
from claim2code_api.reference.sandbox import SandboxRun
from claim2code_api.reference.schemas import (
    ReferenceCheckStatus,
    ReferenceImplementation,
    ReferenceInputs,
)
from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind, RetrieveMatch

_QUOTE = "The loss is quadratic below beta and linear above it."
_REFERENCE = """\
import numpy as np
def reference(x, y, beta=1.0):
    d = np.abs(np.asarray(x, float) - np.asarray(y, float))
    return float(np.mean(np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)))
"""
_INPUTS = "def make_inputs():\n    return [{'args': [[0.0, 2.0], [0.5, 0.0]]}, {'args': [[3.0], [0.0]]}]\n"


@dataclass
class _State:
    analysis_id: UUID
    claims: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    reference_checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Repo:
    repository_id: UUID
    clone_path: str


def _claim() -> Claim:
    return Claim(
        claim_id="c1", claim_type=ClaimType.METHOD, text="Smooth L1 loss with threshold beta",
        confidence=0.9,
        paper_evidence=PaperEvidence(
            section_title="Method", page=3, char_start=0, char_end=len(_QUOTE), quote=_QUOTE,
        ),
    )


def _db(state: _State | None, repo: _Repo | None) -> MagicMock:
    db = MagicMock()
    answers = [state, repo]

    def _query(_model: Any) -> MagicMock:
        q = MagicMock()
        q.filter.return_value.one_or_none.return_value = answers.pop(0) if answers else None
        return q

    db.query.side_effect = _query
    return db


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "losses.py").write_text(
        "import numpy as np\n"
        "def smooth_l1(x, y, beta=1.0):\n"
        "    d = np.abs(np.asarray(x) - np.asarray(y))\n"
        "    return float(np.mean(np.where(d < beta, 0.5 * d**2 / beta, d - 0.5 * beta)))\n"
        "def plain_l1(x, y, beta=1.0):\n"
        "    return float(np.mean(np.abs(np.asarray(x) - np.asarray(y))))\n",
    )
    return root


def _target(name: str) -> RetrieveMatch:
    return RetrieveMatch(
        symbol_id=uuid4(), kind=CodeSymbolKind.FUNCTION, language=CodeLanguage.PYTHON,
        file_path="losses.py", start_line=2, start_col=0, end_line=4, end_col=0,
        name=name, signature="(x, y, beta=1.0)", docstring=None, similarity=1.0,
        source="def ...", class_context=None, callee_context=None, context_spans=[],
    )


_WRONG_REFERENCE = "def reference(x, y, beta=1.0):\n    return -1.0\n"


class _Client:
    def __init__(
        self, *, testable: bool = True, fail: bool = False, references: list[str] | None = None,
        inputs: list[str] | None = None, property_code: str = "",
    ) -> None:
        self.property_code = property_code
        self.testable, self.fail = testable, fail
        self.references = list(references or [_REFERENCE, _REFERENCE])
        self.inputs = list(inputs or [_INPUTS, _INPUTS])
        self.payloads: list[str] = []

    def reference_inputs(self, *, system: str, payload: str) -> ReferenceInputs:
        self.payloads.append(payload)
        if self.fail:
            raise ReferenceCheckLLMError("LLM returned 500: boom")
        return ReferenceInputs(
            testable=self.testable, reason="It reads a dataset from disk." if not self.testable else "pure",
            inputs_code=self.inputs.pop(0) if self.testable else "", instance_code="",
            input_notes="two float lists",
            test_kind="property" if self.property_code else "reference",
            property_statement="Scaling x, y and beta by 2 doubles the loss" if self.property_code else "",
            property_code=self.property_code, rtol=1e-6, atol=1e-9,
        )

    def reference_implementation(self, *, system: str, payload: str) -> ReferenceImplementation:
        self.payloads.append(payload)
        return ReferenceImplementation(
            reference_code=self.references.pop(0), rtol=1e-6, atol=1e-9,
            assumptions=["mean reduction"],
        )


def _setup(
    monkeypatch: pytest.MonkeyPatch, repo_dir: Path, *, name: str = "smooth_l1",
    client: _Client | None = None, target: bool = True,
) -> tuple[MagicMock, _State, _Client]:
    state = _State(analysis_id=uuid4(), claims=[json.loads(_claim().model_dump_json())])
    db = _db(state, _Repo(repository_id=uuid4(), clone_path=str(repo_dir)))
    client = client or _Client()
    monkeypatch.setattr(workflow_mod, "get_llm_client", lambda: client)
    monkeypatch.setattr(
        workflow_mod, "_pick_target",
        lambda *a, **k: (_target(name), name) if target else None,
    )
    monkeypatch.setenv("REFERENCE_CHECK_PYTHON", sys.executable)
    get_settings.cache_clear()
    return db, state, client


def test_matching_function(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, client = _setup(monkeypatch, repo_dir)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.MATCH
    assert [c.outcome.value for c in check.cases] == ["match", "match"]
    assert check.assumptions == ["mean reduction"]
    assert state.reference_checks[0]["status"] == "MATCH"


def test_reference_author_never_sees_the_body(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, client = _setup(monkeypatch, repo_dir)
    workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    inputs_payload, reference_payload = client.payloads[:2]
    assert "source:" in inputs_payload
    assert "source:" not in reference_payload
    assert _QUOTE in reference_payload
    assert "def reference(x, y, beta=1.0)" in reference_payload


def test_wrong_function_is_a_confirmed_mismatch(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, client = _setup(monkeypatch, repo_dir, name="plain_l1")
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.MISMATCH
    assert check.confirmed is True
    assert check.explanation.startswith("2 of 2 inputs gave a different output from plain_l1")
    assert "second reference, written separately, also" in check.explanation
    assert check.second_reference_code == _REFERENCE
    assert len(client.payloads) == 3  # inputs, reference, second reference


def test_a_wrong_reference_is_overruled(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    client = _Client(references=[_WRONG_REFERENCE, _REFERENCE])
    db, state, _ = _setup(monkeypatch, repo_dir, client=client)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.MATCH
    assert check.confirmed is False
    assert check.reference_code == _REFERENCE
    assert check.second_reference_code == _WRONG_REFERENCE
    assert "the first is taken to be wrong" in check.explanation


def test_an_unconfirmed_mismatch_says_so(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    client = _Client(references=[_WRONG_REFERENCE, "def reference(:\n"])
    db, state, _ = _setup(monkeypatch, repo_dir, client=client)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.MISMATCH
    assert check.confirmed is False
    assert "second reference could not confirm" in check.explanation


def test_broken_test_code_gets_one_repair(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    broken = "def make_inputs():\n    raise ValueError('shapes do not fit')\n"
    client = _Client(inputs=[broken, _INPUTS])
    db, state, _ = _setup(monkeypatch, repo_dir, client=client)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.MATCH
    repair_payload = client.payloads[2]
    assert "# Your previous attempt failed" in repair_payload
    assert "shapes do not fit" in repair_payload


def test_repair_is_tried_only_once(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    broken = "def make_inputs():\n    raise ValueError('still broken')\n"
    client = _Client(inputs=[broken, broken])
    db, state, _ = _setup(monkeypatch, repo_dir, client=client)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.ERROR
    assert "still broken" in check.explanation
    assert len(client.payloads) == 4


_HOMOGENEOUS = (
    "def check(f, x, y, beta=1.0):\n"
    "    import numpy as np\n"
    "    return f(np.multiply(x, 2), np.multiply(y, 2), beta * 2), 2 * f(x, y, beta)\n"
)


def test_property_that_holds(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    client = _Client(property_code=_HOMOGENEOUS)
    db, state, _ = _setup(monkeypatch, repo_dir, client=client)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.MATCH
    assert check.test_kind.value == "property"
    assert check.explanation.startswith("The property held on all 2 inputs: Scaling x, y and beta")
    assert check.reference_code == ""
    assert check.property_code == _HOMOGENEOUS
    assert len(client.payloads) == 1  # no reference is written for a property


def test_property_that_fails(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    # Without scaling beta the quadratic region does not scale linearly.
    broken = _HOMOGENEOUS.replace("beta * 2", "beta")
    client = _Client(property_code=broken)
    db, state, _ = _setup(monkeypatch, repo_dir, client=client)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.MISMATCH
    assert check.confirmed is None
    assert "read check()" in check.explanation


def test_a_match_needs_no_second_reference(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, client = _setup(monkeypatch, repo_dir)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.confirmed is None
    assert len(client.payloads) == 2


def test_rerun_replaces_the_entry(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, _ = _setup(monkeypatch, repo_dir)
    state.reference_checks = [{"claim_id": "c1", "status": "ERROR"}, {"claim_id": "c2"}]
    workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert [e["claim_id"] for e in state.reference_checks] == ["c2", "c1"]


def test_untestable_when_model_says_so(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, _ = _setup(monkeypatch, repo_dir, client=_Client(testable=False))
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.UNTESTABLE
    assert check.explanation == "It reads a dataset from disk."
    assert check.cases == []


def test_untestable_without_a_callable_target(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, _ = _setup(monkeypatch, repo_dir, target=False)
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.UNTESTABLE
    assert "nothing to call" in check.explanation


def test_model_failure_is_an_error_result(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, _ = _setup(monkeypatch, repo_dir, client=_Client(fail=True))
    check = workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="c1")
    assert check.status == ReferenceCheckStatus.ERROR
    assert "boom" in check.explanation


def test_unknown_claim(monkeypatch: pytest.MonkeyPatch, repo_dir: Path) -> None:
    db, state, _ = _setup(monkeypatch, repo_dir)
    with pytest.raises(ReferenceCheckNotFoundError):
        workflow_mod.run_reference_check(db=db, analysis_id=state.analysis_id, claim_id="nope")


# ---------------------------------------------------------------- summarise


def _summary(report: dict[str, Any] | None, **kw: Any) -> tuple[str, str]:
    status, text, _ = workflow_mod.summarise(
        SandboxRun(report=report, isolation=[], **kw), "f", 60,
    )
    return status.value, text


def test_summarise_counts_errors_apart_from_comparisons() -> None:
    status, text = _summary({"stage": "done", "cases": [
        {"index": 0, "inputs": "a", "outcome": "match"},
        {"index": 1, "inputs": "b", "outcome": "target_error", "detail": "ValueError"},
    ]})
    assert status == "MATCH"
    assert text == (
        "All 1 input gave the same output from f and from the reference written from the paper. "
        "Raised an error and not compared: 1 input in f."
    )


def test_summarise_errors() -> None:
    assert _summary(None, timed_out=True) == ("ERROR", "The run did not finish within 60s.")
    assert _summary({"stage": "import", "error": "ModuleNotFoundError: torch", "cases": []}) == (
        "ERROR", "Could not import f: ModuleNotFoundError: torch",
    )
    status, _ = _summary({"stage": "done", "cases": [
        {"index": 0, "inputs": "a", "outcome": "reference_error", "detail": "NameError"},
    ]})
    assert status == "ERROR"


# ---------------------------------------------------------------- environment


def test_third_party_imports_skip_stdlib_and_repo_modules(tmp_path: Path) -> None:
    (tmp_path / "util").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "layers.py").write_text("")
    (tmp_path / "models" / "net.py").write_text(
        "from __future__ import annotations\nimport os, math\nimport torch.nn as nn\n"
        "from timm.models import Block\nfrom util.misc import x\nfrom . import layers\n"
        "import layers\nimport wandb\n",
    )
    assert workflow_mod.third_party_imports(tmp_path, "models/net.py") == ["torch", "timm", "wandb"]


def test_environment_is_described_to_the_inputs_author() -> None:
    from claim2code_api.reference.prompts import describe_environment
    text = describe_environment({"torch": "2.11.0", "timm": None, "__cuda__": None})
    assert "installed: torch 2.11.0" in text
    assert "replaced by stubs: timm" in text
    assert "run on the CPU" in text


def test_environment_notes() -> None:
    assert workflow_mod.environment_notes({"stubbed": ["timm"], "cuda_on_cpu": True}) == [
        "Not installed, replaced by stubs: timm", "CUDA calls ran on the CPU",
    ]
    assert workflow_mod.environment_notes(None) == []


# ---------------------------------------------------------------- target choice


def test_reference_signature_renames_self() -> None:
    from claim2code_api.reference.prompts import reference_signature
    method = _target("forward_loss").model_copy(update={"signature": "(self, imgs, pred, mask)"})
    assert reference_signature(method) == "(obj, imgs, pred, mask)"
    assert reference_signature(method.model_copy(update={"signature": "(self)"})) == "(obj)"
    assert reference_signature(_target("f")) == "(x, y, beta=1.0)"


def _sym(**kw: Any) -> MagicMock:
    sym = MagicMock()
    sym.language, sym.kind, sym.file_path = "python", "function", "m.py"
    sym.parent_symbol_id, sym.name, sym.signature = None, "f", "(x)"
    for k, v in kw.items():
        setattr(sym, k, v)
    return sym


def test_qualname_rules() -> None:
    cls = _sym(kind="class", name="Box")
    db = MagicMock()
    db.get.return_value = cls
    assert workflow_mod._qualname(db, _sym()) == "f"
    assert workflow_mod._qualname(db, _sym(language="javascript", file_path="m.js")) is None
    assert workflow_mod._qualname(db, _sym(kind="class")) is None
    method = {"kind": "method", "parent_symbol_id": uuid4(), "name": "scale"}
    assert workflow_mod._qualname(db, _sym(**method, signature="(x, k)")) == "Box.scale"
    assert workflow_mod._qualname(db, _sym(**method, signature="(self, x)")) == "Box.scale"
    db.get.return_value = _sym(kind="function", name="outer")
    assert workflow_mod._qualname(db, _sym(parent_symbol_id=uuid4())) is None


# ---------------------------------------------------------------- router


def _client_with(state: _State | None) -> TestClient:
    app = create_app()
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = state
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def test_post_is_refused_while_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFERENCE_CHECK_ENABLED", raising=False)
    get_settings.cache_clear()
    res = _client_with(None).post(f"/analyses/{uuid4()}/reference-checks", json={"claim_id": "c1"})
    assert res.status_code == 403
    assert "REFERENCE_CHECK_ENABLED" in res.json()["detail"]


def test_get_reports_enabled_and_skips_bad_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFERENCE_CHECK_ENABLED", "true")
    get_settings.cache_clear()
    good = {
        "claim_id": "c1", "status": "MATCH", "explanation": "ok",
        "generated_at": "2026-11-02T00:00:00Z",
    }
    state = _State(analysis_id=uuid4(), reference_checks=[good, {"claim_id": "broken"}])
    res = _client_with(state).get(f"/analyses/{state.analysis_id}/reference-checks")
    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True
    assert [c["claim_id"] for c in body["checks"]] == ["c1"]


def test_get_unknown_analysis() -> None:
    res = _client_with(None).get(f"/analyses/{uuid4()}/reference-checks")
    assert res.status_code == 404
