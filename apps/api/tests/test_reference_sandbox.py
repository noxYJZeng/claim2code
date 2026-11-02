"""The harness's comparison rules and the sandboxed run, on a throwaway repo."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from claim2code_api.reference.harness import compare
from claim2code_api.reference.sandbox import run_harness

_LOSSES = """\
import numpy as np
from .util import absdiff

def smooth_l1(x, y, beta=1.0):
    d = absdiff(x, y)
    return float(np.mean(np.where(d < beta, 0.5 * d**2 / beta, d - 0.5 * beta)))

def l1_by_mistake(x, y, beta=1.0):
    return float(np.mean(absdiff(x, y)))

def spin(x):
    while True:
        pass

def probe(x):
    import os
    import socket
    try:
        socket.create_connection(("1.1.1.1", 80), timeout=2)
        net = "network"
    except OSError:
        net = "no network"
    try:
        with open(os.path.expanduser("~/../claim2code-escape.txt"), "w") as fh:
            fh.write("x")
        fs = "wrote outside"
    except OSError:
        fs = "no write"
    return [net, fs, os.environ.get("C2C_SECRET", "no secret")]

class Noisy:
    def __init__(self):
        self.w = np.random.rand(3)

    def apply(self, x):
        return self.w * x


class Box:
    calls = 0

    def area(self, w):
        # Fails if the same object were reused across inputs.
        assert self.calls == 0
        self.calls += 1
        return w * w

    @staticmethod
    def scale(x, k):
        return x * k
"""

_REFERENCE = """\
import numpy as np
def reference(x, y, beta=1.0):
    d = np.abs(np.asarray(x, float) - np.asarray(y, float))
    return float(np.mean(np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)))
"""

_INPUTS = """\
def make_inputs():
    return [
        {"args": [[0.0, 2.0], [0.5, 0.0]]},
        {"args": [[0.1], [0.0]], "kwargs": {"beta": 0.5}},
    ]
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    pkg = tmp_path / "repo" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "util.py").write_text(
        "import numpy as np\n"
        "def absdiff(x, y):\n    return np.abs(np.asarray(x) - np.asarray(y))\n",
    )
    (pkg / "losses.py").write_text(_LOSSES)
    return tmp_path / "repo"


def _run(repo: Path, qualname: str, *, reference: str = _REFERENCE, inputs: str = _INPUTS,
         timeout: int = 30):
    return run_harness(
        python=sys.executable, repo_root=repo, module_file="pkg/losses.py", qualname=qualname,
        reference_code=reference, inputs_code=inputs, rtol=1e-6, atol=1e-9,
        timeout_seconds=timeout,
    )


# ---------------------------------------------------------------- compare


def test_reference_reads_a_same_seed_copy_of_the_object(repo: Path) -> None:
    run = run_harness(
        python=sys.executable, repo_root=repo, module_file="pkg/losses.py", qualname="Noisy.apply",
        reference_code="def reference(obj, x):\n    return obj.w * x\n",
        inputs_code="def make_inputs():\n    return [{'args': [2.0]}, {'args': [-1.0]}]\n",
        instance_code="def make_instance():\n    from pkg.losses import Noisy\n    return Noisy()\n",
        rtol=1e-9, atol=0, timeout_seconds=30,
    )
    assert run.report is not None
    assert [c["outcome"] for c in run.report["cases"]] == ["match", "match"]


@pytest.mark.parametrize(("a", "b", "equal"), [
    (1.0, 1.0 + 1e-9, True),
    (1.0, 1.1, False),
    (np.array([1.0, 2.0]), [1.0, 2.0], True),
    (np.zeros((2, 3)), np.zeros((3, 2)), False),
    ((1, [2.0, 3.0]), [1, [2.0, 3.0]], True),
    ({"a": 1.0}, {"b": 1.0}, False),
    (True, 1, False),
    (float("nan"), float("nan"), True),
    ("x", "x", True),
])
def test_compare(a: object, b: object, equal: bool) -> None:
    assert compare(a, b, 1e-6, 1e-9)[0] is equal


def test_compare_reports_the_largest_difference() -> None:
    ok, why = compare(np.array([1.0, 2.0]), np.array([1.0, 2.5]), 1e-6, 1e-9)
    assert not ok
    assert "0.5" in why


# ---------------------------------------------------------------- runs


def test_matching_implementation(repo: Path) -> None:
    run = _run(repo, "smooth_l1")
    assert run.report is not None
    assert [c["outcome"] for c in run.report["cases"]] == ["match", "match"]


def test_wrong_implementation_is_a_mismatch(repo: Path) -> None:
    run = _run(repo, "l1_by_mistake")
    assert run.report is not None
    assert {c["outcome"] for c in run.report["cases"]} == {"mismatch"}


def test_static_method_is_callable(repo: Path) -> None:
    run = _run(
        repo, "Box.scale",
        reference="def reference(x, k):\n    return x * k\n",
        inputs="def make_inputs():\n    return [{'args': [3, 2]}]\n",
    )
    assert run.report is not None
    assert run.report["cases"][0]["outcome"] == "match"


def test_instance_method_needs_make_instance(repo: Path) -> None:
    run = _run(repo, "Box.area")
    assert run.report is not None
    assert run.report["stage"] == "import"
    assert "make_instance" in run.report["error"]


def test_instance_method_runs_on_a_fresh_object(repo: Path) -> None:
    run = run_harness(
        python=sys.executable, repo_root=repo, module_file="pkg/losses.py", qualname="Box.area",
        reference_code="def reference(obj, w):\n    return w * w\n",
        inputs_code="def make_inputs():\n    return [{'args': [3]}, {'args': [4]}]\n",
        instance_code="def make_instance():\n    from pkg.losses import Box\n    return Box()\n",
        rtol=0, atol=0, timeout_seconds=30,
    )
    assert run.report is not None
    assert [c["outcome"] for c in run.report["cases"]] == ["match", "match"]


def test_missing_function_is_an_import_error(repo: Path) -> None:
    run = _run(repo, "does_not_exist")
    assert run.report is not None
    assert run.report["stage"] == "import"


def test_broken_reference_is_a_setup_error(repo: Path) -> None:
    run = _run(repo, "smooth_l1", reference="def reference(:\n")
    assert run.report is not None
    assert run.report["stage"] == "setup"


def test_runaway_code_is_stopped(repo: Path) -> None:
    run = _run(repo, "spin", inputs="def make_inputs():\n    return [{'args': [1]}]\n", timeout=2)
    assert run.report is None
    assert run.timed_out


def test_child_gets_no_environment(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("C2C_SECRET", "hunter2")
    run = _run(repo, "probe", reference="def reference(x):\n    return 0\n",
               inputs="def make_inputs():\n    return [{'args': [1]}]\n")
    assert run.report is not None
    assert "no secret" in run.report["cases"][0]["actual"]


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_macos_sandbox_blocks_network_and_outside_writes(repo: Path) -> None:
    run = _run(repo, "probe", reference="def reference(x):\n    return 0\n",
               inputs="def make_inputs():\n    return [{'args': [1]}]\n")
    assert run.report is not None
    actual = run.report["cases"][0]["actual"]
    assert "'no network'" in actual
    assert "'no write'" in actual
    assert "no network" in run.isolation
