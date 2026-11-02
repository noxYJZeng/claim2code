"""Run the differential-test harness against untrusted code.

Both sides of a reference check are code nobody has reviewed: the
cloned repository and the model-written reference and inputs. They run
in a child process with:

  * an empty environment (no API keys, tokens or database URL reach
    the child) and a throwaway working directory as ``HOME``;
  * CPU-time, file-size and (on Linux) address-space limits, plus a
    wall-clock timeout that kills the whole process group;
  * where the platform offers it, no network and writes confined to
    the working directory: ``sandbox-exec`` on macOS, ``unshare -rn``
    (network only) on Linux.

``isolation`` in the result lists what was actually applied, so the
UI never claims a sandbox that was not there. This is containment for
a local tool, not a security boundary for hostile code; the feature is
off unless ``REFERENCE_CHECK_ENABLED`` is set.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HARNESS = Path(__file__).with_name("harness.py")
_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_FILE_SIZE_LIMIT = 64 * 1024 * 1024
_MEMORY_LIMIT = 4 * 1024 * 1024 * 1024
_OUTPUT_CAP = 4000

# Seatbelt profile: everything the interpreter needs to start, minus
# the network and minus writes outside the run directory.
_SEATBELT = """\
(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/tty"))
"""


@dataclass
class SandboxRun:
    """What came back from one harness run."""

    report: dict[str, Any] | None
    isolation: list[str]
    timed_out: bool = False
    returncode: int | None = None
    stderr: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _linux_unshare() -> list[str] | None:
    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    probe = subprocess.run(
        [unshare, "-rn", "true"], capture_output=True, timeout=5, check=False,
    )
    return [unshare, "-rn"] if probe.returncode == 0 else None


def _limits(cpu_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_FILE_SIZE_LIMIT, _FILE_SIZE_LIMIT))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT, _MEMORY_LIMIT))


def run_harness(
    *,
    python: str,
    repo_root: Path,
    module_file: str,
    qualname: str,
    reference_code: str,
    inputs_code: str,
    instance_code: str = "",
    rtol: float,
    atol: float,
    timeout_seconds: int,
    max_cases: int = 12,
    seed: int = 0,
) -> SandboxRun:
    """Copy the harness into a fresh directory and run it there."""
    with tempfile.TemporaryDirectory(prefix="claim2code-ref-") as tmp:
        workdir = Path(tmp).resolve()
        harness = workdir / "harness.py"
        shutil.copyfile(_HARNESS, harness)
        result_path = workdir / "result.json"
        spec_path = workdir / "spec.json"
        spec_path.write_text(json.dumps({
            "repo_root": str(repo_root.resolve()),
            "module_file": module_file,
            "qualname": qualname,
            "reference_code": reference_code,
            "inputs_code": inputs_code,
            "instance_code": instance_code,
            "rtol": rtol,
            "atol": atol,
            "max_cases": max_cases,
            "seed": seed,
            "result_path": str(result_path),
        }), encoding="utf-8")

        isolation = [
            "separate process with an empty environment",
            f"{timeout_seconds}s time limit",
        ]
        command = [python, "-I", str(harness), str(spec_path)]
        if sys.platform == "darwin" and Path(_SANDBOX_EXEC).exists():
            command = [_SANDBOX_EXEC, "-p", _SEATBELT.format(workdir=workdir), *command]
            isolation += ["no network", "writes limited to a temporary directory"]
        elif sys.platform.startswith("linux"):
            isolation.append("memory limit")
            prefix = _linux_unshare()
            if prefix is not None:
                command = [*prefix, *command]
                isolation.append("no network")

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "CUDA_VISIBLE_DEVICES": "",
        }
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            command,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=lambda: _limits(timeout_seconds),  # noqa: PLW1509
        )
        timed_out = False
        try:
            _, stderr = proc.communicate(timeout=timeout_seconds + 5)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, stderr = proc.communicate()
        # The CPU limit ends a busy loop before the wall clock does.
        timed_out = timed_out or proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL)

        report = None
        if result_path.is_file():
            try:
                report = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
        return SandboxRun(
            report=report,
            isolation=isolation,
            timed_out=timed_out,
            returncode=proc.returncode,
            stderr=stderr.decode("utf-8", errors="replace")[-_OUTPUT_CAP:],
        )


__all__ = ["SandboxRun", "run_harness"]
