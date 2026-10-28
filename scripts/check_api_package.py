#!/usr/bin/env python3
"""Build and smoke-test the distributable API, without editable imports.

Run from any directory with ``python scripts/check_api_package.py``.
The temporary source copy keeps setuptools build artifacts out of the
checkout. A fresh virtual environment installs the wheel and its actual
runtime dependencies, then imports the app and reads all alias resources.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

_SMOKE = """\
import json
import pathlib
import sys
from importlib.resources import files

import claim2code_api
from claim2code_api.app.main import app
from claim2code_api.verification.normalizers import (
    DatasetNormalizer,
    MetricNameComparator,
    ModelNameNormalizer,
)

assert pathlib.Path(claim2code_api.__file__).is_relative_to(sys.prefix)
assert app.title == 'Claim2Code API'
for name in ('models', 'datasets', 'metrics'):
    resource = files('claim2code_api.verification').joinpath('aliases', name + '.json')
    assert json.loads(resource.read_text())
for normalizer in (DatasetNormalizer, MetricNameComparator, ModelNameNormalizer):
    normalizer()
print('Installed wheel: application and alias resources verified')
"""


def main() -> None:
    api = Path(__file__).resolve().parents[1] / "apps" / "api"
    with tempfile.TemporaryDirectory(prefix="claim2code-package-") as directory:
        root = Path(directory)
        source = root / "api"
        shutil.copytree(
            api, source,
            ignore=shutil.ignore_patterns(
                "build", "dist", "*.egg-info", ".venv", "__pycache__",
                ".pytest_cache", ".mypy_cache", ".ruff_cache",
            ),
        )
        wheels = root / "wheels"
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps",
             "--wheel-dir", str(wheels), str(source)],
            check=True,
        )
        wheel, = wheels.glob("claim2code_api-*.whl")
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheel)], check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", _SMOKE],
            cwd=root, check=True,
        )


if __name__ == "__main__":
    main()
