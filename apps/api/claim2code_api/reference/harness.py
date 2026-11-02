"""Differential-test harness, executed in a child process by ``sandbox.py``.

This file is run as a script (``python -I harness.py spec.json``) and
must not import anything from ``claim2code_api``: the child sees only
the standard library, whatever the configured interpreter has
installed, and the cloned repository.

It imports the repository function under test, builds the inputs from
the model-written ``make_inputs``, calls both the repository function
and the model-written ``reference`` on a fresh copy of each input, and
writes one JSON report to ``spec["result_path"]``. Output printed by
the code under test goes to the child's stdout and is ignored.
"""
from __future__ import annotations

import copy
import importlib
import importlib.abc
import importlib.util
import inspect
import json
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Any

_REPR_CAP = 400


def _short(value: Any) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 — repr of arbitrary objects
        text = f"<unprintable {type(value).__name__}: {exc}>"
    return text if len(text) <= _REPR_CAP else text[: _REPR_CAP - 1] + "…"


def _error(exc: BaseException) -> str:
    frames = traceback.format_exception_only(type(exc), exc)
    return "".join(frames).strip()[-_REPR_CAP:]


def _seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- stubs
#
# Research code imports far more than one function needs: a loss in
# trainers.py drags in wandb, transformers and datasets at module level.
# A package that is not installed is replaced by a stub, so the module
# loads and the function under test can run. Numerical libraries are
# never stubbed: without them there is nothing meaningful to compare.

_NEVER_STUB = frozenset({
    "numpy", "torch", "jax", "jaxlib", "tensorflow", "scipy", "pandas", "sklearn",
    "einops", "math", "random",
})
STUBBED: list[str] = []


class _StubMeta(type):
    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_class(f"{cls.__qualname__}.{name}")

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Used as a bare decorator (``@stub``): keep the decorated function.
        if len(args) == 1 and not kwargs and (inspect.isfunction(args[0]) or inspect.isclass(args[0])):
            return args[0]
        return super().__call__(*args, **kwargs)


class _Stub(metaclass=_StubMeta):
    """Stands in for anything from a missing package: callable, subclassable."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_class(f"{type(self).__qualname__}.{name}")()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs and (inspect.isfunction(args[0]) or inspect.isclass(args[0])):
            return args[0]
        return type(self)()

    def __iter__(self) -> Any:
        return iter(())

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<stub {type(self).__qualname__}>"


def _stub_class(qualname: str) -> type:
    cls = _StubMeta(qualname.rsplit(".", 1)[-1], (_Stub,), {})
    cls.__qualname__ = qualname
    return cls


class _StubModule(type(sys)):  # type: ignore[misc]
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_class(f"{self.__name__}.{name}")


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec: Any) -> Any:
        module = _StubModule(spec.name)
        module.__path__ = []  # a package, so submodule imports resolve too
        return module

    def exec_module(self, module: Any) -> None:
        pass


class _StubFinder:
    """Last on ``sys.meta_path``: only names no real finder could locate get here."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        top = name.split(".", 1)[0]
        if top in _NEVER_STUB or top in sys.stdlib_module_names:
            return None
        if top not in STUBBED:
            STUBBED.append(top)
        return importlib.util.spec_from_loader(name, _StubLoader(), is_package=True)


def is_stub(value: Any) -> bool:
    return isinstance(value, _Stub) or (isinstance(value, type) and issubclass(value, _Stub))


# ---------------------------------------------------------------- CUDA on CPU
#
# Research code often hard-codes ``.cuda()`` or ``.to("cuda")``. The test
# inputs are tiny, so on a machine without CUDA those calls are sent to
# the CPU instead; the arithmetic is the same.

_FACTORIES = (
    "tensor", "as_tensor", "zeros", "ones", "empty", "full", "rand", "randn", "randint",
    "randperm", "arange", "linspace", "eye", "zeros_like", "ones_like", "empty_like",
    "full_like", "rand_like", "randn_like",
)


def _cpu_device(device: Any) -> Any:
    text = str(device) if device is not None else ""
    return "cpu" if text.startswith("cuda") or isinstance(device, int) else device


def install_cuda_shim() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if torch.cuda.is_available():
        return False

    def to(original: Any) -> Any:
        def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
            args = tuple(_cpu_device(a) if isinstance(a, str | int | torch.device) else a for a in args)
            if "device" in kwargs:
                kwargs["device"] = _cpu_device(kwargs["device"])
            return original(self, *args, **kwargs)
        return patched

    def factory(original: Any) -> Any:
        def patched(*args: Any, **kwargs: Any) -> Any:
            if "device" in kwargs:
                kwargs["device"] = _cpu_device(kwargs["device"])
            return original(*args, **kwargs)
        return patched

    torch.Tensor.to = to(torch.Tensor.to)  # type: ignore[method-assign]
    torch.nn.Module.to = to(torch.nn.Module.to)  # type: ignore[method-assign]
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    torch.nn.Module.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]
    for name in _FACTORIES:
        if hasattr(torch, name):
            setattr(torch, name, factory(getattr(torch, name)))
    return True


def _to_array(value: Any) -> Any:
    """numpy view of a tensor or array, or ``None`` for anything else."""
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(value, np.ndarray | np.generic):
        return np.asarray(value)
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return None


def compare(actual: Any, expected: Any, rtol: float, atol: float) -> tuple[bool, str]:
    """Structural comparison with numeric tolerance; returns (equal, reason)."""
    a_arr, e_arr = _to_array(actual), _to_array(expected)
    if a_arr is not None or e_arr is not None:
        import numpy as np
        a_arr = np.asarray(actual) if a_arr is None else a_arr
        e_arr = np.asarray(expected) if e_arr is None else e_arr
        if a_arr.shape != e_arr.shape:
            return False, f"shape {a_arr.shape} != {e_arr.shape}"
        try:
            ok = bool(np.allclose(a_arr, e_arr, rtol=rtol, atol=atol, equal_nan=True))
        except TypeError:
            ok = bool(np.array_equal(a_arr, e_arr))
        if ok:
            return True, ""
        try:
            diff = float(np.max(np.abs(a_arr.astype(float) - e_arr.astype(float))))
            return False, f"max abs difference {diff:.3g}"
        except (TypeError, ValueError):
            return False, "values differ"
    if isinstance(actual, bool) or isinstance(expected, bool):
        # True == 1 in Python; a flag and a count are not the same answer.
        ok = isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
        return ok, "" if ok else f"{actual!r} != {expected!r}"
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        if math.isnan(actual) and math.isnan(expected):
            return True, ""
        ok = math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)
        return ok, "" if ok else f"{actual!r} != {expected!r}"
    if isinstance(actual, list | tuple) and isinstance(expected, list | tuple):
        if len(actual) != len(expected):
            return False, f"length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            ok, why = compare(a, e, rtol, atol)
            if not ok:
                return False, f"[{i}]: {why}"
        return True, ""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return False, f"keys {sorted(map(str, actual))} != {sorted(map(str, expected))}"
        for key in actual:
            ok, why = compare(actual[key], expected[key], rtol, atol)
            if not ok:
                return False, f"[{key!r}]: {why}"
        return True, ""
    try:
        ok = bool(actual == expected)
    except Exception as exc:  # noqa: BLE001 — arbitrary __eq__
        return False, f"cannot compare: {_error(exc)}"
    return ok, "" if ok else "values differ"


def _load_target(root: Path, module_file: str, qualname: str) -> Any:
    path = (root / module_file).resolve()
    rel = path.relative_to(root).with_suffix("")
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(root))
    dotted = ".".join(rel.parts)
    try:
        module = importlib.import_module(dotted)
    except ImportError as first:
        spec = importlib.util.spec_from_file_location(rel.name, path)
        if spec is None or spec.loader is None:
            raise first
        module = importlib.util.module_from_spec(spec)
        sys.modules[rel.name] = module
        spec.loader.exec_module(module)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def _needs_instance(target: Any) -> bool:
    params = list(inspect.signature(target).parameters)
    return bool(params) and params[0] == "self"


def _call(fn: Any, case: dict[str, Any]) -> Any:
    args = copy.deepcopy(list(case.get("args", [])))
    kwargs = copy.deepcopy(dict(case.get("kwargs", {})))
    return fn(*args, **kwargs)


def run(spec: dict[str, Any]) -> dict[str, Any]:
    _seed(int(spec.get("seed", 0)))
    rtol, atol = float(spec["rtol"]), float(spec["atol"])
    environment = {"stubbed": STUBBED, "cuda_on_cpu": install_cuda_shim()}
    sys.meta_path.append(_StubFinder())
    try:
        target = _load_target(Path(spec["repo_root"]), spec["module_file"], spec["qualname"])
    except BaseException as exc:  # noqa: BLE001 — repository import side effects
        return {"stage": "import", "error": _error(exc), "cases": [], **environment}

    method = spec["qualname"].rsplit(".", 1)[-1]
    if _needs_instance(target) and not spec.get("instance_code", "").strip():
        return {
            "stage": "import",
            "error": f"{spec['qualname']} is an instance method and no make_instance() was written",
            "cases": [],
            **environment,
        }

    namespace: dict[str, Any] = {"__name__": "claim2code_reference"}
    try:
        exec(spec["inputs_code"], namespace)  # noqa: S102 — runs inside the sandbox
        if _needs_instance(target):
            exec(spec["instance_code"], namespace)  # noqa: S102
        raw_cases = namespace["make_inputs"]()
        exec(spec["reference_code"], namespace)  # noqa: S102
        reference = namespace["reference"]
    except BaseException as exc:  # noqa: BLE001 — model-written code
        return {"stage": "setup", "error": _error(exc), "cases": [], **environment}

    needs_instance = _needs_instance(target)

    def side(fn: Any, case: dict[str, Any], seed: int, *, method_of: bool) -> Any:
        """Call one side on a fresh object built from the same seed.

        Both sides get identical parameters; the target calls the method
        on its object, the reference receives its own copy as ``obj``.
        """
        _seed(seed)
        if not needs_instance:
            return _call(fn, case)
        obj = namespace["make_instance"]()
        _seed(seed)
        if method_of:
            return _call(getattr(obj, method), case)
        return _call(fn, {"args": [obj, *case.get("args", [])], "kwargs": case.get("kwargs", {})})

    cases = []
    for index, raw in enumerate(list(raw_cases)[: int(spec["max_cases"])]):
        case = raw if isinstance(raw, dict) else {"args": list(raw)}
        seed = int(spec.get("seed", 0)) + index
        record: dict[str, Any] = {
            "index": index,
            "inputs": _short({"args": case.get("args", []), "kwargs": case.get("kwargs", {})}),
        }
        try:
            actual = side(target, case, seed, method_of=True)
            record["actual"] = _short(actual)
            if is_stub(actual):
                raise RuntimeError(
                    f"the output comes from a package that is not installed "
                    f"(stubbed: {', '.join(STUBBED)})",
                )
        except BaseException as exc:  # noqa: BLE001
            record["outcome"], record["detail"] = "target_error", _error(exc)
            cases.append(record)
            continue
        try:
            expected = side(reference, case, seed, method_of=False)
            record["expected"] = _short(expected)
        except BaseException as exc:  # noqa: BLE001
            record["outcome"], record["detail"] = "reference_error", _error(exc)
            cases.append(record)
            continue
        ok, why = compare(actual, expected, rtol, atol)
        record["outcome"], record["detail"] = ("match" if ok else "mismatch"), why
        cases.append(record)
    return {"stage": "done", "error": None, "cases": cases, **environment}


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        report = run(spec)
    except BaseException as exc:  # noqa: BLE001 — always leave a report
        report = {"stage": "harness", "error": _error(exc), "cases": []}
    Path(spec["result_path"]).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
