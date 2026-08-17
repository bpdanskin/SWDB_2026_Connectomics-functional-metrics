"""Shared scaffolding for the validation tests.

Deliberately not pytest. These tests need to run on a Code Ocean capsule where the only
reliable entry point is a notebook cell or a bare `python`, and adding a dependency to the
environment to run three hundred lines of asserts is a poor trade. Each file is runnable
on its own:

    python code/validation/tests/test_receptive_fields.py

and exits non-zero if anything failed, so `run_all.py` and CI can both use the exit code.

**Skipping is not failing.** Most of these tests build synthetic data and need nothing
mounted, but a few check the real reference tables. Those raise `SkipTest` and exit 2, so
a laptop run reports them as skipped rather than drowning the real signal in red.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, List

__all__ = ["REPO", "SkipTest", "check", "fails", "load", "require_dataset", "summary"]

#: tests/ -> validation/ -> code/ -> repo root
REPO = Path(__file__).resolve().parents[3]

for _group in ("utils", "validation"):
    _p = str(REPO / "code" / _group)
    if _p not in sys.path:
        sys.path.insert(0, _p)


class SkipTest(Exception):
    """Raised when a test needs data that is not mounted here."""


def _skip_is_not_failure(exc_type, exc, tb):
    """Make `SkipTest -> exit 2` hold wherever it is raised, not just inside `main`.

    These tests are flat scripts, so `require_dataset` is naturally called at module
    scope — above any function `main` could wrap. An uncaught exception there exits 1,
    which `run_all.py` reads as a failure. That is exactly what happened on the first
    reproducible run: the reference tables were not attached to the capsule, the skip
    became a red FAIL, and the validation notebook told the reader not to trust an asset
    that was in fact clean. An excepthook fixes every test at once and cannot be
    forgotten by the next one.
    """
    if isinstance(exc, SkipTest):
        print(f"  SKIP  {exc}")
        # `os._exit` rather than `SystemExit`: the interpreter is already unwinding by the
        # time an excepthook runs, so raising here prints "Error in sys.excepthook" and
        # still exits 1. Flush first -- `os._exit` skips buffer teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _skip_is_not_failure


#: Names of the checks that failed, in order. Tests append via `check`.
fails: List[str] = []


def check(name: str, condition: Any, detail: str = "") -> bool:
    """Record one assertion. Prints either way, so a passing run still reads as evidence."""
    ok = bool(condition)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)
    return ok


def load(module: str):
    """Import a module from `code/utils` or `code/validation` by name, freshly.

    Fresh rather than cached because several of these files rebuild a module after
    monkeypatching it, and a stale `sys.modules` entry would silently test the wrong code.
    """
    for group in ("utils", "validation"):
        path = REPO / "code" / group / f"{module}.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location(module, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"no {module}.py under code/utils or code/validation")


def require_dataset(*names: str) -> str:
    """Resolve a mounted dataset directory, or skip the test.

    The reference-table tests are the only ones that need anything mounted; everything
    else is synthetic. Skipping keeps a laptop run honest instead of red.
    """
    paths = load("paths")
    root = paths.resolve_data_root(names[0])
    got = paths.resolve_dataset_dir(*names, root=root, required=False)
    if got is None:
        raise SkipTest(f"dataset not mounted: {' | '.join(names)} (looked under {root})")
    return got


def summary() -> None:
    """Print the verdict and exit with the code `run_all.py` reads."""
    print()
    print("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}")
    raise SystemExit(1 if fails else 0)


def main(fn: Callable[[], None]) -> None:
    """Run a test body, turning SkipTest into exit code 2."""
    try:
        fn()
    except SkipTest as exc:
        print(f"  SKIP  {exc}")
        raise SystemExit(2)
    summary()
