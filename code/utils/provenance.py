"""Provenance primitives for the analysis pipeline.

These live on the pipeline side of the split, not with the validation code, because the
pipeline writes its own provenance record: the metrics asset carries a JSON naming the
seed, the config and the package versions that produced it. That record is part of the
deliverable, not part of checking it.

`jsonable` exists because `json.dump` fails on `np.float32` and, worse, will happily write
a bare `NaN` literal that strict parsers reject. Casting up front means an artifact never
fails to write *after* an expensive run.

`run_stamp` and `latest_run` exist because a re-run that overwrites its predecessor
destroys the only thing that can tell you whether a refactor changed the numbers. Writing
each run to its own stamped directory makes "did this change anything?" answerable by
comparing two directories, which is the gate every refactor step here depends on.
"""

import os
import re
import subprocess
import time
from typing import Any, List, Optional

import numpy as np

__all__ = ["jsonable", "git_sha", "run_stamp", "run_dir", "list_runs", "latest_run"]

#: `<name>_YYYY-MM-DD_HH-MM-SS`. Sortable lexicographically, safe on every filesystem,
#: and readable — the same shape the CCM asset already uses.
_STAMP_RE = re.compile(r"^(?P<name>.+)_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def run_stamp(when: Optional[float] = None) -> str:
    """A sortable, filesystem-safe timestamp for one run. Local time, second resolution.

    Call once per run and reuse it: deriving it twice can straddle a second boundary and
    scatter one run's outputs across two directories.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(when))


def run_dir(root: str, name: str, stamp: Optional[str] = None) -> str:
    """Path for this run's outputs, `<root>/<name>_<stamp>`. Does not create it."""
    return os.path.join(root, f"{name}_{stamp or run_stamp()}")


def list_runs(root: str, name: str) -> List[str]:
    """Existing run directories for `name`, oldest first.

    Sorted by the stamp in the directory name rather than by mtime, because copying a
    directory between machines — which is exactly how these artifacts travel — rewrites
    mtimes and would silently reorder them.
    """
    if not os.path.isdir(root):
        return []
    found = []
    for entry in os.listdir(root):
        match = _STAMP_RE.match(entry)
        if match and match.group("name") == name and os.path.isdir(os.path.join(root, entry)):
            found.append((match.group("stamp"), os.path.join(root, entry)))
    return [path for _, path in sorted(found)]


def latest_run(root: str, name: str) -> Optional[str]:
    """The most recent existing run directory for `name`, or None."""
    runs = list_runs(root, name)
    return runs[-1] if runs else None


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and containers to plain JSON types.

    `json.dump` fails on `np.float32` and silently mangles `np.bool_` in some versions;
    casting up front means an artifact never fails to write after an expensive run.
    """
    # float first, and before the passthrough: np.float64 subclasses float, so a
    # passthrough branch would let NaN reach json.dump, which writes a bare `NaN`
    # literal that strict JSON parsers reject.
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None          # JSON has no NaN/Inf
    if obj is None or isinstance(obj, (bool, np.bool_)):
        return bool(obj) if obj is not None else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return str(obj)


def git_sha(repo: Optional[str] = None) -> Optional[str]:
    """The commit this code was built from, or None if nothing can say.

    `SWDB_CODE_VERSION` is consulted first, and taken verbatim, for the same reason
    `metadata.py` does it: a reproducible run copies `code/` without `.git`, so the git
    fallback cannot answer in the one environment a published asset is built in. Without
    the ladder this returned None there, and the 2026-09-01 asset shipped
    `stimulus_metrics_provenance.json` with `git_sha: null` beside a `processing.json`
    carrying a version. Two sidecars in one asset disagreeing about the same fact is worse
    than either answer alone.

    The git fallback stays short, which is what this field has always carried in a
    checkout; the environment value is a full sha, so the two are prefix-compatible rather
    than contradictory.
    """
    env = os.environ.get("SWDB_CODE_VERSION", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", repo or os.getcwd(), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
