"""Provenance primitives for the analysis pipeline.

These live on the pipeline side of the split, not with the validation code, because the
pipeline writes its own provenance record: the metrics asset carries a JSON naming the
seed, the config and the package versions that produced it. That record is part of the
deliverable, not part of checking it.

`jsonable` exists because `json.dump` fails on `np.float32` and, worse, will happily write
a bare `NaN` literal that strict parsers reject. Casting up front means an artifact never
fails to write *after* an expensive run.
"""

import os
import subprocess
from typing import Any, Optional

import numpy as np

__all__ = ["jsonable", "git_sha"]


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
    """Best-effort short commit hash; None outside a checkout (e.g. a capsule)."""
    try:
        out = subprocess.run(
            ["git", "-C", repo or os.getcwd(), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
