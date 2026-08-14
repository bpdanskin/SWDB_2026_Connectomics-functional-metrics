"""Small, committable verification artifacts.

This port is developed against data that only exists on the Code Ocean capsule, so the
loop is: run a milestone there, commit the artifact it wrote, read it somewhere else.
That only works if the artifacts are small, diffable, and honest about which run produced
them — a stale JSON that looks fresh is worse than no JSON.

So: JSON for anything to reason about, no arrays, and a provenance stamp on every file.
Per-ROI comparison tables go out as CSV alongside, because at a few thousand rows they
are still a reasonable thing to commit.

    from checkpoints import checkpoint
    checkpoint("schema_report", report, save_dir, seed=0)
"""

import json
import os
import subprocess
from os.path import join as pjoin
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["jsonable", "git_sha", "checkpoint", "CHECKS_SUBDIR"]

CHECKS_SUBDIR = "checks"


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


def checkpoint(
    name: str,
    payload: Dict[str, Any],
    save_dir: str,
    *,
    seed: Optional[int] = None,
    sessions: Optional[Any] = None,
    wall_seconds: Optional[float] = None,
    print_summary: bool = True,
) -> str:
    """Write `save_dir/checks/{name}.json` and print a one-line summary.

    The provenance block is not optional decoration: without it there is no way to tell,
    reading the file later, whether it came from the run you think it did.
    """
    out_dir = pjoin(save_dir, CHECKS_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    path = pjoin(out_dir, f"{name}.json")

    body = {
        "_provenance": jsonable({
            "name": name,
            "seed": seed,
            "sessions": sessions,
            "wall_seconds": wall_seconds,
            "git_sha": git_sha(),
            "numpy": np.__version__,
        }),
        **jsonable(payload),
    }
    with open(path, "w", encoding="utf-8") as fh:
        # allow_nan=False so a value that slipped past `jsonable` raises here rather
        # than writing a file that looks fine and fails to parse later.
        json.dump(body, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")

    if print_summary:
        size_kb = os.path.getsize(path) / 1024
        print(f"  wrote {path}  ({size_kb:.1f} KB)")
    return path
