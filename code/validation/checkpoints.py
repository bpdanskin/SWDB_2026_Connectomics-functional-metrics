"""Small, committable verification artifacts.

This port is developed against data that only exists on the Code Ocean capsule, so the
loop is: run something there, commit the artifact it wrote, read it somewhere else. That
only works if the artifacts are small, diffable, and honest about which run produced
them -- a stale JSON that looks fresh is worse than no JSON.

So: JSON for anything to reason about, no arrays, and a provenance stamp on every file.
Per-ROI comparison tables go out as CSV alongside, because at a few thousand rows they are
still a reasonable thing to commit.

This is validation-side. `jsonable` and `git_sha` are not -- the pipeline needs them for
its own provenance record -- so they live in `utils/provenance.py` and are imported here.

    from checkpoints import checkpoint
    checkpoint("schema_report", report, save_dir, seed=0)
"""

import json
import os
from os.path import join as pjoin
from typing import Any, Dict, Optional

import numpy as np

from provenance import git_sha, jsonable

__all__ = ["checkpoint", "CHECKS_SUBDIR"]

CHECKS_SUBDIR = "checks"


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
