"""Comparing a regenerated metrics table against a reference one.

The reference is the historical `data_frames/*_M409828.csv` set. Nothing in the pipeline
imports this module: the pipeline produces tables, this checks them, and keeping the
dependency one-directional is what stops validation-only concerns leaking into the
analysis code.

Two rules that are easy to get wrong and expensive to debug:

* **Join on `(column, volume, plane, roi)`, never `roi_unique_id`.** That string is
  `M{mouse}_{volume}_{plane}_{roi}` with the column dropped, so it collides across
  columns -- 56,449 distinct ids for 164,345 reference rows.
* **`volume` is a string.** Volumes run 1-9 and a-f, so `int` crashes on the 3p sessions,
  and a CSV round-trip silently re-infers `int` for an all-numeric column.

`agreement_table` is the shape that actually gets read: agreement with the reference
*beside* agreement between two seeds of this pipeline. The second column is the noise
floor. Without it, every stochastic metric looks broken.
"""

import os
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

__all__ = ["load_reference", "compare_tables", "agreement_table"]


def load_reference(reference_dir: str, family: str, mouse: str = "M409828") -> pd.DataFrame:
    """Read a reference metrics CSV with the dtypes it actually needs.

    `volume` must be read as a string: volumes run 1..9 and a..f, so `int` crashes on the
    3p sessions. Pandas also warns about mixed types in that column without an explicit
    dtype.
    """
    import os

    path = os.path.join(reference_dir, f"{family}_{mouse}.csv")
    df = pd.read_csv(path, dtype={"volume": str})
    return df.astype({"column": int, "plane": int, "roi": int})


def compare_tables(
    new: pd.DataFrame,
    reference: pd.DataFrame,
    metrics: Sequence[str],
    keys: Sequence[str] = ("column", "volume", "plane", "roi"),
    exact: Sequence[str] = (),
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Per-metric agreement between a regenerated table and the published one.

    Joins on `(column, volume, plane, roi)`, never on `roi_unique_id`, which omits the
    column and collides. Reports set differences explicitly, because the NWB asset is
    ROI-filtered while the published tables carry every segmented ROI, so an inner join
    is expected to drop rows on the published side.

    `exact` names categorical columns (preferred direction, preferred image) compared by
    equality rather than correlation.
    """
    keys = list(keys)
    left = new.copy()
    right = reference.copy()
    for frame in (left, right):
        frame["volume"] = frame["volume"].astype(str)
        for k in ("column", "plane", "roi"):
            frame[k] = frame[k].astype(int)

    merged = left.merge(right, on=keys, how="inner", suffixes=("_new", "_pub"))
    only_new = len(left) - len(merged)
    only_pub = len(right) - len(merged)

    report: Dict[str, Any] = {
        "n_new": int(len(left)),
        "n_published": int(len(right)),
        "n_joined": int(len(merged)),
        "n_only_new": int(only_new),
        "n_only_published": int(only_pub),
        "metrics": {},
    }

    for m in metrics:
        a = pd.to_numeric(merged.get(f"{m}_new"), errors="coerce")
        b = pd.to_numeric(merged.get(f"{m}_pub"), errors="coerce")
        if a is None or b is None:
            report["metrics"][m] = {"error": "column missing on one side"}
            continue
        both = a.notna() & b.notna()
        entry: Dict[str, Any] = {
            "n_both_finite": int(both.sum()),
            "n_new_only_finite": int((a.notna() & ~b.notna()).sum()),
            "n_pub_only_finite": int((~a.notna() & b.notna()).sum()),
        }
        if both.sum():
            av, bv = a[both].to_numpy(), b[both].to_numpy()
            diff = np.abs(av - bv)
            entry.update({
                "max_abs_diff": float(diff.max()),
                "median_abs_diff": float(np.median(diff)),
                "p95_abs_diff": float(np.percentile(diff, 95)),
                "frac_within_tol": float(np.mean(diff <= tol)),
                "tol": tol,
            })
            if m in exact:
                # Relative, not bitwise. `preferred_sf` carries a float32 round-trip
                # (0.04 arrives as 0.039999999...), so two values agreeing to 16 digits
                # can still fail `==` and report 0% agreement on a column that is in
                # fact identical. A genuine category change is a 100% difference, so
                # nothing real hides under this tolerance.
                entry["frac_exact"] = float(np.mean(np.isclose(av, bv, rtol=1e-9, atol=0)))
                entry["frac_exact_bitwise"] = float(np.mean(av == bv))
            if both.sum() > 2 and np.std(av) > 0 and np.std(bv) > 0:
                entry["pearson_r"] = float(np.corrcoef(av, bv)[0, 1])
        report["metrics"][m] = entry

    return report


def agreement_table(new_a, new_b, reference, metrics, exact=()):
    """One row per metric: how well it matches the reference, and how well it matches
    itself under a different random seed. The second column is the noise floor.

    A metric is only worth investigating when the reference column is materially worse
    than the seed column -- the seed column is what this pipeline achieves against itself,
    so it is the floor, not zero.
    """
    vs_ref = compare_tables(new_a, reference, metrics, exact=exact)
    vs_seed = compare_tables(new_a, new_b, metrics, exact=exact)
    rows = []
    for m in metrics:
        p, s = vs_ref["metrics"][m], vs_seed["metrics"][m]
        rows.append({
            "metric": m,
            "n": p.get("n_both_finite"),
            "vs_published_median": p.get("median_abs_diff"),
            "vs_seed_median": s.get("median_abs_diff"),
            "vs_published_max": p.get("max_abs_diff"),
            "vs_seed_max": s.get("max_abs_diff"),
            "r_published": p.get("pearson_r"),
            "exact_published": p.get("frac_exact"),
        })
    return pd.DataFrame(rows), vs_ref, vs_seed
