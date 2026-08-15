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

import glob
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["load_reference", "compare_tables", "agreement_table",
           "read_output_csv", "diff_tables", "diff_run_dirs"]

KEYS = ["column", "volume", "plane", "roi"]


def read_output_csv(path: str) -> pd.DataFrame:
    """Read one of this pipeline's CSVs back exactly as written.

    Two settings that are not optional. `volume` as a string, because volumes run 1-9 and
    a-f and a CSV round-trip re-infers `int` for an all-numeric column. And
    `float_precision="round_trip"`, because pandas' default C parser is **off by one ULP**
    — comparing a file against itself through the default reader reports roughly a third
    of float values as changed.
    """
    return pd.read_csv(path, dtype={"volume": str}, float_precision="round_trip")


def diff_tables(a: pd.DataFrame, b: pd.DataFrame, keys: Sequence[str] = KEYS,
                atol: float = 0.0) -> Dict[str, Any]:
    """Column-level differences between two versions of the same table.

    Answers the question a refactor gate actually asks — *which* columns moved and by how
    much — rather than "are these files identical", which tells you nothing about where to
    look. `atol=0` means bit-for-bit; raise it when a change is expected to be numerical.
    """
    keys = list(keys)
    report: Dict[str, Any] = {
        "n_a": int(len(a)), "n_b": int(len(b)),
        "only_in_a": [c for c in a.columns if c not in b.columns],
        "only_in_b": [c for c in b.columns if c not in a.columns],
        "identical": [], "changed": {},
    }
    # Normalise the join keys before merging. Two readings of the same CSV can disagree on
    # integer width -- a scalar column comes back int64 while an arange-derived one is
    # int32 on Windows -- and pandas raises "Buffer dtype mismatch" rather than joining.
    a, b = a.copy(), b.copy()
    for frame in (a, b):
        if "volume" in frame:
            frame["volume"] = frame["volume"].astype(str)
        for k in ("column", "plane", "roi"):
            if k in frame:
                frame[k] = frame[k].astype("int64")

    merged = a.merge(b, on=keys, how="outer", suffixes=("__a", "__b"), indicator=True)
    report["n_joined"] = int((merged["_merge"] == "both").sum())
    report["rows_only_a"] = int((merged["_merge"] == "left_only").sum())
    report["rows_only_b"] = int((merged["_merge"] == "right_only").sum())

    for column in [c for c in a.columns if c in b.columns and c not in keys]:
        x, y = merged[f"{column}__a"], merged[f"{column}__b"]
        numeric = pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
        if numeric:
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            differs = ~(np.isclose(xv, yv, rtol=0, atol=atol, equal_nan=True))
            differs &= ~both_nan
            if not differs.any():
                report["identical"].append(column)
                continue
            d = np.abs(xv[differs] - yv[differs])
            report["changed"][column] = {
                "n_differing": int(differs.sum()),
                "frac_differing": float(differs.mean()),
                "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else None,
                "median_abs_diff": float(np.nanmedian(d)) if np.isfinite(d).any() else None,
                "n_nan_mismatch": int((np.isnan(xv) != np.isnan(yv)).sum()),
            }
        else:
            differs = x.astype(str).to_numpy() != y.astype(str).to_numpy()
            if not differs.any():
                report["identical"].append(column)
            else:
                report["changed"][column] = {"n_differing": int(differs.sum()),
                                             "frac_differing": float(differs.mean()),
                                             "dtype": "non-numeric"}
    return report


def diff_run_dirs(old: str, new: str, keys: Sequence[str] = KEYS,
                  atol: float = 0.0, pattern: str = "*.csv") -> Dict[str, Any]:
    """Compare every table in two run directories.

    Run directories are stamped precisely so this is possible: a re-run that overwrote its
    predecessor would leave nothing to compare, and "did that change the numbers?" would
    become unanswerable rather than merely unanswered.
    """
    def names(d):
        return {os.path.basename(p) for p in glob.glob(os.path.join(d, pattern))}

    shared = sorted(names(old) & names(new))
    report: Dict[str, Any] = {
        "old": old, "new": new,
        "only_in_old": sorted(names(old) - names(new)),
        "only_in_new": sorted(names(new) - names(old)),
        "files": {},
    }
    for name in shared:
        report["files"][name] = diff_tables(
            read_output_csv(os.path.join(old, name)),
            read_output_csv(os.path.join(new, name)), keys=keys, atol=atol)
    report["summary"] = {
        "n_files": len(shared),
        "files_identical": sorted(n for n, r in report["files"].items()
                                  if not r["changed"] and not r["only_in_a"]
                                  and not r["only_in_b"]),
        "files_changed": sorted(n for n, r in report["files"].items() if r["changed"]),
        "files_schema_changed": sorted(n for n, r in report["files"].items()
                                       if r["only_in_a"] or r["only_in_b"]),
    }
    return report


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
    rtol: float = 1e-6,
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
            # int64 explicitly, not `int`: on Windows under numpy 1.x that is int32, and
            # merging an int32 key against an int64 one raises "Buffer dtype mismatch"
            # rather than joining. The capsule is 64-bit Linux and never sees it, which is
            # exactly what makes it worth pinning.
            frame[k] = frame[k].astype("int64")

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
            # float64, not whatever the column happened to be. `pd.to_numeric` leaves a
            # boolean column boolean, and numpy refuses `True - False` outright:
            #     TypeError: numpy boolean subtract ... use bitwise_xor
            # Three of the receptive-field columns are boolean, so every caller comparing
            # them would otherwise have to remember to cast first.
            av = a[both].to_numpy(dtype=np.float64)
            bv = b[both].to_numpy(dtype=np.float64)
            diff = np.abs(av - bv)
            entry.update({
                # Relative agreement, and it is the one to read. `frac_within_tol` uses an
                # ABSOLUTE 1e-9, which on a 0-360 quantity like pref_dir_mean reports
                # near-perfect agreement as ~0 % -- the port agrees with the reference to
                # about 1e-6 relative, which is summation order, not a defect.
                "frac_within_rtol": float(np.mean(np.isclose(av, bv, rtol=rtol, atol=0.0))),
                "rtol": rtol,
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
