"""diff_tables / diff_run_dirs — the refactor gate itself.

If this is wrong, every "nothing changed" verdict it produces is worthless, so it is
tested against changes whose size is known exactly.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from harness import check, fails, load, summary

cmp_ = load("compare")

KEYS = ["column", "volume", "plane", "roi"]
RNG = np.random.default_rng(7)
N = 200


def table(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "roi_unique_id": [f"M409828_3_{i // 40}_{i}" for i in range(N)],
        "mouse": "M409828", "column": 1, "volume": "3",
        "plane": np.arange(N) // 40, "roi": np.arange(N),
        "osi": rng.random(N), "dsi": rng.random(N),
        "preferred_dir": rng.choice([0.0, 90.0, 180.0], N),
        "z_score": np.where(rng.random(N) < 0.2, np.nan, rng.normal(size=N)),
    })


print("[1] a table against itself")
a = table()
r = cmp_.diff_tables(a, a.copy())
check("nothing changed", not r["changed"], str(list(r["changed"])))
# All shared non-key columns, identity strings included -- diff_tables does not care
# which are "metrics".
check("every shared non-key column reported identical", len(r["identical"]) == 6,
      str(r["identical"]))
check("all rows joined", r["n_joined"] == N and not r["rows_only_a"] and not r["rows_only_b"])
check("NaNs count as equal to NaNs", "z_score" in r["identical"])

print("\n[2] a known perturbation is located and measured")
b = a.copy()
b.loc[:9, "osi"] += 0.25                      # exactly 10 rows, exactly 0.25
r = cmp_.diff_tables(a, b)
check("only the perturbed column is flagged", list(r["changed"]) == ["osi"],
      str(list(r["changed"])))
d = r["changed"]["osi"]
check("counts the rows", d["n_differing"] == 10, str(d["n_differing"]))
check("measures the size", abs(d["max_abs_diff"] - 0.25) < 1e-12, str(d["max_abs_diff"]))
check("reports the fraction", abs(d["frac_differing"] - 10 / N) < 1e-12)

print("\n[3] atol distinguishes float noise from a real change")
c = a.copy()
c["osi"] = c["osi"] + 1e-12
check("bit-for-bit sees it", "osi" in cmp_.diff_tables(a, c)["changed"])
check("atol=1e-9 does not", "osi" not in cmp_.diff_tables(a, c, atol=1e-9)["changed"])

print("\n[4] NaN appearing or vanishing is a change, and counted separately")
d2 = a.copy()
d2.loc[0, "osi"] = np.nan
r = cmp_.diff_tables(a, d2)
check("flagged", "osi" in r["changed"])
check("nan mismatch counted", r["changed"]["osi"]["n_nan_mismatch"] == 1,
      str(r["changed"]["osi"].get("n_nan_mismatch")))

print("\n[5] schema and row-set changes are reported, not silently joined away")
e = a.copy(); e["depth_um"] = 242.0
r = cmp_.diff_tables(a, e)
check("added column named", r["only_in_b"] == ["depth_um"], str(r["only_in_b"]))
check("adding a column does not mark others changed", not r["changed"])
f = a.iloc[:-5].copy()
r = cmp_.diff_tables(a, f)
check("missing rows surfaced", r["rows_only_a"] == 5, str(r["rows_only_a"]))

print("\n[6] non-numeric columns compare too")
g = a.copy(); g.loc[0, "roi_unique_id"] = "M999999_3_0_0"
r = cmp_.diff_tables(a, g)
check("string change flagged", r["changed"]["roi_unique_id"]["n_differing"] == 1)

print("\n[6b] boolean columns compare without a hand cast")
# Regression: pd.to_numeric leaves a boolean column boolean, and numpy raises outright on
# `True - False`. has_rf_on/off/on_or_off are boolean, so compare_tables coerces
# internally rather than making every caller remember.
bools = a.copy()
bools["has_rf_on"] = np.arange(N) % 3 == 0
bools["has_rf_off"] = np.arange(N) % 2 == 0
other = bools.copy()
other.loc[:4, "has_rf_on"] = ~other.loc[:4, "has_rf_on"]
rep = cmp_.compare_tables(bools, other, ["has_rf_on", "has_rf_off"],
                          exact=["has_rf_on", "has_rf_off"])
check("boolean column compares without raising",
      rep["metrics"]["has_rf_on"]["n_both_finite"] == N)
check("the flipped rows are counted",
      abs(rep["metrics"]["has_rf_on"]["frac_exact"] - (N - 5) / N) < 1e-12,
      str(rep["metrics"]["has_rf_on"]["frac_exact"]))
check("an unchanged boolean column reports perfect agreement",
      rep["metrics"]["has_rf_off"]["frac_exact"] == 1.0)
check("max_abs_diff is 1.0 for a flipped bool",
      rep["metrics"]["has_rf_on"]["max_abs_diff"] == 1.0)

print("\n[7] diff_run_dirs over a directory pair")
root = tempfile.mkdtemp(prefix="diffruns_")
old, new = os.path.join(root, "old"), os.path.join(root, "new")
os.makedirs(old); os.makedirs(new)
a.to_csv(os.path.join(old, "natural_movie_M409828.csv"), index=False)
a.to_csv(os.path.join(old, "rf_metrics_M409828.csv"), index=False)
b.to_csv(os.path.join(new, "natural_movie_M409828.csv"), index=False)
e.to_csv(os.path.join(new, "rf_metrics_M409828.csv"), index=False)
a.to_csv(os.path.join(new, "brand_new_M409828.csv"), index=False)
r = cmp_.diff_run_dirs(old, new)
s = r["summary"]
check("changed file identified", s["files_changed"] == ["natural_movie_M409828.csv"],
      str(s["files_changed"]))
check("schema-only change identified", s["files_schema_changed"] == ["rf_metrics_M409828.csv"],
      str(s["files_schema_changed"]))
check("new file listed, not compared", r["only_in_new"] == ["brand_new_M409828.csv"])
check("summary counts the shared files", s["n_files"] == 2, str(s["n_files"]))

print("\n[8] read_output_csv survives the round trip pandas' default reader does not")
p = os.path.join(root, "precision.csv")
a.to_csv(p, index=False)
loose = pd.read_csv(p, dtype={"volume": str})
exact = cmp_.read_output_csv(p)
check("default parser loses a ULP somewhere",
      not np.array_equal(loose["osi"].to_numpy(), a["osi"].to_numpy()),
      "if this ever passes, pandas fixed it and the note can go")
check("round_trip reader is exact",
      np.array_equal(exact["osi"].to_numpy(), a["osi"].to_numpy()))
check("volume stays a string", isinstance(exact["volume"].iloc[0], str))

shutil.rmtree(root, ignore_errors=True)
summary()
