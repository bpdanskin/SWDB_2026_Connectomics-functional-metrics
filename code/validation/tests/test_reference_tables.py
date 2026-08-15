
from harness import SkipTest, REPO, check, fails, load, require_dataset, summary
import numpy as np, pandas as pd
load("trial_responses"); paths=load("paths"); sm=load("stimulus_metrics")
_cmp = load("compare")


# The only test here that needs anything mounted. `require_dataset` skips rather than
# fails when the reference tables are absent, so a laptop run stays honest.
pub_dir = require_dataset("data_frames")
check("reference tables resolve", pub_dir is not None, str(pub_dir))

print("\n[1] published tables load with usable dtypes")
nm = _cmp.load_reference(pub_dir, "natural_movie")
check("natural_movie rows", len(nm)==163306, str(len(nm)))
check("volume values are str (pandas 2 gives object, pandas 3 gives str)",
      isinstance(nm["volume"].iloc[0], str), f"dtype={nm['volume'].dtype}")
check("volume includes letters", set("abcdef") & set(nm["volume"].unique()) != set(),
      str(sorted(nm['volume'].unique())))
dgf = _cmp.load_reference(pub_dir, "drifting_gratings_full")
check("drifting_gratings_full rows", len(dgf)==164344, str(len(dgf)))

print("\n[2] our two sessions")
TARGET=[(1,"3"),(1,"5")]
mask = [ (int(c),str(v)) in TARGET for c,v in zip(nm["column"], nm["volume"]) ]
ours = nm.loc[mask]
check("rows for col1 vol3+vol5", len(ours)==10021, str(len(ours)))
have = ours["pref_img"].notna() & (ours["pref_img"]>=0)
check("rows carrying metrics == NWB roi count (2708+965)", int(have.sum())==3673, str(int(have.sum())))
per = ours.loc[have].groupby("volume").size()
check("per-volume split matches NWB (vol3=2708, vol5=965)",
      per.get("3")==2708 and per.get("5")==965, dict(per))
check("(column,volume,plane,roi) is unique", not ours.duplicated(["column","volume","plane","roi"]).any())
check("roi_unique_id is NOT unique (drops column) -- do not join on it",
      nm["roi_unique_id"].nunique() < len(nm), f"{nm['roi_unique_id'].nunique()} of {len(nm)}")

print("\n[3] compare_to_published against a perfect copy")
new = ours.loc[have].copy()
rep = _cmp.compare_tables(new, nm, ["frac_responsive_trials","lifetime_sparseness",
                                        "pref_img","pref_response","z_score"],
                              exact=["pref_img"])
check("joins exactly the rows we supplied", rep["n_joined"]==3673, str(rep["n_joined"]))
check("published side has the extra ROIs", rep["n_only_published"]==163306-3673)
m=rep["metrics"]
check("identical input -> zero difference",
      all(v["max_abs_diff"]==0.0 for v in m.values()), str({k:v["max_abs_diff"] for k,v in m.items()}))
check("identical input -> r == 1", all(abs(v["pearson_r"]-1)<1e-12 for v in m.values() if "pearson_r" in v))
check("pref_img exact match reported", m["pref_img"]["frac_exact"]==1.0)

print("\n[4] compare detects a perturbation")
bad = new.copy(); bad["z_score"] = bad["z_score"]*1.02
rep2 = _cmp.compare_tables(bad, nm, ["z_score"])
z=rep2["metrics"]["z_score"]
check("nonzero difference surfaced", z["max_abs_diff"]>0, f"{z['max_abs_diff']:.4f}")
check("but correlation stays ~1 (why r alone is not enough)", z["pearson_r"]>0.999)
check("frac_within_tol catches it", z["frac_within_tol"]<0.01, f"{z['frac_within_tol']:.4f}")

print("\n[5] published natural_movie sanity vs the schema report")
check("pref_img range is 0..3599", nm.loc[nm.pref_img>=0,"pref_img"].max()==3599)
fr = ours.loc[have,"frac_responsive_trials"]
denoms = {round(1/d,6) for d in (8,9)}
check("frac_responsive_trials is k/8 or k/9 (matches n_repeats 8-9)",
      bool(np.all([any(abs((v*d)-round(v*d))<1e-6 for d in (8,9)) for v in fr.dropna().unique()[:200]])))

summary()
