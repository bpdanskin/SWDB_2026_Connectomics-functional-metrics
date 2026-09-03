"""The windowed-grating aperture centre: extraction, and imputation from the column.

Two of the 25 sessions record no aperture centre. `probe_window_center.py` established
that the `center_azimuth` / `center_elevation` columns are absent from those sessions'
stimulus tables entirely, so filling from the rest of the cortical column fills a real
absence rather than hiding a defect of ours.

Three things are worth asserting and one of them is easy to get wrong:

1. `window_center` reads the centre from **non-blank rows only**, because that is what
   `drifting_gratings_metrics` reads. A value present only on blank sweeps must read as
   absent, or a bug of ours would look like missing data — which is exactly the mistake
   the probe made before it was corrected.
2. The fill is the **median** of the column's donors. Column 2 / volume 2 sits 0.2 degrees
   off the rest of its column, so equality is the wrong test and the median is the right
   fill.
3. **`impute_dgw_center` must be shown to do something.** `pref_cond_fillna` was declared
   and documented from the start and never read, so flipping it was a silent no-op; the
   rule that came out of that is that a flag needs a test exercising both settings and
   asserting they differ. Section [5] is that test.
"""

from harness import check, fails, load, summary

import numpy as np
import pandas as pd

sm = load("stimulus_metrics")


def trial_frame(az, el, *, n=12, blank_az=None, blank_el=None, n_blank=4,
                columns=("center_azimuth", "center_elevation")):
    """Non-blank rows carrying `(az, el)`, plus blank rows carrying `(blank_az, blank_el)`.

    Returns `(frame, is_blank)` in the shape `stimulus_trials` hands the pipeline.
    """
    rows = []
    for _ in range(n):
        rows.append({"direction": 90.0, "center_azimuth": az, "center_elevation": el})
    for _ in range(n_blank):
        rows.append({"direction": np.nan,
                     "center_azimuth": blank_az, "center_elevation": blank_el})
    frame = pd.DataFrame(rows)
    keep = ["direction"] + [c for c in columns]
    frame = frame[keep]
    is_blank = frame["direction"].isna().to_numpy()
    return frame, is_blank


print("[1] window_center reads what the pipeline reads")
f, blank = trial_frame(1.8, -9.7)
check("recovers a recorded centre", sm.window_center(f.loc[~blank]) == (1.8, -9.7),
      str(sm.window_center(f.loc[~blank])))
f, blank = trial_frame(np.nan, np.nan)
got = sm.window_center(f.loc[~blank])
check("all-NaN reads as absent", not np.isfinite(got[0]) and not np.isfinite(got[1]),
      str(got))
f, blank = trial_frame(1.8, -9.7, columns=())
got = sm.window_center(f.loc[~blank])
check("absent columns read as absent, not as an error",
      not np.isfinite(got[0]) and not np.isfinite(got[1]), str(got))

print("\n[2] a centre on blank sweeps only must NOT read as recorded")
# The probe's original defect, in miniature: it counted these values over every trial
# while the pipeline reads only non-blank ones, so this shape read as healthy and still
# produced NaN downstream.
f, blank = trial_frame(np.nan, np.nan, blank_az=1.8, blank_el=-9.7)
got = sm.window_center(f.loc[~blank])
check("blank-only value is absent to the pipeline",
      not np.isfinite(got[0]) and not np.isfinite(got[1]), str(got))
check("and it IS present if you wrongly look at every row",
      sm.window_center(f) == (1.8, -9.7),
      "this is the count that made the probe say 'recorded'")

print("\n[3] the real asset's shape: 5 columns x 5 volumes, two sessions missing")
# Values from the V1DD white paper / the recovered per-column table. Column 2 / volume 2
# is deliberately 0.2 off, and (2, '5') and (4, '1') are the sessions with no centre.
PER_COLUMN = {1: (-8.9, -12.4), 2: (-19.6, -10.0), 3: (1.8, -9.7),
              4: (-15.4, -16.4), 5: (9.9, -14.4)}
observed = {}
for col in range(1, 6):
    for vol in "12345":
        az, el = PER_COLUMN[col]
        if (col, vol) == (2, "2"):
            az = -19.8                     # retargeting jitter, not a different window
        if (col, vol) in ((2, "5"), (4, "1")):
            az, el = np.nan, np.nan        # the two that record nothing
        observed[(col, vol)] = (az, el)

wc = sm.infer_window_centers(observed)
p = wc.provenance
check("23 measured, 2 inferred, 0 unfilled",
      (p["n_measured"], p["n_inferred"], p["n_unfilled"]) == (23, 2, 0),
      f"{p['n_measured']}/{p['n_inferred']}/{p['n_unfilled']}")
check("column 2 / volume 5 filled to the handoff's -19.6 / -10.0",
      wc.centers[(2, "5")] == (-19.6, -10.0), str(wc.centers[(2, "5")]))
check("column 4 / volume 1 filled to -15.4 / -16.4",
      wc.centers[(4, "1")] == (-15.4, -16.4), str(wc.centers[(4, "1")]))
check("both are flagged inferred",
      wc.inferred[(2, "5")] and wc.inferred[(4, "1")])
check("a measured session is not flagged", not wc.inferred[(3, "3")])
check("a measured session keeps its own value, jitter included",
      wc.centers[(2, "2")] == (-19.8, -10.0), str(wc.centers[(2, "2")]))
check("every session ends up with a key", len(wc.centers) == len(observed) == 25,
      f"{len(wc.centers)} vs {len(observed)}")
check("the three states account for every session",
      p["n_measured"] + p["n_inferred"] + p["n_unfilled"] == p["n_sessions"] == 25,
      f"{p['n_measured']}+{p['n_inferred']}+{p['n_unfilled']} vs {p['n_sessions']}")

print("\n[4] the provenance carries what makes the fill judgeable")
c2 = p["columns"]["2"]
check("column 2 has 4 donors", c2["n_donors"] == 4, str(c2["n_donors"]))
check("and names which volumes donated", c2["donor_volumes"] == ["1", "2", "3", "4"],
      str(c2["donor_volumes"]))
check("records the 0.2 deg spread rather than hiding it",
      abs(c2["spread_azimuth"] - 0.2) < 1e-9, str(c2["spread_azimuth"]))
check("two distinct azimuths in column 2", c2["n_distinct_azimuth"] == 2,
      str(c2["n_distinct_azimuth"]))
check("the median is -19.6, not the -19.8 outlier",
      abs(c2["median_azimuth"] + 19.6) < 1e-9, str(c2["median_azimuth"]))
check("elevation has no spread in column 2", abs(c2["spread_elevation"]) < 1e-9)
check("names the volume it filled", c2["missing_volumes"] == ["5"],
      str(c2["missing_volumes"]))
check("`filled` lists both sessions with their values", len(p["filled"]) == 2,
      str(p["filled"]))
check("a fully-measured column reports nothing missing",
      p["columns"]["1"]["missing_volumes"] == [] and p["columns"]["1"]["n_donors"] == 5)

print("\n[5] impute_dgw_center is a flag that does something")
# The pref_cond_fillna rule: exercise both settings and assert they differ, or a
# documented flag may not be an implemented one.
off = sm.infer_window_centers(observed, config=sm.MetricConfig(impute_dgw_center=False))
check("REFERENCE_CONFIG has it off", sm.REFERENCE_CONFIG.impute_dgw_center is False)
check("the default has it on", sm.DEFAULT_CONFIG.impute_dgw_center is True)
check("off leaves the centre NaN",
      not np.isfinite(off.centers[(2, "5")][0]), str(off.centers[(2, "5")]))
check("off flags nothing as inferred", off.provenance["n_inferred"] == 0,
      str(off.provenance["n_inferred"]))
check("off reports them as unfilled instead", off.provenance["n_unfilled"] == 2,
      str(off.provenance["n_unfilled"]))
check("the two settings genuinely disagree", wc.inferred != off.inferred)
check("provenance says which way it ran",
      wc.provenance["enabled"] is True and off.provenance["enabled"] is False)
check("measured sessions are identical either way",
      all(wc.centers[k] == off.centers[k] for k in observed
          if k not in ((2, "5"), (4, "1"))))

print("\n[6] a column with no donor is left alone, not borrowed from elsewhere")
# The justification is that a column agrees with itself; across columns the positions
# genuinely differ, so borrowing would invent a position the stimulus never occupied.
lonely = sm.infer_window_centers({(7, "1"): (np.nan, np.nan),
                                  (7, "2"): (np.nan, np.nan),
                                  (8, "1"): (5.0, 6.0)})
check("no-donor column stays NaN",
      not np.isfinite(lonely.centers[(7, "1")][0]), str(lonely.centers[(7, "1")]))
check("and is not flagged inferred", not lonely.inferred[(7, "1")])
check("counted as unfilled, so it reads as a gap",
      lonely.provenance["n_unfilled"] == 2, str(lonely.provenance["n_unfilled"]))
check("its provenance shows n_donors 0",
      lonely.provenance["columns"]["7"]["n_donors"] == 0)
# The three states are disjoint and must account for every session. Counting "measured"
# as merely not-inferred reported an unfilled session as one carrying its own centre,
# which a two-session dry run on real data printed as "2 measured" out of 2 with one
# missing.
lp = lonely.provenance
check("measured excludes the unfilled", lp["n_measured"] == 1, str(lp["n_measured"]))
check("measured + inferred + unfilled == n_sessions",
      lp["n_measured"] + lp["n_inferred"] + lp["n_unfilled"] == lp["n_sessions"] == 3,
      f"{lp['n_measured']}+{lp['n_inferred']}+{lp['n_unfilled']} vs {lp['n_sessions']}")
check("median is None rather than nan, so the JSON stays strict",
      lonely.provenance["columns"]["7"]["median_azimuth"] is None)
check("the donor column is untouched", lonely.centers[(8, "1")] == (5.0, 6.0))

print("\n[7] half a centre is not a donor and is not half-filled")
half = sm.infer_window_centers({(9, "1"): (1.0, np.nan), (9, "2"): (2.0, 3.0)})
check("a session with one coordinate is reported",
      len(half.provenance["partial_sessions"]) == 1,
      str(half.provenance["partial_sessions"]))
check("it does not donate", half.provenance["columns"]["9"]["n_donors"] == 1,
      str(half.provenance["columns"]["9"]["n_donors"]))
check("it is filled wholly from the donor, not mixed with its own azimuth",
      half.centers[(9, "1")] == (2.0, 3.0), str(half.centers[(9, "1")]))
check("and flagged inferred, since neither coordinate is its own",
      half.inferred[(9, "1")])

print("\n[8] volume keys stay strings")
# Volumes run 1-9 and a-f project-wide, and a CSV round-trip re-infers int for an
# all-numeric column -- so an int key would silently miss.
lettered = sm.infer_window_centers({(1, "a"): (1.0, 2.0), (1, "b"): (np.nan, np.nan)})
check("a letter volume imputes normally", lettered.centers[(1, "b")] == (1.0, 2.0),
      str(lettered.centers[(1, "b")]))
check("donor volumes are strings in the provenance",
      lettered.provenance["columns"]["1"]["donor_volumes"] == ["a"],
      str(lettered.provenance["columns"]["1"]["donor_volumes"]))
check("column keys in the provenance are strings, so json.dump is happy",
      all(isinstance(k, str) for k in lettered.provenance["columns"]))

print("\n[9] the output schema carries the flag")
check("dgw_center_inferred is a published SSI column",
      "dgw_center_inferred" in sm.OUTPUT_COLUMNS["surround_supression_index"])
ssi_cols = list(sm.OUTPUT_COLUMNS["surround_supression_index"])
check("it sits with the other centre columns",
      ssi_cols.index("dgw_center_inferred") == ssi_cols.index("dgw_center_elevation") + 1)
check("the centre columns precede the containment they gate",
      ssi_cols.index("dgw_center_inferred") < ssi_cols.index("dgw_rf_distance_on"))
frame = pd.DataFrame({c: [0.0] for c in ssi_cols})
frame["dgw_center_inferred"] = [1.0]           # arrives as float from np.full
frame["volume"] = ["3"]
out = sm.to_output_schema(frame, "surround_supression_index")
check("to_output_schema writes it as a real bool, like has_rf_on",
      out["dgw_center_inferred"].dtype == bool, str(out["dgw_center_inferred"].dtype))

summary()
