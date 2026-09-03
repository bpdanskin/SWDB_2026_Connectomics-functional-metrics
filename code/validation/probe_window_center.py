"""Why do two sessions ship no windowed-grating aperture centre?

`dgw_center_azimuth` / `dgw_center_elevation` are NaN for column 2 / volume 5 and
column 4 / volume 1 in the 2026-09-01 asset. Both have complete `ssi` and grating values,
so the stimulus ran — only the recorded position is missing. Before imputing it from the
rest of the column, establish which of these is true:

  a. the column is ABSENT from the NWB stimulus table      -> the export dropped it
  b. the column is present but ALL NaN                     -> nothing was recorded
  c. the column holds values                               -> OUR extraction loses them

(c) would make imputation a way of papering over our own bug, which is the reason to look
before filling anything in.

(c) has two shapes and both are counted separately, because the pipeline does not read
every row it is handed: `stimulus_metrics` takes the centre from `trials.loc[~is_blank]`.
So a centre recorded only on blank sweeps is present in the table, present in what
`stimulus_trials` returns, and still NaN in the asset. Counting over all trials would call
that healthy.

The third session is the control. It was re-filtered on the same date as column 4 /
volume 1 yet DOES carry a centre, so if re-filtering were the cause it should have lost
one too.

Reads both the raw stimulus table and what `stimulus_trials` hands the pipeline, because
those are exactly the two places the value could go missing.

    python code/validation/probe_window_center.py            # the three sessions below
    python code/validation/probe_window_center.py --all      # every session in the asset

Read-only. Seconds per session; it opens the stimulus table and nothing else.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

import numpy as np
import pandas as pd

import v1dd_nwb as vn
from paths import resolve_data_root, resolve_dataset_dir

ASSET = "409828_V1DD_Filtered"
COLUMNS = ["center_azimuth", "center_elevation"]

# name -> why it is in the list
SUSPECTS = {
    "409828_2018-12-07_15-05-38_filtered_2026-08-18_23-10-02":
        "col 2 / vol 5 -- no centre; newest re-filter (2026-08-18); newly coregistered",
    "409828_2018-11-21_09-22-23_filtered_2026-04-16_02-51-11":
        "col 4 / vol 1 -- no centre; re-filtered 2026-04-16; the 67 %-low-confidence one",
    "409828_2018-11-28_10-54-56_filtered_2026-04-16_09-56-00":
        "CONTROL: col 3 / vol 3 -- re-filtered the SAME DAY, and DOES carry a centre",
}


def describe(path):
    out = {"session": os.path.basename(os.path.dirname(str(path)))}
    # `session`, not `open_session`: the latter returns `(nwbfile, io)` and is not a
    # context manager, which is what this script did before -- it had never run.
    # `load_stimulus_table` rather than a bare `to_dataframe()` so the probe reads the
    # table through the same call the pipeline does; asking a differently-shaped question
    # is the whole failure mode this script exists to rule out.
    with vn.session(path) as nwb:
        st = vn.load_stimulus_table(nwb)
        out["stimulus_table_columns"] = list(st.columns)
        for fam in ("drifting_gratings_windowed", "drifting_gratings_full"):
            raw = st[st["stim_name"] == fam]
            trials, blank = vn.stimulus_trials(st, fam, vn.DG_PARAM_COLUMNS)
            fam_out = {"n_rows_raw": int(len(raw)),
                       "n_rows_via_stimulus_trials": int(len(trials)),
                       "n_blank": int(np.sum(blank))}
            for c in COLUMNS:
                if c not in st.columns:
                    fam_out[c] = "COLUMN ABSENT from the stimulus table"
                    continue
                vals = pd.to_numeric(raw[c], errors="coerce")
                via = (pd.to_numeric(trials[c], errors="coerce")
                       if c in trials.columns else pd.Series(dtype=float))
                # `stimulus_metrics` reads the centre off `trials.loc[~is_blank]`, so a
                # count over all trials can say "present" where the pipeline sees NaN --
                # values landing only on blank sweeps. Count where the pipeline looks.
                non_blank = via[~blank] if len(via) == len(blank) else pd.Series(dtype=float)
                fam_out[c] = {
                    "n_non_nan_raw": int(vals.notna().sum()),
                    "n_non_nan_via_stimulus_trials": int(via.notna().sum()),
                    "n_non_nan_non_blank": int(non_blank.notna().sum()),
                    "unique_raw": sorted(vals.dropna().unique().tolist())[:8],
                }
            out[fam] = fam_out
    return out


def verdict(rec):
    w = rec.get("drifting_gratings_windowed", {})
    az = w.get("center_azimuth")
    if isinstance(az, str):
        return "(a) COLUMN ABSENT -- the export dropped it; imputation is the only option"
    if az["n_non_nan_raw"] == 0:
        return "(b) present but ALL NaN -- nothing recorded; imputation is justified"
    if az["n_non_nan_via_stimulus_trials"] == 0:
        return "(c) VALUES EXIST but stimulus_trials loses them -- OUR BUG, fix, do not impute"
    if az["n_non_nan_non_blank"] == 0:
        return ("(c) VALUES EXIST but only on BLANK sweeps, and the pipeline reads "
                "trials.loc[~is_blank] -- OUR BUG, fix, do not impute")
    return (f"has {az['n_non_nan_non_blank']} values where the pipeline reads "
            f"({az['n_non_nan_raw']} raw): {az['unique_raw']}")


if __name__ == "__main__":
    root = resolve_data_root(ASSET)
    paths = vn.find_sessions(resolve_dataset_dir(ASSET, root=root))
    if "--all" not in sys.argv:
        paths = [p for p in paths
                 if os.path.basename(os.path.dirname(str(p))) in SUSPECTS]

    records = []
    for p in paths:
        rec = describe(p)
        records.append(rec)
        note = SUSPECTS.get(rec["session"], "")
        print(f"\n=== {rec['session']}")
        if note:
            print(f"    {note}")
        print(f"    {verdict(rec)}")
        w = rec["drifting_gratings_windowed"]
        f = rec["drifting_gratings_full"]
        print(f"    windowed rows {w['n_rows_raw']} (blank {w['n_blank']}), "
              f"full-field rows {f['n_rows_raw']}")
        for c in COLUMNS:
            print(f"      windowed {c:17s}: {w[c]}")
            print(f"      full     {c:17s}: {f[c]}")

    dest = "/scratch/window_center_probe.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {dest}  ({len(records)} session(s))")
