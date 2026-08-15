"""preflight's flattening and summary against synthetic session records.

The probes themselves need a real NWB, but `_row` and `summarise` are pure dict
navigation over deeply optional structures -- which is exactly where this would fail on
the capsule, ten minutes into a thirty-five session run.
"""

from harness import check, fails, load, require_dataset, summary
import json
import sys
import tempfile
from os.path import join as pjoin

import numpy as np
import pandas as pd
pf = load("preflight")


def plane(n_frames=21616, n_rois=409, dff=True, events=True, loc="VISp layer 2/3",
          origin=(0.0, 0.0, 175.0)):
    return {
        "interfaces": ["dff", "events"], "has_events": events, "has_dff": dff,
        "shape": [n_frames, n_rois], "n_frames": n_frames, "dt": 0.16504,
        "duration_s": n_frames * 0.16504,
        "imaging_plane": {"name": "imaging_plane_0", "location": loc,
                          "description": "V1DD", "origin_coords": list(origin),
                          "origin_coords_unit": "micrometers", "device": "2p",
                          "numbers_in_text": [2.0, 3.0]},
    }


def record(name, column=1, volume="3", families=None, n_planes=6, subject="409828",
           lsn_ok=True, spont=1, error=None, fmt="zarr", transposed=False):
    families = pf.FAMILIES if families is None else families
    if error:
        return {"name": name, "path": f"/data/{name}/x.nwb.zarr", "format": fmt,
                "error": error, "mouse_from_name": name.split("_")[0],
                "probe_seconds": 0.1}
    detail = {}
    for p in range(n_planes):
        d = plane()
        if transposed:
            d["shape"] = [d["shape"][1], d["shape"][0]]
        detail[f"plane-{p}"] = d
    return {
        "name": name, "path": f"/data/{name}/x.nwb.zarr", "format": fmt, "error": None,
        "mouse_from_name": name.split("_")[0], "probe_seconds": 12.3,
        "file": {"session_id": "794964451", "intervals": ["epochs", "stimulus_table"]},
        "subject": {"present": subject is not None, "subject_id": subject,
                    "species": "Mus musculus", "sex": "M"},
        "roi_identity": {"column": column, "volume": volume,
                         "roi_columns": ["column", "roi", "volume"]},
        "planes": {"n_planes": n_planes, "detail": detail},
        "stimulus": {
            "epochs": {"n_rows": 11, "stim_names": sorted(families),
                       "n_spontaneous_blocks": spont},
            "spontaneous_block": [876.3, 1176.5],
            "stimulus_table": {"n_rows": 33214, "columns": ["stim_name", "start_time"],
                               "sweeps_per_stimulus": {f: 100 for f in families}},
            "per_family": {
                f: ({"present": True, "n_sweeps": 192, "n_direction": 12,
                     "n_spatial_frequency": 2, "n_blank_sweeps": 16}
                    if f in families else {"present": False, "n_sweeps": 0})
                for f in pf.FAMILIES},
        },
        "lsn_template": {"error": None if lsn_ok else "2x2 blocks not uniform",
                         "native_shape": [16, 28], "reduced_shape": [1705, 8, 14],
                         "pixel_on": 1, "pixel_off": -1},
        "running_speed": {"n_samples": 210000, "min": -2.0, "max": 61.0},
    }


print("[1] a healthy session flattens")
r = pf._row(record("409828_2018-12-13"))
check("n_rois_total sums the planes", r["n_rois_total"] == 6 * 409, str(r["n_rois_total"]))
check("dt recovered", abs(r["dt_median"] - 0.16504) < 1e-9)
check("all six families counted", all(r[f"n_{pf._short(f)}"] == 192 for f in pf.FAMILIES))
check("mouse from both sources", r["mouse_subject"] == "409828" and r["mouse_name"] == "409828")
check("lsn ok", r["lsn_ok"] is True)

print("\n[2] transposed traces do not inflate the ROI count")
rt = pf._row(record("409828_t", transposed=True))
check("n_rois still 6 x 409", rt["n_rois_total"] == 6 * 409, str(rt["n_rois_total"]))

print("\n[3] the degenerate cases this survey exists to find")
r_missing = pf._row(record("409828_3p", column=4, volume="a",
                           families=["drifting_gratings_full", "natural_images"]))
check("absent families read as 0 sweeps",
      r_missing["n_lsn"] == 0 and r_missing["n_nm"] == 0, f"lsn={r_missing['n_lsn']}")
check("present families still counted", r_missing["n_dgf"] == 192)
check("letter volume survives as a string", r_missing["volume"] == "a")

r_err = pf._row(record("409828_bad", error="OSError: cannot open"))
check("an unreadable session still produces a row", r_err["name"] == "409828_bad")
check("its error is recorded", "OSError" in (r_err["error"] or ""))
check("its family counts are 0 not None", r_err["n_dgf"] == 0)

r_nosub = pf._row(record("409828_ns", subject=None))
check("a missing subject leaves mouse_subject None", r_nosub["mouse_subject"] is None)

print("\n[4] summarise over a mixed asset")
recs = [
    record("409828_a", column=1, volume="3"),
    record("409828_b", column=1, volume="5", fmt="hdf5"),
    record("409828_c", column=4, volume="a",
           families=["drifting_gratings_full", "natural_images"], lsn_ok=False),
    record("409828_d", column=5, volume="f", spont=0),
    record("409828_e", error="RuntimeError: boom"),
]
v = pf.summarise(recs)
check("counts sessions and readables", v["n_sessions"] == 5 and v["n_readable"] == 4)
check("formats tallied", v["formats"] == {"zarr": 4, "hdf5": 1}, str(v["formats"]))
check("sessions with all six", v["sessions_with_all_six"] == 3, str(v["sessions_with_all_six"]))
check("names the session missing LSN",
      v["sessions_missing_a_family"]["locally_sparse_noise"] == ["409828_c"],
      str(v["sessions_missing_a_family"]["locally_sparse_noise"]))
check("names the session with no spontaneous block",
      v["sessions_without_spontaneous"] == ["409828_d"])
check("names the unreadable session", v["unreadable"] == ["409828_e"])
check("mouse sources agree", v["mouse_sources_agree"] is True)
check("lsn template ok count excludes the failure", v["lsn_template_ok"] == 3,
      str(v["lsn_template_ok"]))
check("volumes keep letters", set(v["volumes"]) == {"3", "5", "a", "f"}, str(v["volumes"]))

print("\n[5] mouse disagreement is detected, not smoothed over")
v2 = pf.summarise([record("409828_a"), record("999999_b", subject="409828")])
check("disagreement flagged", v2["mouse_sources_agree"] is False)

print("\n[6] the verdict is JSON-serialisable with no NaN")
from checkpoints import jsonable
s = json.dumps(jsonable(v), allow_nan=False, sort_keys=True)
check("serialises", len(s) > 200, f"{len(s)} chars")

print("\n[7] an all-failed asset does not crash the summary")
v3 = pf.summarise([record("x", error="boom"), record("y", error="boom")])
check("handles zero readable sessions", v3["n_readable"] == 0 and v3["n_sessions"] == 2)
json.dumps(jsonable(v3), allow_nan=False)
check("still serialises", True)

summary()
