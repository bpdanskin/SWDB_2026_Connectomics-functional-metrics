"""The notebook's tuning-curve writer cell, run against synthetic planes.

This cell is notebook source, so no ordinary unit test reaches it — and every provenance
defect this pipeline has shipped had that same shape: a check that passed because it ran
somewhere production does not. So this executes the **real cell text** in a prepared
namespace, once on well-formed input and once per deliberately broken input, to confirm
each assertion actually fires rather than merely being present.

The broken cases matter more than the good one. An assertion nobody has seen fail is a
comment.
"""

from harness import REPO, check, summary
import json
import os
import sys
import tempfile
import zipfile
from os.path import join as pjoin

import numpy as np
import pandas as pd


NB = pjoin(REPO, "code", "supplement", "V1DD Stimulus Metrics.ipynb")
N_PLANES, N_ROIS_PER, N_DIR, N_SF, N_TRIALS, N_BLANK = 3, 5, 12, 2, 8, 8

cells = json.load(open(NB, encoding="utf-8"))["cells"]
matches = [i for i, c in enumerate(cells)
           if c["cell_type"] == "code" and "tuning_curves_" in "".join(c["source"])
           and "savez" in "".join(c["source"])]
check("exactly one cell writes the tuning-curve archive", len(matches) == 1, str(matches))
if len(matches) != 1:
    summary()
SRC = "".join(cells[matches[0]]["source"])


class Cfg:
    trace_type = {"drifting_gratings_windowed": "events"}


def make(blank_counts=None, n_trials=None, drop_plane_key=False, short_roi_key=False):
    """Per-plane accumulators shaped exactly as the notebook loop builds them."""
    blank_counts = blank_counts or [N_BLANK] * N_PLANES
    n_trials = n_trials or [N_TRIALS] * N_PLANES
    rng = np.random.default_rng(0)
    tuning = {k: {p: [] for p in ("trials", "blank", "params", "running", "plane_key")}
              for k in ("dgw", "dgf")}
    keys = []
    for p in range(N_PLANES):
        pk = f"M409828_1_3_{p}"
        keys += [f"{pk}_{r}" for r in range(N_ROIS_PER)]
        for k in ("dgw", "dgf"):
            a = tuning[k]
            a["trials"].append(
                rng.normal(size=(N_ROIS_PER, N_DIR, N_SF, n_trials[p])).astype(np.float32))
            a["blank"].append(
                rng.normal(size=(N_ROIS_PER, blank_counts[p])).astype(np.float32))
            a["params"].append(rng.normal(size=(N_ROIS_PER, N_SF, 6)).astype(np.float32))
            a["running"].append(
                rng.normal(size=(N_DIR, N_SF, n_trials[p])).astype(np.float32))
            if not (drop_plane_key and p == 0):
                a["plane_key"].append(pk)
    if short_roi_key:
        keys = keys[:-1]
    return tuning, keys


def run(tuning, keys, tmp):
    ns = {"np": np, "os": os, "pjoin": pjoin, "tuning": tuning,
          "tuning_axes": (np.arange(0, 360, 30.0), np.array([0.04, 0.08])),
          "tables": {"drifting_gratings_windowed": pd.DataFrame({"roi_key": keys})},
          "save_dir": tmp, "mouse_label": "M409828", "CONFIG": Cfg(), "extra_outputs": []}
    exec(compile(SRC, "<tuning-writer>", "exec"), ns)
    return ns


print("[1] a well-formed run writes everything it promises")
with tempfile.TemporaryDirectory() as tmp:
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")          # the cell prints a summary line
    try:
        ns = run(*make(), tmp=tmp)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    path = pjoin(tmp, "tuning_curves_M409828.npz")
    check("the archive is written", os.path.isfile(path))

    z = zipfile.ZipFile(path)
    names = sorted(n[:-4] for n in z.namelist())
    z.close()
    want = sorted(["roi_key", "plane_key", "directions", "spatial_frequencies",
                   "trace_type", "dgw_trials", "dgw_blank", "dgw_params", "dgw_running",
                   "dgf_trials", "dgf_blank", "dgf_params", "dgf_running"])
    check("every promised array is present", names == want,
          str(sorted(set(want) ^ set(names))))

    d = np.load(path, allow_pickle=True)
    n_rois = N_PLANES * N_ROIS_PER
    check("trials are (n_rois, n_dir, n_sf, n_trials)",
          d["dgw_trials"].shape == (n_rois, N_DIR, N_SF, N_TRIALS),
          str(d["dgw_trials"].shape))
    check("both grating types have the same shape",
          d["dgw_trials"].shape == d["dgf_trials"].shape)
    check("running speeds key on the PLANE, with no ROI axis",
          d["dgw_running"].shape == (N_PLANES, N_DIR, N_SF, N_TRIALS),
          str(d["dgw_running"].shape))
    check("roi_key spans the ROI axis", len(d["roi_key"]) == n_rois)
    check("plane_key spans the running-speed plane axis",
          len(d["plane_key"]) == N_PLANES)
    # the join the file promises: strip a roi_key's trailing _{roi} to get its plane
    check("every roi_key prefix appears in plane_key",
          set(k.rsplit("_", 1)[0] for k in d["roi_key"]) <= set(d["plane_key"].tolist()))
    check("stored as float32, not float64", d["dgw_trials"].dtype == np.float32,
          str(d["dgw_trials"].dtype))
    check("the trace type the numbers came from is recorded",
          str(d["trace_type"]) == "events")
    check("axis labels travel with the data",
          len(d["directions"]) == N_DIR and len(d["spatial_frequencies"]) == N_SF)
    check("the writer registers itself for the manifest and provenance",
          ns["extra_outputs"] == ["tuning_curves_M409828.npz"], str(ns["extra_outputs"]))
    d.close()


print("\n[2] malformed input is refused, not silently concatenated")


def refuses(label, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            run(*make(**kw), tmp=tmp)
            raised = None
        except AssertionError as exc:
            raised = str(exc)
        except Exception as exc:                                        # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        finally:
            sys.stdout.close()
            sys.stdout = stdout
    check(label, raised is not None, (raised or "nothing raised")[:96])


# Blank counts are the one dimension the pre-flight never verified across sessions: the
# 192 sweeps it checked are the non-blank total, and blank sweeps are intermingled.
refuses("a plane with a different BLANK count is refused",
        blank_counts=[N_BLANK, N_BLANK, 6])
refuses("a plane with a different TRIAL count is refused",
        n_trials=[N_TRIALS, 7, N_TRIALS])
refuses("running speeds out of step with plane keys are refused", drop_plane_key=True)
refuses("an ROI-axis length that disagrees with the table is refused", short_roi_key=True)

summary()
