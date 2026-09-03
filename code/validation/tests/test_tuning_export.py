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
                   "trace_type",
                   "dgw_trials", "dgw_blank", "dgw_n_blank", "dgw_params", "dgw_running",
                   "dgf_trials", "dgf_blank", "dgf_n_blank", "dgf_params", "dgf_running"])
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
    check("blank counts key on the PLANE too, one per plane",
          d["dgw_n_blank"].shape == (N_PLANES,), str(d["dgw_n_blank"].shape))
    check("equal-width blanks are recorded at their true width, not a padded one",
          d["dgw_n_blank"].tolist() == [N_BLANK] * N_PLANES, str(d["dgw_n_blank"]))
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


refuses("a plane with a different TRIAL count is refused",
        n_trials=[N_TRIALS, 7, N_TRIALS])
refuses("running speeds out of step with plane keys are refused", drop_plane_key=True)
refuses("an ROI-axis length that disagrees with the table is refused", short_roi_key=True)


print("\n[2b] ragged BLANK counts are padded, not refused")
# The 192 sweeps the pre-flight verified are the TOTAL per grating type, blanks included,
# so a session showing more grey sweeps shows fewer grating trials. The 2026-09-03 run
# measured 5-8 blanks per session, which is what this case now stands for: real data, not
# a malformed accumulator. The old expectation -- a raise -- is what cost that run its
# tuning-curve archive after five hours of compute.
RAGGED = [N_BLANK, 6, 5]
with tempfile.TemporaryDirectory() as tmp:
    tuning, keys = make(blank_counts=RAGGED)
    originals = [a.copy() for a in tuning["dgw"]["blank"]]
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        run(tuning, keys, tmp=tmp)
        raised = None
    except Exception as exc:                                            # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    check("ragged blank counts no longer abort the archive", raised is None,
          raised or "")
    if raised is None:
        d = np.load(pjoin(tmp, "tuning_curves_M409828.npz"), allow_pickle=True)
        check("blanks are padded to the WIDEST plane",
              d["dgw_blank"].shape == (N_PLANES * N_ROIS_PER, max(RAGGED)),
              str(d["dgw_blank"].shape))
        check("the true per-plane width is recorded, so a pad is not read as a zero",
              d["dgw_n_blank"].tolist() == RAGGED, str(d["dgw_n_blank"]))
        # padding must be NaN, not 0: a zero would be averaged into the baseline
        for p, n in enumerate(RAGGED):
            rows = slice(p * N_ROIS_PER, (p + 1) * N_ROIS_PER)
            check(f"plane {p}: its {n} real sweeps survive unchanged",
                  np.array_equal(d["dgw_blank"][rows, :n], originals[p]))
            check(f"plane {p}: columns {n}..{max(RAGGED)} are NaN, not zero",
                  bool(np.isnan(d["dgw_blank"][rows, n:]).all()),
                  str(d["dgw_blank"][rows, n:]))
        # what a reader actually does with the array
        check("nanmean over the blank axis ignores the padding",
              np.allclose(np.nanmean(d["dgw_blank"][:N_ROIS_PER * 3], axis=1)[-N_ROIS_PER:],
                          np.nanmean(originals[2], axis=1), equal_nan=True))
        d.close()


print("\n[3] the condition-means writer cell")
cond = [i for i, c in enumerate(cells)
        if c["cell_type"] == "code" and "condition_means_" in "".join(c["source"])
        and "savez" in "".join(c["source"])]
check("exactly one cell writes the condition-means archive", len(cond) == 1, str(cond))
CSRC = "".join(cells[cond[0]]["source"])
N_IMG, N_IMG12 = 118, 12


def cmake(widths=None, ids_short=False, short_roi_key=False):
    rng = np.random.default_rng(1)
    widths = widths or [N_IMG] * N_PLANES
    means = {"natural_images": [], "natural_images_12": []}
    ids = {"natural_images": np.arange(N_IMG),
           "natural_images_12": np.arange(N_IMG12) * 3}     # sparse subset, as in reality
    if ids_short:
        ids["natural_images"] = np.arange(N_IMG - 1)
    keys = []
    for p in range(N_PLANES):
        keys += [f"M409828_1_3_{p}_{r}" for r in range(N_ROIS_PER)]
        means["natural_images"].append(
            rng.normal(size=(N_ROIS_PER, widths[p])).astype(np.float32))
        means["natural_images_12"].append(
            rng.normal(size=(N_ROIS_PER, N_IMG12)).astype(np.float32))
    if short_roi_key:
        keys = keys[:-1]        # table shorter than the matrix, which must be refused
    return means, ids, keys


def crun(means, ids, keys, tmp):
    ns = {"np": np, "os": os, "pjoin": pjoin, "cond_means": means, "cond_ids": ids,
          "tables": {"natural_images": pd.DataFrame({"roi_key": keys})},
          "save_dir": tmp, "mouse_label": "M409828", "CONFIG": Cfg2(), "extra_outputs": []}
    exec(compile(CSRC, "<condition-means>", "exec"), ns)
    return ns


class Cfg2:
    trace_type = {"natural_images": "events"}


with tempfile.TemporaryDirectory() as tmp:
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        ns = crun(*cmake(), tmp=tmp)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    path = pjoin(tmp, "condition_means_M409828.npz")
    check("the archive is written", os.path.isfile(path))
    d = np.load(path, allow_pickle=True)
    n_rois = N_PLANES * N_ROIS_PER
    check("ni_mean is (n_rois, n_images)", d["ni_mean"].shape == (n_rois, N_IMG),
          str(d["ni_mean"].shape))
    check("ni12_mean is (n_rois, 12)", d["ni12_mean"].shape == (n_rois, N_IMG12),
          str(d["ni12_mean"].shape))
    check("natural movie is deliberately absent",
          not any(k.startswith("nm") for k in d.files), str(d.files))
    check("image ids travel with each matrix",
          len(d["ni_images"]) == N_IMG and len(d["ni12_images"]) == N_IMG12)
    check("ni12 image ids are a sparse subset of the 118-image namespace, not 0..11",
          bool(d["ni12_images"].max() > N_IMG12), str(d["ni12_images"][:4]))
    check("roi_key spans the ROI axis", len(d["roi_key"]) == n_rois)
    check("stored as float32", d["ni_mean"].dtype == np.float32)
    check("the writer registers itself for the manifest and provenance",
          ns["extra_outputs"] == ["condition_means_M409828.npz"])
    d.close()


def crefuses(label, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            crun(*cmake(**kw), tmp=tmp)
            raised = None
        except AssertionError as exc:
            raised = str(exc)
        except Exception as exc:                                        # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        finally:
            sys.stdout.close()
            sys.stdout = stdout
    check(label, raised is not None, (raised or "nothing raised")[:92])


crefuses("planes disagreeing on image count are refused", widths=[N_IMG, N_IMG, 90])
crefuses("an ROI count that disagrees with the table is refused", short_roi_key=True)
crefuses("image ids that do not match the matrix width are refused", ids_short=True)

summary()
