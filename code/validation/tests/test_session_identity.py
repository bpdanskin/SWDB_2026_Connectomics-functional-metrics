"""Who the ROI belongs to and where it sits: mouse derivation, depth, absent families.

These three were hard-coded, missing, and unguarded respectively before P2. Each has a
failure mode that produces plausible output rather than an error, which is why they are
tested rather than eyeballed:

* a wrong mouse id silently corrupts every `roi_unique_id`
* a missing depth is indistinguishable from depth zero if you let it default
* `bool(nan)` is **True**, so an absent stimulus would report a receptive field for
  every ROI in the session
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from harness import check, fails, load, summary

vn = load("v1dd_nwb")
sm = load("stimulus_metrics")


def fake_plane(mouse_id="409828", depth_um=150.0, n=5, confidence=None):
    roi = np.arange(n)
    table = pd.DataFrame() if confidence is None else pd.DataFrame(
        {"pika_roi_confidence": confidence})
    return vn.PlaneData(
        column=1, volume="3", plane=2, roi=roi, is_valid=np.ones(n, bool),
        timestamps=np.arange(n, dtype=float), traces={}, roi_table=table,
        dt=0.165, mouse_id=mouse_id, depth_um=depth_um)


class Module:
    """Minimal stand-in for an NWB processing module: subscriptable, has data_interfaces."""
    def __init__(self, interfaces): self.data_interfaces = interfaces
    def __getitem__(self, key): return self.data_interfaces[key]


def nwb_with(subject_id=None, location="146 um"):
    subject = SimpleNamespace(subject_id=subject_id) if subject_id is not None else None
    series = SimpleNamespace(rois=SimpleNamespace(
        table=SimpleNamespace(imaging_plane=SimpleNamespace(location=location))))
    return SimpleNamespace(subject=subject,
                           processing={"plane-0": Module({"dff": series})})


print("[1] session_mouse: the ladder, and that it never silently defaults")
check("reads nwb.subject.subject_id", vn.session_mouse(nwb_with("409828")) == ("409828", "M409828"))
check("strips a leading M if the file already has one",
      vn.session_mouse(nwb_with("M409828")) == ("409828", "M409828"))
check("falls back to the session directory name",
      vn.session_mouse(nwb_with(None), path="/data/409828_2018-12-13_15-10-05/x.nwb.zarr")
      == ("409828", "M409828"))
check("subject wins over the path when both are present",
      vn.session_mouse(nwb_with("111111"), path="/data/409828_x/y.nwb.zarr")[0] == "111111")
for bad, why in ((None, "no subject and no path"), ("", "empty subject_id"),
                 ("   ", "whitespace subject_id")):
    try:
        vn.session_mouse(nwb_with(bad))
        check(f"raises on {why}", False)
    except ValueError as exc:
        check(f"raises on {why}", "cannot determine the mouse" in str(exc))
try:
    vn.session_mouse(nwb_with(None), path="/data/not_a_number_session/x.nwb.zarr")
    check("raises when the directory name is not numeric", False)
except ValueError:
    check("raises when the directory name is not numeric", True)

print("\n[2] plane_depth_um: parses the free text, never raises")
check("parses '146 um'", vn.plane_depth_um(nwb_with("409828", "146 um"), 0) == 146.0)
check("parses a bare number", vn.plane_depth_um(nwb_with("409828", "434"), 0) == 434.0)
check("parses a decimal", vn.plane_depth_um(nwb_with("409828", "82.5 um"), 0) == 82.5)
check("None when location is absent", vn.plane_depth_um(nwb_with("409828", None), 0) is None)
check("None when location has no number",
      vn.plane_depth_um(nwb_with("409828", "VISp layer 2/3"), 0) is None)
check("None rather than an exception for a missing plane",
      vn.plane_depth_um(nwb_with("409828"), 7) is None)
# The anchored match earns its keep here: an unanchored search returns 2.0 for this,
# which is a plausible depth and silently wrong.
check("free-text location with an embedded number yields None, not that number",
      vn.plane_depth_um(nwb_with("409828", "layer 2/3 at 50 um"), 0) is None)
check("unit is optional and case-insensitive",
      vn.plane_depth_um(nwb_with("409828", "50UM"), 0) == 50.0)

print("\n[3] roi_frame carries both, and refuses to guess the mouse")
f = sm.roi_frame(fake_plane())
check("mouse column is the M-prefixed label", set(f["mouse"]) == {"M409828"})
check("roi_unique_id omits the column (historical format)",
      f["roi_unique_id"].iloc[0] == "M409828_3_2_0", f["roi_unique_id"].iloc[0])
check("roi_key includes the column, so it does not collide",
      f["roi_key"].iloc[0] == "M409828_1_3_2_0", f["roi_key"].iloc[0])
check("depth_um present and float", f["depth_um"].dtype == float and set(f["depth_um"]) == {150.0})
check("explicit mouse= overrides the plane",
      set(sm.roi_frame(fake_plane(), mouse="M999999")["mouse"]) == {"M999999"})
check("a bare id is accepted for mouse= too",
      set(sm.roi_frame(fake_plane(), mouse="999999")["mouse"]) == {"M999999"})
check("depth NaN, not zero, when the file does not say",
      bool(np.isnan(sm.roi_frame(fake_plane(depth_um=None))["depth_um"]).all()))
try:
    sm.roi_frame(fake_plane(mouse_id=""))
    check("raises rather than emitting M_3_2_0", False)
except ValueError as exc:
    check("raises rather than emitting M_3_2_0", "no mouse id" in str(exc))

print("\n[3b] pika_roi_confidence: the label that says which ROIs are unreliable")
conf = [0.9, 0.2, 0.51, 0.5, 0.99]
f = sm.roi_frame(fake_plane(confidence=conf))
check("carried through from the ROI table", f["pika_roi_confidence"].tolist() == conf)
check("float dtype", f["pika_roi_confidence"].dtype == float)
# is_valid is confidence > 0.5, strictly -- exactly 0.5 is not valid.
check("the >0.5 threshold that defines is_valid",
      (f["pika_roi_confidence"] > 0.5).tolist() == [True, False, True, False, True])
check("NaN, not 0, when the ROI table does not carry it",
      bool(np.isnan(sm.roi_frame(fake_plane())["pika_roi_confidence"]).all()),
      "a missing confidence must not read as zero confidence")
check("it is identity, not a metric -- absent_frame keeps it",
      "pika_roi_confidence" in sm.absent_frame(fake_plane(confidence=conf), "rf_metrics")
      and sm.absent_frame(fake_plane(confidence=conf),
                          "rf_metrics")["pika_roi_confidence"].tolist() == conf)

print("\n[4] depth_um reaches every output table")
# Driven from OUTPUT_COLUMNS rather than a hand-written list: a new family then gets
# these checks automatically instead of silently skipping them, which is how `locomotion`
# would have slipped through.
for fam in sorted(sm.OUTPUT_COLUMNS):
    check(f"{fam} schema has depth_um", "depth_um" in sm.OUTPUT_COLUMNS[fam])
    check(f"{fam} schema has pika_roi_confidence",
          "pika_roi_confidence" in sm.OUTPUT_COLUMNS[fam])
check("depth_um sits after the four join keys",
      sm.OUTPUT_COLUMNS["natural_images"].index("depth_um")
      == sm.OUTPUT_COLUMNS["natural_images"].index("roi") + 1)

print("\n[5] absent_frame: an absent stimulus reads as absent, not as zero")
for fam in sorted(sm.OUTPUT_COLUMNS):
    a = sm.absent_frame(fake_plane(), fam)
    out = sm.to_output_schema(a, fam)
    check(f"{fam}: schema and row count intact",
          list(out.columns) == list(sm.OUTPUT_COLUMNS[fam]) and len(out) == 5)
    ident = {"roi_unique_id", "mouse", "column", "volume", "plane", "roi", "depth_um",
             "pika_roi_confidence"}
    metrics = [c for c in out.columns if c not in ident]
    # The trap: to_output_schema casts booleans with astype(bool), and bool(nan) is True,
    # so leaving one NaN would claim a receptive field -- or an imputed aperture centre --
    # for every ROI in a session that never saw the stimulus. Driven off
    # sm.BOOLEAN_COLUMNS rather than a prefix or a hand-written list: the prefix version
    # of this check passed while `dgw_center_inferred` was reporting True.
    bools = [c for c in metrics if c in sm.BOOLEAN_COLUMNS]
    if bools:
        check(f"{fam}: {', '.join(bools)} are False, not True-from-NaN",
              not out[bools].to_numpy().any(), str(bools))
    metrics = [c for c in metrics if c not in bools]
    if fam == "rf_metrics":
        check("rf: centres are NaN", bool(out[metrics].isna().all().all()))
    elif "pref_img" in metrics:
        # Keyed on the SCHEMA, not on a list of family names. Keyed on names this missed
        # natural_images, which shares this schema and was simply absent from the
        # hand-written list -- the same drift the loop above now prevents.
        check(f"{fam}: pref_img uses the -1 sentinel", bool((out["pref_img"] == -1).all()))
        rest = [c for c in metrics if c != "pref_img"]
        check(f"{fam}: remaining metrics NaN", bool(out[rest].isna().all().all()))
    else:
        check(f"{fam}: every metric NaN", bool(out[metrics].isna().all().all()))

print("\n[6] the guards fire on an empty trials frame")
empty = pd.DataFrame(columns=["stim_name", "start_time", "stop_time", "frame"])
got = sm.natural_movie_metrics(fake_plane(), empty, (0.0, 1.0))
check("natural movie returns absent rows rather than computing", len(got) == 5
      and bool(pd.isna(got["lifetime_sparseness"]).all()))
try:
    sm.drifting_gratings_metrics(fake_plane(), empty, np.zeros(0, bool), (0.0, 1.0),
                                 (np.zeros(3), np.zeros(3)), dg_type="windowed")
    check("drifting gratings raises instead of faking a DGResult", False)
except ValueError as exc:
    check("drifting gratings raises instead of faking a DGResult",
          "no drifting_gratings_windowed sweeps" in str(exc)
          and "column 1" in str(exc), str(exc)[:80])

summary()
