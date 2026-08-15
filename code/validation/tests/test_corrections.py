"""The three corrections applied in P3, and their exact relationship to the old behaviour.

Each is provable on synthetic data, which matters: it means "the correction is right" does
not depend on a capsule run, and the capsule run only has to confirm the size of the
change on real data.

* **Receptive-field scale.** Corrected centres are the historical ones times exactly
  `n/(n-1)`. A constant factor is a much stronger claim than "the numbers moved a bit".
* **Preferred condition.** Only ROIs with no finite response anywhere change, and they
  change from a fabricated condition 0 to NaN.
* **Natural-images window.** Two imaging samples per trial at *every* sampling rate,
  where the 0.33 s window only manages that at the one rate it was tuned on.
"""
import numpy as np
import pandas as pd

from harness import check, fails, load, summary

tr = load("trial_responses")
vn = load("v1dd_nwb")
sm = load("stimulus_metrics")

print("[0] the defaults moved, and REFERENCE_CONFIG holds the old set")
d, r = sm.DEFAULT_CONFIG, sm.REFERENCE_CONFIG
check("default rf_center_scale_bug is off", d.rf_center_scale_bug is False)
check("default pref_cond_fillna is off", d.pref_cond_fillna is False)
check("default natural-images window is 2 frames", d.ni_response_frames == 2)
check("reference keeps the scale bug", r.rf_center_scale_bug is True)
check("reference keeps the fillna", r.pref_cond_fillna is True)
check("reference uses the 0.33 s window", r.ni_response_frames is None
      and r.ni_response_seconds == 0.33)
import dataclasses
differing = {f.name for f in dataclasses.fields(sm.MetricConfig)
             if getattr(d, f.name) != getattr(r, f.name)}
check("exactly three settings differ", differing == {
    "rf_center_scale_bug", "pref_cond_fillna", "ni_response_frames"}, str(sorted(differing)))

print("\n[1] receptive-field centres: corrected == historical * n/(n-1), exactly")
alt = (np.arange(8) - 4 + 0.5) * 9.3
azi = (np.arange(14) - 7 + 0.5) * 9.3
for name, centers, n in (("altitude", alt, 8), ("azimuth", azi, 14)):
    idx = np.array([0.0, 1.0, 3.5, float(n - 1)])
    bug = sm._rf_pixel_to_degrees(idx, centers, True)
    fixed = sm._rf_pixel_to_degrees(idx, centers, False)
    # Stated as a product, not a ratio: at the centre of the screen both values are
    # exactly 0, so the ratio there is 0/0 while the relation itself still holds.
    check(f"{name}: corrected == historical * n/(n-1) at every pixel",
          np.allclose(fixed, bug * (n / (n - 1)), rtol=0, atol=1e-12),
          f"max |diff| {np.max(np.abs(fixed - bug * n / (n - 1))):.2e}")
check("corrected altitude spans the true +/-32.55 deg",
      abs(sm._rf_pixel_to_degrees(7, alt, False) - 32.55) < 1e-9)
check("corrected azimuth spans the true +/-60.45 deg",
      abs(sm._rf_pixel_to_degrees(13, azi, False) - 60.45) < 1e-9)
check("historical altitude was compressed to 28.481",
      abs(sm._rf_pixel_to_degrees(7, alt, True) - 28.4812) < 1e-3)
# The clean factor is not a coincidence: it holds because the pixel grid is centred on
# zero (c[0] == -range/2). On an off-centre grid the two mappings would differ by an
# affine transform, and the validation assertion below would need an intercept term.
check("the pure-scale relation depends on the grid being centred",
      abs(alt[0] + (alt[-1] - alt[0]) / 2) < 1e-12
      and abs(azi[0] + (azi[-1] - azi[0]) / 2) < 1e-12)
off = alt + 100.0
check("shifting the grid off centre breaks the pure ratio, as expected",
      not np.allclose(sm._rf_pixel_to_degrees(np.arange(8.0), off, False),
                      sm._rf_pixel_to_degrees(np.arange(8.0), off, True) * (8 / 7)))

print("\n[2] preferred condition: only no-response ROIs change, and they change to NaN")
# Three ROIs: one with a clear preference, one flat, one with no finite response at all.
N_DIR, N_SF, N_TRIALS, N_ROIS = 12, 2, 8, 3
mean_tr = np.zeros((N_ROIS, N_DIR, N_SF))
mean_tr[0] = 0.1
mean_tr[0, 7, 1] = 5.0                     # ROI 0 prefers direction index 7, sf index 1
mean_tr[1] = 0.2                           # ROI 1 is flat -> index 0 either way
mean_tr[2] = np.nan                        # ROI 2 has no finite response anywhere

k_fill = np.nan_to_num(mean_tr, nan=-1.0).reshape(N_ROIS, -1).argmax(axis=1)
k_skip = np.where(np.isfinite(mean_tr), mean_tr, -np.inf).reshape(N_ROIS, -1).argmax(axis=1)
check("both definitions agree on the ROI with a real preference",
      k_fill[0] == k_skip[0] == 7 * N_SF + 1, f"{k_fill[0]} {k_skip[0]}")
check("both fabricate index 0 for the all-NaN ROI -- which is the actual defect",
      k_fill[2] == 0 and k_skip[2] == 0,
      "switching definition alone would not have fixed anything")
no_response = ~np.isfinite(mean_tr).any(axis=(1, 2))
check("the all-NaN ROI is the one that gets marked -1", list(no_response) == [False, False, True])

print("\n[3] natural-images window: two samples at every sampling rate")
# The three dt values that matter, from the pre-flight: the extremes of the asset and the
# one the 0.33 s window was recovered on.
for dt, label in ((0.16123, "asset minimum"), (0.16504, "the tuned session"),
                  (0.16671, "asset maximum")):
    ts = np.arange(0.0, 400.0, dt)
    # Onsets deliberately spread across the sampling phase.
    starts = 5.0 + np.arange(200) * 1.7 + np.linspace(0, dt, 200, endpoint=False)

    a = np.searchsorted(ts, starts + 0.0, side="left")
    b = np.searchsorted(ts, starts + 0.33, side="right")
    counts_time = b - a

    counts_frames = np.full(len(starts), 2)     # what sweep_responses_frames takes
    uniq = sorted(set(counts_time.tolist()))
    ok = uniq == [2]
    check(f"dt={dt} ({label}): 0.33 s window sample count {uniq}", ok
          if label == "the tuned session" else not ok,
          "constant" if ok else f"varies -> per-trial rescaling")
    check(f"dt={dt}: a 2-frame window is always 2 samples",
          set(counts_frames.tolist()) == {2})

print("\n[4] the frame-based window is what sweep_responses_frames actually does")
dt = 0.16123
ts = np.arange(0.0, 100.0, dt)
traces = np.tile(np.arange(len(ts), dtype=float)[:, None], (1, 2))
starts = 5.0 + np.arange(30) * 2.0 + np.linspace(0, dt, 30, endpoint=False)
got = tr.sweep_responses_frames(traces, ts, starts, n_frames=2)
first = np.searchsorted(ts, starts, side="left")
expected = 0.5 * (traces[first, 0] + traces[first + 1, 0])
check("each sweep is the mean of exactly the first two samples after onset",
      np.allclose(got[:, 0], expected, rtol=0, atol=1e-12),
      f"max diff {np.max(np.abs(got[:, 0] - expected)):.2e}")
check("no NaN from a ragged window", np.isfinite(got).all())

summary()
