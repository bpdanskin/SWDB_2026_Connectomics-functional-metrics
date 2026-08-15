"""receptive_field_metrics against synthetic locally-sparse-noise data."""

from harness import check, fails, load, require_dataset, summary
import sys

import numpy as np
import pandas as pd


tr = load("trial_responses")
vn = load("v1dd_nwb")
sm = load("stimulus_metrics")


RNG = np.random.default_rng(19)
DT = 0.16504
ROWS, COLS, GRID = 8, 14, 9.3
# Sized to the real asset (1705 LSN sweeps). This matters: the 0.25 significance
# threshold is applied per pixel, so with few presentations per pixel a 5%-by-chance
# rate produces false receptive fields. At 240 sweeps that is ~4.4 false pixels per ROI;
# at 1600 it is 1e-10. Period spaced so the (-1,0)s baseline and (0,4*dt) response never
# overlap between sweeps.
N_FRAMES, PERIOD = 1600, 1.8
N_ROIS = 4
TARGET_R, TARGET_C = 3, 5

# --- template: -1 (dark) / 0 (gray) / +1 (bright), as this asset encodes it
images = np.zeros((N_FRAMES, ROWS, COLS), dtype=np.int8)
for f in range(N_FRAMES):
    on = RNG.choice(ROWS * COLS, 6, replace=False)
    off = RNG.choice(np.setdiff1d(np.arange(ROWS * COLS), on), 6, replace=False)
    images[f].flat[on] = 1
    images[f].flat[off] = -1
# guarantee the target pixel is bright on a decent number of frames
# The target must be bright on only a SMALL fraction of frames. The method counts, per
# pixel, how often its presentations coincided with a significant response -- so if the
# driving pixel is bright on a third of all frames, every *other* pixel also coincides a
# third of the time and clears the 0.25 threshold. That is why the stimulus is called
# locally SPARSE noise: sparsity is what keeps non-RF pixels below threshold.
# 5%, not 10%. A non-RF pixel's coincidence rate is (drive rate) + (5% chance from the
# 95th-percentile threshold), and it must sit well below 0.25 or spurious pixels join the
# centroid. At a 10% drive rate that margin is 2.9 sigma (~0.2 spurious pixels per ROI,
# which is enough to move the centre); at 5% it is 4.8 sigma.
bright_frames = np.arange(0, N_FRAMES, 20)
images[bright_frames, TARGET_R, TARGET_C] = 1
dark_only = np.setdiff1d(np.arange(N_FRAMES), bright_frames)
images[dark_only, TARGET_R, TARGET_C] = np.where(
    images[dark_only, TARGET_R, TARGET_C] == 1, 0, images[dark_only, TARGET_R, TARGET_C])

azimuths = (np.arange(COLS) - COLS // 2 + 0.5) * GRID
altitudes = (np.arange(ROWS) - ROWS // 2 + 0.5) * GRID
lsn = {"images": images, "azimuths": azimuths, "altitudes": altitudes,
       "pixel_on": 1, "pixel_off": -1, "pixel_gray": 0, "pixel_values": [-1, 0, 1]}

starts = 30.0 + np.arange(N_FRAMES) * PERIOD
trials = pd.DataFrame({"stim_name": "locally_sparse_noise", "start_time": starts,
                       "stop_time": starts + 0.30, "frame": np.arange(N_FRAMES).astype(float)})

spont_start = starts[-1] + 5.0
spont_stop = spont_start + 300.0
ts = np.arange(0.0, spont_stop + 20.0, DT)
traces = RNG.normal(0.0, 0.02, size=(len(ts), N_ROIS))     # dF/F-like, zero-mean

# ROI 0 responds whenever the target pixel is bright; ROI 1 is silent; ROI 2 responds to
# everything (should light up broadly); ROI 3 silent.
for f, s in enumerate(starts):
    w = (ts >= s) & (ts <= s + 4 * DT)
    if images[f, TARGET_R, TARGET_C] == 1:
        traces[w, 0] += 2.0
    traces[w, 2] += 2.0

roi_table = pd.DataFrame({"column": 1, "volume": 3, "plane": 0,
                          "roi": np.arange(N_ROIS), "pika_roi_confidence": 0.9})
plane = vn.PlaneData(column=1, volume="3", plane=0, roi=np.arange(N_ROIS),
                     is_valid=np.ones(N_ROIS, bool), timestamps=ts,
                     traces={"dff": traces}, roi_table=roi_table, dt=DT)

cfg = sm.MetricConfig(other_n_boot=2000)

print("[1] receptive_field_metrics")
out = sm.receptive_field_metrics(plane, trials, (spont_start, spont_stop), lsn,
                                 config=cfg, rng=np.random.default_rng(0))
check("one row per ROI", len(out) == N_ROIS)
check("has the published columns",
      {"has_rf_on", "has_rf_off", "has_rf_on_or_off", "azimuth_rf_on", "altitude_rf_on",
       "azimuth_rf_off", "altitude_rf_off"} <= set(out.columns))
check("pixel-selective ROI has an ON receptive field", bool(out.has_rf_on[0]),
      str(out.has_rf_on[0]))
check("silent ROIs have none",
      not out.has_rf_on[1] and not out.has_rf_off[1] and not out.has_rf_on_or_off[1])
check("has_rf_on_or_off is the OR", bool(out.has_rf_on_or_off[0]))
check("no-RF ROIs get NaN centres", bool(np.isnan(out.azimuth_rf_on[1])))

print("\n[2] the centre lands on the driving pixel")
exp_alt = (TARGET_R + 0.5) * ((altitudes[-1] - altitudes[0]) / ROWS) + altitudes[0]
exp_azi = (TARGET_C + 0.5) * ((azimuths[-1] - azimuths[0]) / COLS) + azimuths[0]
check("altitude matches the target pixel", abs(out.altitude_rf_on[0] - exp_alt) < 1e-9,
      f"{out.altitude_rf_on[0]:.4f} vs {exp_alt:.4f}")
check("azimuth matches the target pixel", abs(out.azimuth_rf_on[0] - exp_azi) < 1e-9,
      f"{out.azimuth_rf_on[0]:.4f} vs {exp_azi:.4f}")

print("\n[3] the point_to_alt_azi scale bug, reproduced on purpose")
# a centroid on the LAST pixel is the sharpest test: the published tables span
# +/-28.481 altitude and +/-56.132 azimuth, not the true +/-32.55 and +/-60.45
last_alt_bug = sm._rf_pixel_to_degrees(ROWS - 1, altitudes, True)
last_azi_bug = sm._rf_pixel_to_degrees(COLS - 1, azimuths, True)
check("buggy altitude at the last row is 28.481 (published range)",
      abs(last_alt_bug - 28.4812) < 1e-3, f"{last_alt_bug:.4f}")
check("buggy azimuth at the last column is 56.132 (published range)",
      abs(last_azi_bug - 56.1321) < 1e-3, f"{last_azi_bug:.4f}")
check("corrected altitude is the true 32.55",
      abs(sm._rf_pixel_to_degrees(ROWS - 1, altitudes, False) - 32.55) < 1e-9)
check("corrected azimuth is the true 60.45",
      abs(sm._rf_pixel_to_degrees(COLS - 1, azimuths, False) - 60.45) < 1e-9)
check("the ratio is exactly (n-1)/n",
      abs(last_alt_bug / 32.55 - 7 / 8) < 1e-9 and abs(last_azi_bug / 60.45 - 13 / 14) < 1e-9,
      f"alt {last_alt_bug / 32.55:.6f} (7/8), azi {last_azi_bug / 60.45:.6f} (13/14)")
fixed = sm.receptive_field_metrics(plane, trials, (spont_start, spont_stop), lsn,
                                   config=sm.MetricConfig(other_n_boot=2000,
                                                          rf_center_scale_bug=False),
                                   rng=np.random.default_rng(0))
check("the flag changes the centres but not which ROIs have an RF",
      np.array_equal(out.has_rf_on.to_numpy(), fixed.has_rf_on.to_numpy())
      and abs(out.altitude_rf_on[0] - fixed.altitude_rf_on[0]) > 1e-6,
      f"{out.altitude_rf_on[0]:.4f} vs {fixed.altitude_rf_on[0]:.4f}")

print("\n[4] the pixel encoding, which is where a literal port breaks")
wrong = dict(lsn, pixel_on=255, pixel_off=0)      # the original's hard-coded constants
wrong_out = sm.receptive_field_metrics(plane, trials, (spont_start, spont_stop), wrong,
                                       config=cfg, rng=np.random.default_rng(0))
# 255 matches nothing here, so every ON field silently disappears
check("hard-coded pixel_on=255 erases every ON receptive field",
      not wrong_out.has_rf_on.any(),
      f"{int(wrong_out.has_rf_on.sum())} ON fields (correct run: {int(out.has_rf_on.sum())})")
# and pixel_off=0 collides with GRAY, so the OFF map stops meaning "dark pixel" and
# starts meaning "background pixel" -- answering a different question over ~16x as many
# presentations. Check the design matrix directly rather than a downstream count.
flat = images.reshape(N_FRAMES, -1)
true_off = int((flat == -1).sum())
gray_as_off = int((flat == 0).sum())
check("pixel_off=0 selects background instead of dark, over ~16x more presentations",
      gray_as_off > 10 * true_off,
      f"{gray_as_off} gray vs {true_off} genuinely dark")
check("so a literal port returns plausible-looking nonsense rather than an error",
      len(wrong_out) == N_ROIS and not wrong_out.isna().all().all())
missing = dict(lsn, pixel_on=None)
try:
    sm.receptive_field_metrics(plane, trials, (spont_start, spont_stop), missing,
                               config=cfg, rng=np.random.default_rng(0))
    check("raises when the codes cannot be determined", False)
except ValueError as e:
    check("raises when the codes cannot be determined", "pixel codes" in str(e))

print("\n[5] guardrails and schema")
bad = trials.copy()
bad.loc[0, "frame"] = 9999.0
try:
    sm.receptive_field_metrics(plane, bad, (spont_start, spont_stop), lsn, config=cfg,
                               rng=np.random.default_rng(0))
    check("raises on a frame index past the template", False)
except ValueError as e:
    check("raises on a frame index past the template", "exceeds" in str(e))

pub = sm.to_output_schema(out, "rf_metrics")
check("column order matches published",
      list(pub.columns) == list(sm.OUTPUT_COLUMNS["rf_metrics"]))
check("has_rf_* written as bool", all(pub[c].dtype == bool for c in
                                      ("has_rf_on", "has_rf_off", "has_rf_on_or_off")))

summary()
