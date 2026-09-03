"""Drifting gratings + surround suppression against analytically-known synthetic data.

Tuning curves are chosen so osi / dsi / gosi / pref_dir_mean have closed-form values:

  ROI 0  direction tuned  1 + cos(theta - 90)   -> osi 1/3, dsi 1.0, gosi 0.0, mean dir 90
  ROI 1  orientation tuned 1 + cos(2(theta-90)) -> osi 1.0, dsi 0.0, gosi 0.5
  ROI 2  flat zero                              -> unresponsive
"""

from harness import check, fails, load, require_dataset, summary
import sys

import numpy as np
import pandas as pd


tr = load("trial_responses")
vn = load("v1dd_nwb")
sm = load("stimulus_metrics")


RNG = np.random.default_rng(7)
DT = 0.16504
DIRS = np.arange(0, 360, 30).astype(float)
SFS = np.array([0.04, 0.08])
N_TRIALS, N_ROIS = 8, 4
WINDOW = 2.0
PERIOD = 3.0

theta = np.deg2rad(DIRS)
target = np.zeros((N_ROIS, 12, 2))
target[0, :, 0] = 1 + np.cos(theta - np.pi / 2)          # direction tuned, at sf 0.04
target[0, :, 1] = 0.05
target[1, :, 0] = 1 + np.cos(2 * (theta - np.pi / 2))    # orientation tuned, at sf 0.04
target[1, :, 1] = 0.05
target[2, :, :] = 0.0                                     # silent
target[3, :, :] = 0.5                                     # flat but responsive


def build_session(scale=1.0, seed=0):
    """A plane whose windowed response equals `scale * target` exactly."""
    rng = np.random.default_rng(seed)
    combos = [(d, s) for d in range(12) for s in range(2)] * N_TRIALS
    rng.shuffle(combos)
    rows, t = [], 50.0
    for d, s in combos:
        rows.append({"stim_name": "dg", "start_time": t, "stop_time": t + 1.985,
                     "direction": DIRS[d], "spatial_frequency": SFS[s],
                     "temporal_frequency": 1.0})
        t += PERIOD
    for _ in range(8):                                    # blank sweeps
        rows.append({"stim_name": "dg", "start_time": t, "stop_time": t + 1.985,
                     "direction": np.nan, "spatial_frequency": np.nan,
                     "temporal_frequency": np.nan})
        t += PERIOD
    trials = pd.DataFrame(rows)
    is_blank = trials[["temporal_frequency", "spatial_frequency", "direction"]].isna().any(axis=1).to_numpy()

    spont_start, spont_stop = t + 10.0, t + 310.0
    ts = np.arange(0.0, spont_stop + 20.0, DT)
    traces = np.zeros((len(ts), N_ROIS))
    for _, r in trials.loc[~is_blank].iterrows():
        d = int(np.argmin(np.abs(DIRS - r["direction"])))
        s = int(np.argmin(np.abs(SFS - r["spatial_frequency"])))
        w = (ts >= r["start_time"]) & (ts <= r["start_time"] + WINDOW)
        traces[w] = scale * target[:, d, s]
    spont_ix = ts >= spont_start
    traces[spont_ix] = rng.gamma(1.0, 0.1, size=(int(spont_ix.sum()), N_ROIS))

    roi_table = pd.DataFrame({"column": 1, "volume": 3, "plane": 0,
                              "roi": np.arange(N_ROIS), "pika_roi_confidence": 0.9})
    plane = vn.PlaneData(mouse_id="409828", depth_um=150.0, column=1, volume="3", plane=0, roi=np.arange(N_ROIS),
                         is_valid=np.ones(N_ROIS, bool), timestamps=ts,
                         traces={"events": traces}, roi_table=roi_table, dt=DT)
    # running speed: first half of every condition's trials fast, rest slow
    rts = np.arange(0.0, spont_stop + 20.0, 1 / 60)
    speed = np.zeros_like(rts)
    for i, (_, r) in enumerate(trials.loc[~is_blank].iterrows()):
        if i % 2 == 0:
            speed[(rts >= r["start_time"] - 0.2) & (rts <= r["stop_time"] + 0.2)] = 5.0
    return plane, trials, is_blank, (spont_start, spont_stop), (speed, rts)


print("[1] drifting gratings: analytic tuning")
plane, trials, is_blank, spont, running = build_session()
dgw = sm.drifting_gratings_metrics(plane, trials, is_blank, spont, running,
                                   dg_type="windowed", rng=np.random.default_rng(0))
m = dgw.metrics
check("one row per ROI", len(m) == N_ROIS)
check("12 directions, 2 SFs found",
      list(dgw.dir_list) == list(DIRS) and np.allclose(dgw.sf_list, SFS))
check("trial array shape (rois, dir, sf, trials)",
      dgw.trial_responses.shape == (N_ROIS, 12, 2, N_TRIALS), str(dgw.trial_responses.shape))
check("blank sweeps captured separately", dgw.blank_responses.shape == (N_ROIS, 8),
      str(dgw.blank_responses.shape))

check("ROI 0 preferred dir = 90", m.preferred_dir[0] == 90.0, str(m.preferred_dir[0]))
check("ROI 0 preferred sf = 0.04", np.isclose(m.preferred_sf[0], 0.04), str(m.preferred_sf[0]))
check("ROI 0 dsi = 1.0 (null response is zero)", abs(m.dsi[0] - 1.0) < 1e-9, f"{m.dsi[0]:.6f}")
check("ROI 0 osi = 1/3", abs(m.osi[0] - 1 / 3) < 1e-9, f"{m.osi[0]:.6f}")
check("ROI 0 gosi = 0 (pure 1st harmonic has no orientation vector)",
      abs(m.gosi[0]) < 1e-9, f"{m.gosi[0]:.2e}")
check("ROI 0 pref_dir_mean = 90", abs(m.pref_dir_mean[0] - 90.0) < 1e-6,
      f"{m.pref_dir_mean[0]:.4f}")

check("ROI 1 osi = 1.0", abs(m.osi[1] - 1.0) < 1e-9, f"{m.osi[1]:.6f}")
check("ROI 1 dsi = 0.0", abs(m.dsi[1]) < 1e-9, f"{m.dsi[1]:.2e}")
check("ROI 1 gosi = 0.5", abs(m.gosi[1] - 0.5) < 1e-9, f"{m.gosi[1]:.6f}")

check("ROI 2 (silent) is not responsive", m.frac_responsive_trials[2] == 0.0
      and m.is_responsive[2] == 0.0, f"{m.frac_responsive_trials[2]}")
check("ROIs 0/1/3 are responsive",
      all(m.is_responsive[i] == 1.0 for i in (0, 1, 3)), str(m.is_responsive.to_list()))
check("frac is quantised to k/8",
      bool(np.all([abs(v * 8 - round(v * 8)) < 1e-9 for v in m.frac_responsive_trials])))

print("\n[2] the two preferred-condition definitions")
check("pref_cond_index matches published preferred_dir",
      np.array_equal(dgw.dir_list[dgw.pref_cond_index[:, 0]], m.preferred_dir.to_numpy()))
check("invalid ROIs would be -1", dgw.pref_cond_index.shape == (N_ROIS, 2))

print("\n[3] guardrails")
bad = trials.copy()
bad.loc[bad["direction"] == 330.0, "direction"] = 300.0     # collapse to 11 directions
try:
    sm.drifting_gratings_metrics(plane, bad, is_blank, spont, running,
                                 dg_type="windowed", rng=np.random.default_rng(0))
    check("raises when directions != 12", False)
except ValueError as e:
    check("raises when directions != 12", "expected 12" in str(e))
check("_ratio returns 0 on a zero denominator", sm._ratio(1.0, 0.0) == 0.0)
check("_metric_index returns NaN on a zero denominator",
      bool(np.isnan(sm._metric_index(0.0, 0.0))))

print("\n[4] von Mises fit")
# exact recovery when the target genuinely lies in the model family
true_params = (1.0, 2.0, 90.0, 0.3, 1.5, 0.05)
y_vm = sm.vonmises_two_peak(DIRS, *true_params)
p_vm = sm.vonmises_two_peak_fit(DIRS, y_vm)
check("recovers true von Mises parameters",
      p_vm is not None and np.allclose(p_vm, true_params, atol=1e-6),
      str(np.round(p_vm, 4)))
check("and reproduces that curve to ~1e-10",
      np.max(np.abs(sm.vonmises_two_peak(DIRS, *p_vm) - y_vm)) < 1e-8)

# a cosine is OUTSIDE the model family, so residuals are expected. What matters for
# ssi_tuning_fit is only that the preferred direction is still recovered.
p = sm.vonmises_two_peak_fit(DIRS, target[0, :, 0])
check("fit converges on an out-of-family (cosine) curve", p is not None)
if p is not None:
    pred = sm.vonmises_two_peak(DIRS, *p)
    y = target[0, :, 0]
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    check("cosine fit is good but not exact (model mismatch, not failure)",
          0.9 < r2 < 1.0, f"R^2 = {r2:.4f}, max resid {np.max(np.abs(pred - y)):.3f}")
    check("preferred direction still recovered to <0.01 deg",
          abs(sm.vonmises_pref_dir(p) - 90) < 0.01, f"{sm.vonmises_pref_dir(p):.5f}")
check("fit returns None on an all-NaN curve",
      sm.vonmises_two_peak_fit(DIRS, np.full(12, np.nan)) is None)

print("\n[5] surround suppression")
# full-field responses at half the windowed amplitude -> ssi = (W-F)/(W+F) = 1/3
plane_f, trials_f, blank_f, spont_f, running_f = build_session(scale=0.5, seed=1)
dgf = sm.drifting_gratings_metrics(plane_f, trials_f, blank_f, spont_f, running_f,
                                   dg_type="full", rng=np.random.default_rng(0))
ssi = sm.surround_suppression_metrics(dgw, dgf, plane)
check("has the eight published SSI columns",
      all(c in ssi.columns for c in sm.SSI_COLUMNS))
check("ssi = 1/3 when W is twice F", abs(ssi.ssi[0] - 1 / 3) < 1e-9, f"{ssi.ssi[0]:.6f}")
check("ssi_avg = 1/3 too (uniform scaling)", abs(ssi.ssi_avg[0] - 1 / 3) < 1e-9,
      f"{ssi.ssi_avg[0]:.6f}")
check("ssi_avg_at_pref_sf = 1/3", abs(ssi.ssi_avg_at_pref_sf[0] - 1 / 3) < 1e-9)
check("running and stationary variants both finite (4 trials each side)",
      np.isfinite(ssi.ssi_running[0]) and np.isfinite(ssi.ssi_stationary[0]),
      f"run={ssi.ssi_running[0]:.4f} stat={ssi.ssi_stationary[0]:.4f}")
check("ssi_tuning_fit finite for a well-fit ROI", np.isfinite(ssi.ssi_tuning_fit[0]),
      f"{ssi.ssi_tuning_fit[0]:.4f}")

print("\n[6] SSI edge cases on hand-built results")


def fake(resp, speeds, pref=(0, 0)):
    n = 1
    return sm.DGResult(
        metrics=None, trial_responses=resp, trial_running_speeds=speeds,
        pref_cond_index=np.array([[pref[0], pref[1]]]),
        tuning_params=np.full((n, 2, 6), np.nan),
        dir_list=DIRS, sf_list=SFS, blank_responses=np.empty((n, 0)))


rp = vn.PlaneData(mouse_id="409828", depth_um=150.0, column=1, volume="3", plane=0, roi=np.array([0]),
                  is_valid=np.ones(1, bool), timestamps=np.arange(10) * DT,
                  traces={}, roi_table=pd.DataFrame({"column": [1], "volume": [3],
                                                     "plane": [0], "roi": [0]}), dt=DT)
W = np.full((1, 12, 2, 8), np.nan); W[0, 0, 0] = [2.0] * 8
F = np.full((1, 12, 2, 8), np.nan); F[0, 0, 0] = [1.0] * 8
sp_all_run = np.full((12, 2, 8), 5.0)
sp_two_run = np.full((12, 2, 8), 0.0); sp_two_run[0, 0, :2] = 5.0

r = sm.surround_suppression_metrics(fake(W, sp_all_run), fake(F, sp_all_run), rp)
check("all-running: ssi_running finite, ssi_stationary NaN",
      np.isfinite(r.ssi_running[0]) and np.isnan(r.ssi_stationary[0]))
r2 = sm.surround_suppression_metrics(fake(W, sp_two_run), fake(F, sp_two_run), rp)
check("only 2 running trials -> ssi_running NaN (needs 3)", np.isnan(r2.ssi_running[0]),
      f"{r2.ssi_running[0]}")
check("but 6 stationary trials -> ssi_stationary finite", np.isfinite(r2.ssi_stationary[0]))
sp_exact = np.full((12, 2, 8), 1.0)                       # exactly at the threshold
r3 = sm.surround_suppression_metrics(fake(W, sp_exact), fake(F, sp_exact), rp)
check("speed exactly 1.0 counts as neither running nor stationary",
      np.isnan(r3.ssi_running[0]) and np.isnan(r3.ssi_stationary[0]))
r4 = sm.surround_suppression_metrics(fake(W, sp_all_run, pref=(-1, -1)),
                                     fake(F, sp_all_run, pref=(-1, -1)), rp)
check("invalid preferred condition -> all NaN",
      bool(np.all([np.isnan(r4[c][0]) for c in sm.SSI_COLUMNS])))

print("\n[7] published schema")
dgp = sm.to_output_schema(dgw.metrics, "drifting_gratings_windowed")
check("DG column order matches published",
      list(dgp.columns) == list(sm.OUTPUT_COLUMNS["drifting_gratings_windowed"]))
check("is_responsive is float 0.0/1.0", dgp.is_responsive.dtype == float
      and set(np.unique(dgp.is_responsive)) <= {0.0, 1.0})
ssp = sm.to_output_schema(ssi, "surround_supression_index")
check("SSI column order matches published (misspelling kept)",
      list(ssp.columns) == list(sm.OUTPUT_COLUMNS["surround_supression_index"]))

print("\n[N] tuning curves survive to the exported arrays")
# What the .npz ships is DGResult.trial_responses verbatim. The property that matters is
# that the published scalar columns are recoverable from it -- otherwise the array and the
# table could drift and nothing would notice.
tc = dgw.trial_responses                      # (n_rois, n_dir, n_sf, n_trials)
curve = np.nanmean(tc, axis=3)                # (n_rois, n_dir, n_sf) -- the tuning curves

check("trial array is float-typed and finite where trials exist",
      np.isfinite(tc).any() and tc.dtype.kind == "f")
check("float32 round-trip preserves the curve to single precision",
      np.allclose(np.nanmean(tc.astype(np.float32), axis=3), curve,
                  rtol=1e-6, atol=1e-9, equal_nan=True),
      "the export casts to float32")

# preferred_dir must be the argmax of the curve at the preferred SF -- the same reduction
# the published column reports, recomputed from the shipped array
for roi in range(min(N_ROIS, 4)):
    if not np.isfinite(m.preferred_sf[roi]):
        continue
    sf_i = int(np.argmin(np.abs(dgw.sf_list - m.preferred_sf[roi])))
    at_pref = curve[roi, :, sf_i]
    if not np.isfinite(at_pref).any():
        continue
    recovered = dgw.dir_list[int(np.nanargmax(at_pref))]
    check(f"ROI {roi}: preferred_dir recomputed from the array matches the column",
          recovered == m.preferred_dir[roi],
          f"array {recovered} vs column {m.preferred_dir[roi]}")

# osi recomputed from the array, for the ROI whose analytic value is known above
sf_i = int(np.argmin(np.abs(dgw.sf_list - m.preferred_sf[0])))
at_pref = curve[0, :, sf_i]
di = int(np.nanargmax(at_pref))
orth = 0.5 * (at_pref[(di + 3) % 12] + at_pref[(di - 3) % 12])
check("ROI 0: osi recomputed from the shipped array equals the published column",
      abs((at_pref[di] - orth) / (at_pref[di] + orth) - m.osi[0]) < 1e-9)

check("blank sweeps are per-ROI and separate from the conditions",
      dgw.blank_responses.shape[0] == N_ROIS
      and dgw.blank_responses.shape[1] not in (12, 24))
check("running speeds have no ROI axis, so they key on the plane",
      dgw.trial_running_speeds.shape == (12, 2, N_TRIALS),
      str(dgw.trial_running_speeds.shape))
check("tuning params are (n_rois, n_sf, 6) von Mises coefficients",
      dgw.tuning_params.shape == (N_ROIS, 2, 6), str(dgw.tuning_params.shape))
# after the fit-only-the-used-SF speedup the unread SF is NaN BY DESIGN; a reader who
# does not know that will report it as a failed fit
fitted_per_roi = np.isfinite(dgw.tuning_params).all(axis=2).sum(axis=1)
check("at most one SF is fitted per ROI (the speedup, not a fit failure)",
      bool((fitted_per_roi <= 1).all()), f"max fitted SFs = {int(fitted_per_roi.max())}")

print("\n[N] fit_all_sf: both settings exercised, because a documented flag is not an "
      "implemented one")
# The pref_cond_fillna incident: a flag declared, documented, and never read, so flipping
# it was a silent no-op. Anything with a flag gets both settings run and asserted to
# differ.
cfg_all = sm.MetricConfig(fit_all_sf=True)
dgw_all = sm.drifting_gratings_metrics(plane, trials, is_blank, spont, running,
                                       dg_type="windowed", config=cfg_all,
                                       rng=np.random.default_rng(0))
dgf_all = sm.drifting_gratings_metrics(plane_f, trials_f, blank_f, spont_f, running_f,
                                       dg_type="full", config=cfg_all,
                                       fit_sf_index=dgw_all.pref_cond_index[:, 1],
                                       rng=np.random.default_rng(0))

fitted_fast = np.isfinite(dgw.tuning_params).all(axis=2).sum(axis=1)
fitted_all = np.isfinite(dgw_all.tuning_params).all(axis=2).sum(axis=1)
check("default fits at most one SF per ROI", bool((fitted_fast <= 1).all()),
      f"max {int(fitted_fast.max())}")
check("fit_all_sf=True fits more SFs than the default",
      int(fitted_all.sum()) > int(fitted_fast.sum()),
      f"{int(fitted_all.sum())} vs {int(fitted_fast.sum())} (ROI, SF) fits")
check("fit_all_sf=True overrides the WINDOWED self-selection, which has no other escape",
      bool((fitted_all >= fitted_fast).all() and (fitted_all > 1).any()),
      f"max fitted SFs {int(fitted_all.max())}")
# Note the `dgf` fixture above passes no fit_sf_index, so it already fits every SF --
# unlike the notebook, which passes dgw's preferred SF. The override has to be tested
# against a call that is actually using the speedup.
dgf_fast = sm.drifting_gratings_metrics(plane_f, trials_f, blank_f, spont_f, running_f,
                                        dg_type="full",
                                        fit_sf_index=dgw.pref_cond_index[:, 1],
                                        rng=np.random.default_rng(0))
check("fit_all_sf=True overrides an explicit fit_sf_index on full field",
      int(np.isfinite(dgf_all.tuning_params).all(axis=2).sum())
      > int(np.isfinite(dgf_fast.tuning_params).all(axis=2).sum()),
      f"{int(np.isfinite(dgf_all.tuning_params).all(axis=2).sum())} vs "
      f"{int(np.isfinite(dgf_fast.tuning_params).all(axis=2).sum())} (ROI, SF) fits")

# The flag must not move a published column: ssi_tuning_fit reads one SF per ROI either
# way, so the extra fits are genuinely extra.
ssi_all = sm.surround_suppression_metrics(dgw_all, dgf_all, plane, config=cfg_all)
check("no published SSI column moves when the extra SFs are fitted",
      all(np.allclose(ssi[c], ssi_all[c], equal_nan=True, rtol=1e-9, atol=1e-12)
          for c in sm.SSI_COLUMNS),
      "output-neutral for the tables, not for the exported params")

check("REFERENCE_CONFIG fits every SF, as the original did",
      sm.REFERENCE_CONFIG.fit_all_sf is True)
check("the default does not, so a fast run shows in differs_from_reference_config",
      sm.DEFAULT_CONFIG.fit_all_sf is False)

summary()
