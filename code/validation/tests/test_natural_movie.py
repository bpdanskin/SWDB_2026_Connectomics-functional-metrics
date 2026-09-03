"""natural_movie_metrics against a synthetic plane where the answer is known."""

from harness import check, fails, load, require_dataset, summary
import numpy as np, pandas as pd
tr=load("trial_responses"); vn=load("v1dd_nwb"); sm=load("stimulus_metrics")


RNG=np.random.default_rng(3)
N_MOVIE, N_REPEAT, N_ROIS = 60, 9, 5
DT = 0.16504
# movie sweeps at 1/30 s, then a spontaneous block after
sweep_dt = 1/30
n_sweeps = N_MOVIE*N_REPEAT
starts = 10.0 + np.arange(n_sweeps)*sweep_dt
frames = np.tile(np.arange(N_MOVIE), N_REPEAT)
spont_start = starts[-1] + 5.0
spont_stop = spont_start + 300.0
ts = np.arange(0, spont_stop+20.0, DT)
# Deconvolved events are non-negative and exactly zero most of the time. That matters:
# frac_responsive_trials is mean(response > 0), so ANY strictly-positive background makes
# it 1.0 for every ROI. Start from exact zeros.
traces = np.zeros((len(ts), N_ROIS))

# The response window is 3 imaging frames (~0.5 s) but movie frames are 1/30 s apart, so
# a response at one frame lands inside the window of the next ~15 frames. Injecting at a
# single frame therefore cannot isolate it -- that overlap is the original's design.
SPILL = int(np.ceil(3 * DT / sweep_dt))
TARGET_FRAME = 17

# ROI 0: an event at frame 17 on every repeat
for rep in range(N_REPEAT):
    t0 = starts[rep*N_MOVIE + TARGET_FRAME]
    traces[np.argmin(np.abs(ts - t0)), 0] += 5.0
# ROI 1: the same event on only 4 of 9 repeats
for rep in range(4):
    t0 = starts[rep*N_MOVIE + TARGET_FRAME]
    traces[np.argmin(np.abs(ts - t0)), 1] += 5.0
# ROI 2: silent -> frac_responsive 0
# ROIs 3-4: sparse events well away from the movie block, so they stay unresponsive there
_spont_ix = np.flatnonzero(ts > spont_start)
traces[RNG.choice(_spont_ix, 60), 3] = 1.0
traces[RNG.choice(_spont_ix, 60), 4] = 1.0

trials = pd.DataFrame({"start_time": starts, "stop_time": starts+sweep_dt,
                       "frame": frames.astype(float), "stim_name": "natural_movie"})
roi_table = pd.DataFrame({"column":1,"volume":3,"plane":0,"roi":np.arange(N_ROIS)*3,
                          "pika_roi_confidence":np.full(N_ROIS,0.9)})
plane = vn.PlaneData(mouse_id="409828", depth_um=150.0, column=1, volume="3", plane=0, roi=np.arange(N_ROIS)*3,
                     is_valid=np.ones(N_ROIS,bool), timestamps=ts,
                     traces={"events": traces}, roi_table=roi_table, dt=DT)

print("[1] natural_movie_metrics")
out = sm.natural_movie_metrics(plane, trials, (spont_start, spont_stop),
                               rng=np.random.default_rng(0))
check("one row per ROI", len(out)==N_ROIS, str(len(out)))
check("has the published metric columns",
      set(["frac_responsive_trials","lifetime_sparseness","pref_img","pref_response","z_score"])
      <= set(out.columns))
# The window looks FORWARD from each frame's onset, so an event at frame 17 lands
# inside the window of frames 17-15 .. 17. The reported preferred frame can therefore
# PRECEDE the frame that actually drove the response.
check("ROI 0 preferred frame is at or before the injected one, within one window",
      TARGET_FRAME - SPILL <= out.pref_img[0] <= TARGET_FRAME,
      f"{out.pref_img[0]} in [{TARGET_FRAME-SPILL}, {TARGET_FRAME}]")
check("ROI 1 preferred frame is at or before the injected one, within one window",
      TARGET_FRAME - SPILL <= out.pref_img[1] <= TARGET_FRAME,
      f"{out.pref_img[1]} in [{TARGET_FRAME-SPILL}, {TARGET_FRAME}]")
check("ROI 0 responds on every repeat", abs(out.frac_responsive_trials[0]-1.0)<1e-12,
      f"{out.frac_responsive_trials[0]}")
check("ROI 1 responds on 4 of 9 repeats", abs(out.frac_responsive_trials[1]-4/9)<1e-12,
      f"{out.frac_responsive_trials[1]}")
check("a 0.5 s window spans ~15 movie frames (why trials are autocorrelated)",
      SPILL == 15, str(SPILL))
check("ROI 2 (all-zero trace) has frac 0", abs(out.frac_responsive_trials[2])<1e-12,
      f"{out.frac_responsive_trials[2]}")
check("ROI 0 has the largest z_score", int(np.nanargmax(out.z_score))==0, str(out.z_score.to_list()))
check("frac is quantised to k/9", bool(np.all([abs(v*9-round(v*9))<1e-9
      for v in out.frac_responsive_trials.dropna()])))

print("\n[2] determinism")
again = sm.natural_movie_metrics(plane, trials, (spont_start, spont_stop),
                                 rng=np.random.default_rng(0))
det = ["frac_responsive_trials","lifetime_sparseness","pref_img","pref_response"]
check("deterministic columns are bit-identical across runs",
      all(np.allclose(out[c].to_numpy(float), again[c].to_numpy(float), equal_nan=True) for c in det))
check("z_score reproducible under the same seed",
      np.allclose(out.z_score.to_numpy(), again.z_score.to_numpy(), equal_nan=True))
diff_seed = sm.natural_movie_metrics(plane, trials, (spont_start, spont_stop),
                                     rng=np.random.default_rng(1))
check("z_score moves with the seed (it is the only stochastic column)",
      not np.allclose(out.z_score.to_numpy(), diff_seed.z_score.to_numpy(), equal_nan=True))
check("but the deterministic columns do NOT move with the seed",
      all(np.allclose(out[c].to_numpy(float), diff_seed[c].to_numpy(float), equal_nan=True) for c in det))

print("\n[3] published schema")
pub = sm.to_output_schema(out, "natural_movie")
check("column order matches the published file",
      list(pub.columns)==list(sm.OUTPUT_COLUMNS["natural_movie"]), str(list(pub.columns)))
check("pref_img is int", pd.api.types.is_integer_dtype(pub.pref_img))
check("volume is str", isinstance(pub.volume.iloc[0], str))
check("roi_unique_id drops the column (published format)",
      pub.roi_unique_id.iloc[0]=="M409828_3_0_0", pub.roi_unique_id.iloc[0])
check("roi_key keeps it (non-colliding)", out.roi_key.iloc[0]=="M409828_1_3_0_0", out.roi_key.iloc[0])

print("\n[4] lifetime sparseness chunking matches the tested reference")
ta = RNG.gamma(1.0,1.0,size=(40,6,7)); ta[RNG.random(ta.shape)<0.1]=np.nan
ref = tr.lifetime_sparseness(ta.transpose(2,0,1).reshape(7,-1))
got = sm._lifetime_sparseness_chunked(ta, block=7)
check("chunked == reference", np.allclose(ref,got,equal_nan=True),
      f"max diff {np.nanmax(np.abs(ref-got)):.2e}")

print("\n[5] reliability: reported on both trace types")
# The plane above carries only events, which is the degrade path: reliability_dff must be
# NaN rather than raising, so a single-trace plane still yields a valid frame.
check("reliability computed on the events trace",
      np.isfinite(out["reliability"]).any(),
      f"{int(np.isfinite(out['reliability']).sum())} of {N_ROIS} ROIs finite")
check("reliability_dff is all-NaN when dff was not loaded",
      bool(out["reliability_dff"].isna().all()))

# Now with dF/F present. Build it as a smoothed, always-positive version of the same
# events so the two traces describe the same cell -- then dF/F reliability should exceed
# events reliability, because events are exactly zero on most frames and a mostly-flat
# repeat carries almost no pattern to correlate.
kernel = np.exp(-np.arange(12) / 3.0)
dff = np.stack([np.convolve(traces[:, r], kernel, mode="same") + 0.05
                for r in range(N_ROIS)], axis=1)
plane_both = vn.PlaneData(
    mouse_id="409828", depth_um=150.0, column=1, volume="3", plane=0,
    roi=np.arange(N_ROIS) * 3, is_valid=np.ones(N_ROIS, bool), timestamps=ts,
    traces={"events": traces, "dff": dff}, roi_table=roi_table, dt=DT)
both = sm.natural_movie_metrics(plane_both, trials, (spont_start, spont_stop),
                                rng=np.random.default_rng(0))
check("reliability_dff is populated when dff is loaded",
      np.isfinite(both["reliability_dff"]).any(),
      f"{int(np.isfinite(both['reliability_dff']).sum())} of {N_ROIS} ROIs finite")
check("adding a dff trace does not disturb the events reliability",
      np.allclose(out["reliability"], both["reliability"], equal_nan=True))
check("adding a dff trace does not disturb any other column",
      all(np.allclose(out[c], both[c], equal_nan=True)
          for c in ("frac_responsive_trials", "lifetime_sparseness", "pref_img",
                    "pref_response", "z_score")))
resp = np.isfinite(both["reliability"]) & np.isfinite(both["reliability_dff"])
# The two must actually be different numbers -- that is what says they were computed from
# different traces rather than the same one twice. Which is *larger* is deliberately not
# asserted: that is an empirical question about the data, and a unit test on synthetic
# traces cannot answer it. (On this fixture events score higher, because the synthetic
# dF/F is a convolution that smears each response across neighbouring movie frames and so
# blurs the very frame-indexed pattern reliability measures. Nothing follows from that
# about real recordings.)
check("the two reliabilities are genuinely different numbers",
      not np.allclose(both.loc[resp, "reliability"], both.loc[resp, "reliability_dff"]),
      f"dff median {both.loc[resp, 'reliability_dff'].median():+.3f} vs "
      f"events median {both.loc[resp, 'reliability'].median():+.3f}")

summary()
