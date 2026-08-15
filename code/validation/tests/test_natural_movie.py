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
plane = vn.PlaneData(column=1, volume="3", plane=0, roi=np.arange(N_ROIS)*3,
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
check("roi_key keeps it (non-colliding)", out.roi_key.iloc[0]=="M409828_13_0_0", out.roi_key.iloc[0])

print("\n[4] lifetime sparseness chunking matches the tested reference")
ta = RNG.gamma(1.0,1.0,size=(40,6,7)); ta[RNG.random(ta.shape)<0.1]=np.nan
ref = tr.lifetime_sparseness(ta.transpose(2,0,1).reshape(7,-1))
got = sm._lifetime_sparseness_chunked(ta, block=7)
check("chunked == reference", np.allclose(ref,got,equal_nan=True),
      f"max diff {np.nanmax(np.abs(ref-got)):.2e}")

summary()
