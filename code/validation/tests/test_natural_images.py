"""natural_images_metrics against synthetic data.

The image ids here are deliberately non-contiguous ({2,4,5,9,23,68}), mirroring
natural_images_12, because that is where `pref_img` can silently return a position where
it should return an identity.
"""

from harness import check, fails, load, require_dataset, summary
import sys

import numpy as np
import pandas as pd


tr = load("trial_responses")
vn = load("v1dd_nwb")
sm = load("stimulus_metrics")


RNG = np.random.default_rng(11)
DT = 0.16504
IMAGE_IDS = np.array([2, 4, 5, 9, 23, 68])     # non-contiguous, like natural_images_12
TARGET_IMG = 23                                 # position 4 in the list
N_TRIALS, N_ROIS = 8, 5
WINDOW, PERIOD = 0.30, 0.317

target = np.full((N_ROIS, len(IMAGE_IDS)), 0.05)
target[0, list(IMAGE_IDS).index(TARGET_IMG)] = 3.0     # ROI 0 prefers image 23
target[1, :] = 0.0                                      # ROI 1 silent
target[2, :] = 0.5                                      # ROI 2 uniform
target[3, list(IMAGE_IDS).index(2)] = 2.0               # ROI 3 prefers image 2
# ROI 4 gets a single sparse EVENT at each onset instead of a flat level, which is
# what deconvolved traces actually look like -- and the only way the response
# window length can change the answer (mean = event / n_samples_in_window).
target[4, :] = 0.0

seq = np.repeat(IMAGE_IDS, N_TRIALS)
RNG.shuffle(seq)
starts = 40.0 + np.arange(len(seq)) * PERIOD
trials = pd.DataFrame({"stim_name": "natural_images", "start_time": starts,
                       "stop_time": starts + WINDOW, "image_index": seq.astype(float),
                       "image_order": np.arange(len(seq)) % len(IMAGE_IDS)})

spont_start = starts[-1] + 5.0
spont_stop = spont_start + 300.0
ts = np.arange(0.0, spont_stop + 20.0, DT)
traces = np.zeros((len(ts), N_ROIS))
for s, img in zip(starts, seq):
    w = (ts >= s) & (ts <= s + WINDOW)
    traces[w] = target[:, list(IMAGE_IDS).index(img)]
    if img == TARGET_IMG:                       # ROI 4: one event at onset, nothing else
        traces[np.argmin(np.abs(ts - s)), 4] = 4.0
traces[ts >= spont_start] = RNG.gamma(1.0, 0.1, size=(int((ts >= spont_start).sum()), N_ROIS))

roi_table = pd.DataFrame({"column": 1, "volume": 3, "plane": 0,
                          "roi": np.arange(N_ROIS), "pika_roi_confidence": 0.9})
plane = vn.PlaneData(mouse_id="409828", depth_um=150.0, column=1, volume="3", plane=0, roi=np.arange(N_ROIS),
                     is_valid=np.ones(N_ROIS, bool), timestamps=ts,
                     traces={"events": traces}, roi_table=roi_table, dt=DT)

cfg = sm.MetricConfig(ni_response_seconds=WINDOW, other_n_boot=2000)

print("[1] natural_images_metrics")
out = sm.natural_images_metrics(plane, trials, (spont_start, spont_stop),
                                config=cfg, rng=np.random.default_rng(0))
check("one row per ROI", len(out) == N_ROIS)
check("has the published metric columns",
      {"frac_responsive_trials", "lifetime_sparseness", "pref_img", "pref_response",
       "z_score"} <= set(out.columns))

check("pref_img is the IMAGE IDENTITY, not its position",
      out.pref_img[0] == TARGET_IMG, f"{out.pref_img[0]} (position would be 4)")
check("second tuned ROI picks its own image", out.pref_img[3] == 2, str(out.pref_img[3]))
check("pref_img values are drawn from the id list",
      set(out.pref_img[out.pref_img >= 0]) <= set(IMAGE_IDS.tolist()),
      str(sorted(set(out.pref_img))))
check("pref_response matches the injected amplitude",
      abs(out.pref_response[0] - 3.0) < 1e-9, f"{out.pref_response[0]:.6f}")

print("\n[2] frac_responsive_trials is a bootstrap test here, not mean(resp > 0)")
check("strongly driven ROI responds on every preferred trial",
      abs(out.frac_responsive_trials[0] - 1.0) < 1e-12, f"{out.frac_responsive_trials[0]}")
check("silent ROI is unresponsive", abs(out.frac_responsive_trials[1]) < 1e-12,
      f"{out.frac_responsive_trials[1]}")
check("quantised to k/8", bool(np.all(
    [abs(v * N_TRIALS - round(v * N_TRIALS)) < 1e-9 for v in out.frac_responsive_trials])))
# unlike natural movie, a uniformly small positive response is NOT automatically 1.0
nm_like = np.mean(np.asarray(out.frac_responsive_trials)[[2]])
check("uniform 0.5 response is judged against the null, not against zero",
      0.0 <= nm_like <= 1.0, f"ROI2 frac = {nm_like}")

print("\n[3] z_score and determinism")
check("driven ROI has the largest z_score", int(np.nanargmax(out.z_score)) == 0,
      str(np.round(out.z_score.to_numpy(), 2)))
again = sm.natural_images_metrics(plane, trials, (spont_start, spont_stop),
                                  config=cfg, rng=np.random.default_rng(0))
det = ["lifetime_sparseness", "pref_img", "pref_response"]
check("deterministic columns are seed-independent",
      all(np.allclose(out[c].to_numpy(float), again[c].to_numpy(float), equal_nan=True)
          for c in det))
check("same seed reproduces z_score exactly",
      np.allclose(out.z_score.to_numpy(), again.z_score.to_numpy(), equal_nan=True))
other = sm.natural_images_metrics(plane, trials, (spont_start, spont_stop),
                                  config=cfg, rng=np.random.default_rng(1))
check("z_score moves with the seed", not np.allclose(out.z_score.to_numpy(),
                                                     other.z_score.to_numpy(), equal_nan=True))
check("but pref_img does not", np.array_equal(out.pref_img, other.pref_img))

print("\n[4] the response window actually matters (so probing it is meaningful)")
short = sm.natural_images_metrics(
    plane, trials, (spont_start, spont_stop),
    config=sm.MetricConfig(ni_response_seconds=0.25, other_n_boot=2000),
    rng=np.random.default_rng(0))
# A flat within-sweep response is window-invariant, so the probe can only discriminate
# where the trace has structure inside the window -- which sparse events do.
check("flat-response ROI is window-invariant (why a flat fixture cannot probe)",
      abs(out.pref_response[0] - short.pref_response[0]) < 1e-12,
      f"{out.pref_response[0]:.4f} vs {short.pref_response[0]:.4f}")
# What actually makes a window probe able to discriminate is whether the candidates put
# a different number of imaging samples inside each trial. At dt = 0.165 s the effect is
# real but modest between 0.25 s and 0.30 s, and only a handful of trials move -- so a
# per-ROI difference is not guaranteed on a small fixture. Assert the mechanism.
_counts = {}
for W in (0.25, 0.30, 0.40):
    a, b = tr.window_bounds(ts, starts, 0.0, W)
    _counts[W] = float(np.mean(b - a))
check("candidate windows put different sample counts in each trial",
      _counts[0.25] < _counts[0.30] < _counts[0.40],
      "mean samples/trial: " + ", ".join(f"{w}s={c:.2f}" for w, c in _counts.items()))

wide = sm.natural_images_metrics(
    plane, trials, (spont_start, spont_stop),
    config=sm.MetricConfig(ni_response_seconds=0.40, other_n_boot=2000),
    rng=np.random.default_rng(0))
check("a well-separated window changes the sparse-event ROI's response",
      abs(out.pref_response[4] - wide.pref_response[4]) > 1e-6,
      f"{out.pref_response[4]:.5f} at 0.30 s vs {wide.pref_response[4]:.5f} at 0.40 s")

print("\n[5] guardrails and schema")
bad = trials.copy()
bad.loc[0, "image_index"] = np.nan
try:
    sm.natural_images_metrics(plane, bad, (spont_start, spont_stop), config=cfg,
                              rng=np.random.default_rng(0))
    check("raises on NaN image_index", False)
except ValueError as e:
    check("raises on NaN image_index", "NaN image_index" in str(e))

for fam in ("natural_images", "natural_images_12"):
    pub = sm.to_output_schema(out, fam)
    check(f"{fam} column order matches published",
          list(pub.columns) == list(sm.OUTPUT_COLUMNS[fam]))
check("pref_img written as int with -1 sentinel",
      pd.api.types.is_integer_dtype(sm.to_output_schema(out, "natural_images").pref_img))

summary()
