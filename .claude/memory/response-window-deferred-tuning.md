---
name: response-window-deferred-tuning
description: "Stimulus response windows are deliberately frozen to allen_v1dd's values during the port; revisit and tune them per trace type after validating against the 2025 published dataframes."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-14T23:10:13.301Z
---

Decided 2026-08-14 while porting the `allen_v1dd` stimulus metrics onto NWB-Zarr
(`SWDB_2026_Connectomics`, branch `functional-features`,
`code/supplement/Functional Data Stimulus Metrics.ipynb`).

**The choice:** reproduce the original's response windows exactly, even where they look
wrong, until the port is validated against the published `data_frames/*_M409828.csv`
tables. Changing a window and disagreeing with the published numbers is uninterpretable —
it could mean the fix worked or the port broke. Match first, then tune.

**Revisit after validation.** The user explicitly wants to come back to this.

## What to reconsider

**Natural movie `pref_img` cannot mean what it appears to.** The window is
`3 × imaging dt` ≈ 0.495 s, but movie frames are 1/30 s apart, so one trial's window spans
the ~15 movie frames that follow. A burst of activity is counted in ~15 consecutive
trials, and which one wins the argmax is decided by small differences — including whether
a window happens to contain 12 or 13 imaging samples, since the response is
`sum / n_samples`. `pref_img` localises a ~0.5 s neighbourhood, not a frame. It also means
consecutive "trials" are heavily autocorrelated, so `lifetime_sparseness` over
3,600 × 9 of them is not measuring lifetime sparseness in the usual sense.

**Windows should differ by trace type, and the original already half-acknowledged this.**
Deconvolved events are sparse and exactly zero most of the time; dF/F carries slow
calcium transients. The original used `3 × dt` for events vs `4 × dt` for dF/F on natural
movie, and no baseline subtraction for events vs a `(-3, 0)` s baseline for dF/F on
drifting gratings — but never revisited whether those were right. An optimal window for
sparse event trains is probably shorter than for dF/F.

**Drifting gratings: 1.985 s vs 2.0 s.** M1 measured `stop_time - start_time` at 1.985 s
in the NWB per-trial table; the original took 2.0 s from an NWB attribute. Default is 2.0
for bug-compatibility. Measured on 2026-08-14: 1.985 s moves `preferred_dir` on 2.7 % of
ROIs and shifts `osi` by ~5e-3, against ~1e-9 agreement at 2.0 s. So 2.0 s is right.

**Natural images: 0.33 s, recovered empirically (2026-08-14).** The original read this
from an NWB `duration_sec` attribute the current files no longer carry. Scanning against
the published table gives a sharp optimum at **0.33 s** (exact, 1e-16) where 0.30 s gives
6e-3 and 0.35 s gives 2e-3. The cause is discrete rather than continuous: at dt = 0.165 s
a 0.33 s window is just under 2*dt = 0.33008 and so catches **exactly two samples every
trial**, while 0.30 s catches one or two depending on where the onset falls between
frames. That varying count rescales each trial differently — which is why
`lifetime_sparseness` diagnosed it while `pref_response` did not: it is invariant to a
global scale but not a per-trial one. Equivalent and dt-robust alternative:
`ni_response_frames=2`. The margin on 0.33 s is only 8e-5 s, so re-probe if dt changes.

**General lesson for the revisit:** response windows here behave as *discrete* sample
counts, not continuous durations. Two windows differing by less than a sample interval
are often identical; two that straddle a sample boundary differ a lot. Tune in units of
dt, and use a scale-invariant metric to detect a varying count.

## The knobs already in place

`MetricConfig` in `code/utils/stimulus_metrics.py`:
`nm_response_frames` (3), `lsn_response_frames` (4), `dg_response_seconds` (2.0, also
accepts `"per_trial"`), `ni_response_seconds` (0.30). Changing any of them is a one-line
edit; the validation harness `compare_to_published` then quantifies the effect against the
published tables.

Related: [[aind-metadata-for-derived-assets]], [[v1dd-functional-metrics-fork]].
