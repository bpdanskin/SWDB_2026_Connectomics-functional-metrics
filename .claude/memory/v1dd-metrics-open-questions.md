---
name: v1dd-metrics-open-questions
description: "Deliberately unresolved questions about the V1DD stimulus metrics — deferred choices, two things the next run should capture (grating-window geometry, per-cell RF maps), an upstream data question, and imperfections shipped on purpose."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-25T18:53:57.257Z
---

Things left open as of 2026-08-16, each on purpose. None is a bug to fix unprompted; each
is a decision waiting on either evidence or the user.

## Response windows were matched, not chosen

The windows reproduce an earlier pipeline built for slow calcium transients; this one runs
on deconvolved events, which are far sparser. A window right for one is unlikely to be
right for the other. Matching first was the only way to make the port checkable — see
[[response-window-deferred-tuning]] for the recovered values and the discrete
sample-count reasoning. `MetricConfig` makes revisiting them a one-line change, and
`code/validation/` can say whether a change helped.

**The most defensible thing to tune first** is natural images: it is already expressed as
`ni_response_frames = 2`, so alternatives are integers rather than a duration to re-derive.

## Why is 67 % of one session low-confidence?

Column 4 / volume 1 (`409828_2018-11-21_09-22-23_filtered_2026-04-16_02-51-11`) has 1,038
of 1,550 ROIs at `pika_roi_confidence <= 0.5`. Every other session is unremarkable, and
nothing else about this one is: 6 planes, mid-range ROI count, dt in the middle of the
spread, all six stimuli present.

It has the **second-newest filtering date** in the asset (`2026-04-16` where most are
`2026-04-09`), which is a thread worth pulling but a sample of one. This is a question for
whoever produced the filtered NWB asset, not something the metrics pipeline can settle.
Until then a population average over all ROIs is weighted by one session's segmentation
quality — the access notebook says so, and the column now makes it filterable.

## The windowed-grating window is not recorded anywhere — grab it on the next run

`ssi_*` compares the windowed grating response against the full-field one, but nothing in
the asset says **where the window was or how big it was**. So the index cannot be
separated from a targeting miss: a cell whose receptive field falls outside the window has
a weak windowed response and reads as "suppressed" when it was simply not stimulated.
**We want centre and diameter attached to the SSI metrics**, so cells whose RFs are
reasonably contained in the window can be selected and the rest treated as
uninterpretable rather than merely noisy.

What was established 2026-08-24, from `checks/schema_report.json`:

* **One fixed window position per session.** `drifting_gratings_windowed` has 24
  conditions = 12 directions x 2 spatial frequencies, with no location factor, and the
  metrics code keys conditions off `direction x spatial_frequency` only.
* In **column 1, volumes 3 and 5** (different acquisition days) it is
  **azimuth −8.9°, elevation −12.4°**. Full-field rows carry a `(0.0, 0.0)` placeholder,
  which is how you tell the two apart in `center_azimuth`.
* **Only those two of 25 sessions were ever checked.** Acquisition notes say the position
  was tuned per session, so it plausibly varies by column — the two that agree are both
  column 1, so they test volume-to-volume stability and nothing else.
* **Diameter is absent from the NWB stimulus table entirely** — there is no size column.
  It has to come from acquisition-side metadata.

Three small changes on the next fresh run, none urgent:

1. `preflight.py`, `_stimulus_coverage`: add `center_azimuth` / `center_elevation` beside
   the `direction` / `spatial_frequency` / `temporal_frequency` collection. Two lines, and
   it answers the per-column question for all 25 sessions permanently. **Nothing we built
   reads those columns today**, which is why this is still unknown.
2. `stimulus_metrics.surround_suppression_metrics`: carry the centre into the output as
   `dgw_center_azimuth` / `dgw_center_elevation` (per-session constants, one column each).
3. Then containment is computable per ROI against the `rf_metrics` centres
   (`azimuth_rf_on`, `altitude_rf_on`), which are already in the same degree frame. Worth
   testing whether `ssi` falls off with RF-to-window distance — if it does, some of the
   published index is alignment rather than physiology.

## The per-cell 2D receptive-field map is computed and thrown away

We report only the ON/OFF **centres** and `has_rf_*`, but the full per-ROI subfield map —
the thing you would plot as an 8x14 ON panel beside an 8x14 OFF panel — is already built
in `receptive_field_metrics` and discarded when the function returns:

```python
rf = frac.reshape(plane.n_rois, 2, n_rows, n_cols)   # dim 1: 0 = ON, 1 = OFF
```

Each pixel is **the fraction of that pixel's presentations that produced a response above
that ROI's own bootstrapped spontaneous 95th percentile**. So exporting it is a retention
change, not an analysis step: **no extra computation at all**, and the centres already
shipped are derived from this same array.

Decisions if we do it:

* **Export the map from *before* `frac[frac < rf_frac_thresh] = 0.0`** (threshold 0.25).
  The post-threshold version is what the centroid uses, but it is recoverable from the
  continuous one in a line and the reverse is not. The continuous version is also what a
  familiar RF figure looks like — graded values across the whole grid rather than a few
  isolated survivors.
* **It does not fit the per-ROI CSV schema.** 2 x 8 x 14 = 224 floats per ROI, ~8.8 M
  values over 39,407 ROIs, ~35 MB as float32 (ample for a fraction backed by ~100
  presentations per pixel). Wants its own container — an `.npz` holding one
  `(n_rois, 2, 8, 14)` array plus a `roi_key` index is the obvious shape.
* **Three things must travel with it** or it is uninterpretable: `altitudes` / `azimuths`
  from the `lsn` dict, so pixel indices map to degrees (`_rf_pixel_to_degrees` already
  uses them); the **seed**, since the per-pixel significance comes from a bootstrap; and
  the fact that invalid ROIs are zeroed by `frac[~plane.is_valid] = 0.0`, so a blank map
  means *excluded*, not *no receptive field*.

Pairs naturally with the window geometry above — both are things the next fresh run should
capture rather than force a re-derivation later.

## Imperfections shipped deliberately

* **`ssi_tuning_fit` evaluates the fitted curve *including* its baseline offset, while the
  preferred direction feeding it is chosen with the baseline subtracted.** Inherited and
  reproduced on purpose. Whether to correct it is open; it would need the same
  before/after treatment the other two corrections got, and it is the one SSI variant with
  no seed-to-seed noise floor to judge against until tuning fits run on both seeds.
* **`roi_key` is only in the wide feather**, not the per-family CSVs, which keep the
  historical column set. Anyone working from the CSVs alone has no unique per-ROI string
  and must join on `(column, volume, plane, roi)`. Adding it is a schema change and
  another 7 h rerun.
* **`pref_img` for natural movie is approximate by construction.** The response window
  spans several frames and frames are 1/30 s apart, so activity from one frame lands in
  its neighbours' windows. Read it as "around here in the clip".
* **`_ratio` returns 0 on a zero denominator while `_metric_index` returns NaN.** Two
  lines, opposite conventions, both inherited. Not unified, because unifying them would
  change published columns for no measured benefit.

## Efficiency

Two unimplemented speedups worth ~2x, with the analysis already done and the
output-neutral one distinguished from the one that would move numbers — see
[[v1dd-metrics-speedups]]. Deferred so the first full run could proceed.

## Not open

The two corrections in [[v1dd-metrics-refactor-decisions]] (receptive-field scale,
preferred condition) are settled and validated. The natural-images window switch to a
frame count is settled. Do not reopen these without new evidence.
