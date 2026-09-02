---
name: v1dd-metrics-open-questions
description: "Open questions about the V1DD stimulus metrics — response window tuning, the 67%-low-confidence session, imperfections shipped on purpose. Grating-window geometry and RF map export are now implemented."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-09-01T00:00:00.000Z
---

Things left open as of 2026-08-16, each on purpose. None is a bug to fix unprompted; each
is a decision waiting on either evidence or the user.

**Status update 2026-09-01:** Items 1 (grating-window geometry) and 3 (RF map export)
are now implemented and ready for the next run. See details below.

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

## The windowed-grating window is now captured ✓ DONE (2026-09-01)

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
* (Superseded 2026-09-01 by the shipped table below — at the time only those two of 25 sessions had been checked.) **Only those two of 25 sessions were ever checked.** Acquisition notes say the position
  was tuned per session, so it plausibly varies by column — the two that agree are both
  column 1, so they test volume-to-volume stability and nothing else.
* **Diameter is absent from the NWB stimulus table entirely** — there is no size column.
  It has to come from acquisition-side metadata.

**All three sub-items implemented 2026-09-01:**

1. `preflight.py` `_stimulus_coverage`: now collects `center_azimuth` / `center_elevation`
   for `drifting_gratings_windowed` rows.
2. `surround_suppression_metrics` now emits `dgw_center_azimuth` / `dgw_center_elevation`
   per-ROI (session constants); `OUTPUT_COLUMNS["surround_supression_index"]` updated.
3. `DGResult` has a new `center: Tuple[float, float]` field; `drifting_gratings_metrics`
   extracts it from the non-blank windowed trials.

### Answered 2026-09-01: the window is fixed per COLUMN

Read straight off `surround_supression_index_M409828.csv` in the
`409828_V1DD_stimulus_metrics_2026-09-01_07-37-53` asset, all 25 sessions. The position is
constant across a column's five volumes:

| column | azimuth | elevation |
|---|---|---|
| 1 | -8.9 | -12.4 |
| 2 | -19.6 (volume 2: **-19.8**) | -10.0 |
| 3 | +1.8 | -9.7 |
| 4 | -15.4 | -16.4 |
| 5 | +9.9 | -14.4 |

Two caveats that matter for any containment analysis:

* **Column 2 / volume 2 sits 0.2 deg off its column.** Retargeting jitter rather than a
  different window, but do not test for exact equality within a column.
* **Two sessions have no centre at all** — column 2 / volume 5 (906 ROIs) and column 4 /
  volume 1 (1,550 ROIs), both blank. Both have complete SSI and DGW values
  (`preferred_dir` non-null for every valid ROI), so the stimulus ran and only the
  recorded position is missing. **2,456 ROIs cannot be filtered for RF containment.**
  Column 4 / volume 1 is also the 67 %-low-confidence session; the two problems look
  unrelated, and its 512 non-null SSI rows are exactly its 1,550 minus 1,038 invalid, so
  the low-confidence story is unchanged.

### Answered 2026-09-01 from the white paper: the diameter is 30 degrees

`V1DD_WhitePaper_v6.pdf` (Abbasi-Asl et al., Aug 2019) states it twice: "the stimulus was
restricted to a **30 degree diameter window**" and "The radius of window is 15 degrees."
It also gives the reason for the per-column position -- "For each column, the position of
the window was determined separately to align with the population receptive fields of
imaged neurons" -- which is exactly the pattern measured off the shipped SSI table.

**So containment is computable, and it was computed. The result is a negative one.**

* Of ROIs that are windowed-responsive, have an ON receptive field, and have a recorded
  window: **2,387 of 39,407 (6.1 %)**. Of those, only **970 (40.6 %)** have their RF centre
  within 15 degrees. So the cleanly interpretable SSI population is **2.5 % of the asset**.
  In column 1 alone, **67.1 %** of RF-on cells sit outside the window -- the white paper
  found the same ("over half") and listed surround suppression under "ongoing analysis".
* **But `ssi` does not track that distance at all**: Pearson r = -0.03, and binned means are
  flat (~0.48-0.53) from 0 to beyond 37 degrees. Not a selection effect -- dropping the
  responsiveness filter gives r = -0.02 over 6,827 ROIs. A targeting miss would push
  `ssi = (W-F)/(W+F)` *negative*, and no such trend exists at any distance.

**The most likely reason the test is blunt, and what to do instead.** RF pixels are 9.3
degrees, so a 30-degree window is about three pixels across and a centre two pixels off can
still have most of its field inside. Centre distance is the wrong measurement; **overlap
between the RF map and the window disc is the right one**, and `rf_maps_M409828.npz` now
ships the per-cell maps needed for it. Secondary: our centre is an unweighted centroid, so
one marginal pixel moves it ~4.6 degrees, attenuating any real relationship.

Practical guidance until that is done: `rf_inside_window_on` is a **conservative** filter --
cells passing it are well targeted -- but cells failing it should not be discarded, because
the distance test is not sensitive enough to justify it.

Diameter (size of the aperture) is still absent — it has to come from acquisition-side
metadata and is not in the NWB stimulus table. The per-session centre is now in the asset;
containment testing against `rf_metrics` centres is now possible.

## The per-cell 2D receptive-field map is now exported ✓ DONE (2026-09-01)

**Implemented 2026-09-01.** `receptive_field_metrics` now returns `(df, rf_map)` where
`rf_map` is `(n_rois, 2, n_rows, n_cols)` float32 — the pre-threshold continuous fraction.
The notebook saves `rf_maps_M{mouse}.npz` with `roi_key`, `altitudes`, `azimuths`, `seed`.

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
