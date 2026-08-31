---
name: v1dd-stimulus-metrics-port-status
description: "How the allen_v1dd stimulus-metrics port onto NWB was validated, milestone by milestone — complete; the two-seed control methodology is the reusable part."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-15T00:19:44.026Z
---

Porting the seven `allen_v1dd` stimulus-metric tables onto the NWB functional asset, in
`SWDB_2026_Connectomics` branch `functional-features`. Two coregistered sessions
(column 1 / volumes 3 and 5), **3,673 ROIs** — which exactly equals the number of
published rows carrying metrics for those sessions (2,708 + 965), so the join is complete.

**Code:** `code/utils/{v1dd_nwb,trial_responses,stimulus_metrics,checkpoints}.py` plus
`code/supplement/Functional Data Stimulus Metrics.ipynb` (33 cells). Verification
artifacts are committed under `scratch/v1dd_1196_coreg_functional_metrics/checks/` —
CodeOcean refuses to push `/scratch`, so the user copies it into the repo by hand.

**The port is complete, and so is the P0-P7 refactor.** The pipeline now lives in a fork
([[v1dd-functional-metrics-fork]]) as three notebooks plus `code/utils` and
`code/validation`; only the reproducible run remains. For what to check when it returns
see [[v1dd-metrics-asset-sanity-checks]]; for what was deliberately left open,
[[v1dd-metrics-open-questions]].

Everything below is the historical record of *how the port was validated*. The two-seed
control at the bottom is the part worth reusing elsewhere.

## Milestones

| # | Family | Status as of 2026-08-14 |
|---|---|---|
| M1 | schema truth | done — `stimulus_table` has all 12 expected columns; n_trials DG 8 / NI 8 / NI12 **40** / NM **9**; LSN template natively **8x14** with pixel values **-1/0/+1** |
| M2 | response engine | done — prefix-sum window means, verified against `xarray.sel` label semantics on 800 windows |
| M3 | natural movie | done — `frac_responsive_trials` **exact (0.0)**, `pref_img` 100 % |
| M4 | drifting gratings + SSI | done — deterministic metrics ~1e-9, `preferred_dir` 100 %, all 8 SSI variants ~1e-9 with seed-to-seed difference exactly 0 |
| M5 | natural images / images 12 | done **after a window fix** — `lifetime_sparseness` 1.1e-16, `pref_img` 99.92 % / 100 % |
| M6 | receptive fields | done — `has_rf_*` 97.1-97.7 % vs published against a **seed floor of 97.6-97.8 %**; centre regression slope 0.963-0.988, nowhere near the 1.143/1.077 that would mean the scale bug got corrected |
| M7 | packaging | done — wide feather + provenance, verified bit-for-bit against the per-family CSVs |

## How M6 was judged — the pattern generalises

Receptive fields are the one family where **no metric can be exact**, because every value
is a threshold on a bootstrap. So the check was structural rather than numerical:

1. **The degree scale, via regression of our centres on published.** Slope 0.963-0.988
   with the identical extreme values on both sides (±28.4812 altitude, ±56.1321 azimuth)
   pins grid size *and* the `(n-1)/n` `point_to_alt_azi` bug at once. Slope 1.143 or 1.077
   would have meant we corrected the bug by accident. The shortfall below 1 is regression
   dilution — both sides are noisy estimates, so the slope attenuates to about the
   reliability, and r is 0.949-0.974.
2. **Booleans against the seed floor, not against 100 %.** 97.1-97.7 % vs published sits
   *inside* the 97.6-97.8 % seed-to-seed floor.
3. **Centre disagreement measured in pixels, not degrees.** 75.6 % of shared ROIs land on
   the exact same centre, 87 % within one pixel, 93 % within two. Because the centroid is
   **unweighted**, one marginal pixel crossing the 0.25 threshold moves the centre by half
   a pixel or more — so single-pixel wobble is the expected failure mode, and seeing it
   (rather than a spread) is what says the maps agree.
4. **Per-plane rates to rule out anything structural.** The residual tilt (published calls
   ~1 pp more RFs) is even across all 12 planes, 6 of 12 in each direction, and the two
   planes where both sides find near-zero fields agree exactly.

## M7 as built

The seven CSVs are already written by M3–M6, so M7 recomputes nothing:

* wide `stimulus_metrics_M409828.feather` — 3,673 ROIs x 55 columns. Every family gets a
  column prefix **except `ssi_*` and the RF columns**, whose published names already carry
  the family; `ssi_ssi_avg` would be worse than the inconsistency. Guarded by
  `validate="one_to_one"` on each merge plus an explicit ROI-key-set equality check
  against natural movie — both verified to fire on deliberately broken input.
* `stimulus_metrics_provenance.json` — defaults once, then each family's **delta** from
  them (all empty here, because the defaults *are* the published-matching settings), plus
  seeds, sessions, package versions, git SHA, and one adversarially-chosen headline per
  family: the worst-agreeing metric beside that same metric's seed floor.
* the all-sessions switch, commented out, with the multiplier computed at runtime from
  planes-processed vs planes-in-asset rather than a guessed wall time.

Cell text was dry-run locally against synthetic frames (`scratchpad/test_m7.py`), both
with and without published tables attached — the no-published branch used to crash on
`pd.DataFrame([]).set_index("family")`.

## Methodology that made this work — reuse it

**The two-seed control is the load-bearing idea.** Everything runs twice with different
seeds and the report shows agreement-with-published *beside* agreement-with-itself. It
repeatedly turned alarming numbers into non-issues: DG `frac_responsive_trials` max
difference 0.5 looked broken until the seed column showed the identical 0.5; natural-images
`frac_responsive_trials` r = 0.78 is its noise floor (r vs seed is 0.775, i.e. we match the
original *better* than we match ourselves). Conversely it is what proved M5 was genuinely
wrong: `seed_med = 0.0` with `pub_med = 2.9e-3` means systematic, not noise.

**Join on `(column, volume, plane, roi)`, never `roi_unique_id`** — that string omits the
column and collides (56,449 distinct ids for 164,345 published rows).

**Volume is a string throughout** (volumes run 1-9 and a-f). Both sides of any comparison
must agree, and a CSV round-trip re-infers int for an all-numeric column.

See [[response-window-deferred-tuning]] for the recovered window values and the discrete
sample-count reasoning that fixed M5. Also [[v1dd-functional-metrics-fork]].
