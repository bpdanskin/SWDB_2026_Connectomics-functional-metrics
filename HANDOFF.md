# Handoff — V1DD stimulus metrics

Written 2026-08-31 to carry context from a long desktop Claude Code session into sessions
running elsewhere (Code Ocean, Claude Code on the web). Those environments clone this
repo and see nothing of the local machine, so anything they need has to live here.

**Read this first, then `.claude/memory/` for depth.** This file is the map; the memory
files are the territory, and each one goes deeper than the section that points at it.

---

## What this repo is

A fork of `AllenSWDB/SWDB_2026_Connectomics` carrying a port of the `allen_v1dd`
stimulus-analysis pipeline onto the published NWB assets. The original read a private
Isilon HDF5 tree that no longer exists; this reproduces its seven metric tables from the
NWB-Zarr sessions mounted in Code Ocean.

**The port is complete and has shipped an asset.** What remains is tuning, efficiency,
and two data-capture additions — all listed under [Open work](#open-work). None of it is
a bug hunt.

---

## Orientation

### Repos, and the trap between them

| | |
|---|---|
| **this fork — work here** | `SWDB_2026_Connectomics-functional-metrics`, branch `main` |
| `origin` | `github.com/bpdanskin/SWDB_2026_Connectomics-functional-metrics` — **push freely** |
| `upstream` | `github.com/AllenSWDB/SWDB_2026_Connectomics` — **PR only, never push** |
| the other clone | `SWDB_2026_Connectomics` — the shared workshop repo, `origin` = AllenSWDB |

The two working copies look identical and have different remotes. **Run `git remote -v`
before anything git-shaped.** Committing metrics work in the other clone aims it at the
shared org repo.

### Working agreements

- **Do not run `git commit`.** Finish the work, say what changed, and stop. The user
  reviews diffs before committing — that review is how they stay oriented while an agent
  moves fast. A one-off "help me with this merge" is permission for *that* commit only.
- Pushing to `origin` is normal. `upstream` is PR-only, and contributions there must be
  self-contained — upstream has none of this fork's modules, so a fix touching a shared
  notebook writes the few lines inline rather than importing `v1dd_nwb`.

---

## Layout

```
code/
  run                          one line -> run_stimulus_metrics.py   (the whole upstream overlap)
  run_stimulus_metrics.py      entry point: version gate -> processing -> validation -> metadata
  metadata.py                  AIND sidecars (subject / data_description / processing)
  utils/                       pipeline only; must NEVER import from code/validation/
    v1dd_nwb.py                the only file touching hdmf_zarr / NWBHDF5IO
    trial_responses.py         numpy-only response engine (prefix sums, bootstraps)
    stimulus_metrics.py        the seven metric families + MetricConfig + OUTPUT_COLUMNS
    provenance.py              jsonable / git_sha
    paths.py                   data-root resolution (shared with upstream)
  supplement/
    V1DD Stimulus Metrics.ipynb          <- CURRENT processing notebook
    V1DD Stimulus Metrics - Access.ipynb <- loads the shipped asset and plots it
    Functional Data Stimulus Metrics.ipynb  <- LEGACY, pre-refactor. Do not edit.
  validation/
    V1DD Stimulus Metrics Validation.ipynb
    V1DD Preflight.ipynb
    preflight.py  compare.py  schema_report.py  diff_runs.py  checkpoints.py
    tests/        17 files, ~448 checks, no pytest — run_all.py + a check() harness
results/409828_V1DD_stimulus_metrics_2026-08-16_19-40-03/   the shipped asset
```

The `utils` -> `validation` import boundary is enforced by `test_import_boundary.py`,
which greps for it, because that boundary erodes on its own otherwise.

---

## Running it

### In the capsule

```bash
export SWDB_CODE_VERSION=$(git rev-parse HEAD)
python code/run_stimulus_metrics.py
```

**`run_stimulus_metrics.py` refuses to start without `SWDB_CODE_VERSION`**, deliberately
and with no bypass flag. A reproducible run copies `code/` without `.git`, so
`git rev-parse` returns nothing and the asset would ship a null commit hash. Failing in
one second beats discovering it after seven hours. Setting the variable to a value you
choose is already the escape hatch, and it is an honest one.

Order matters and is fixed in the entry point: **version gate, then processing, then
validation, then metadata.** Metadata runs last because it records what validation found.

### The test suite

Needs numpy/pandas, so it will not run in a bare environment. In the capsule:

```bash
python code/validation/tests/run_all.py
```

Expect **16 passed + 1 skipped, 448 checks**. `test_reference_tables.py` skips whenever
the `data_frames` asset is not attached — **skip is not failure**, and `harness.py`
installs a `sys.excepthook` so `SkipTest -> exit 2` holds even at module scope.

---

## The shipped asset

`results/409828_V1DD_stimulus_metrics_2026-08-16_19-40-03/`

| | |
|---|---|
| sessions / planes / ROIs | 25 / 150 / **39,407** |
| wide table | `stimulus_metrics_M409828.feather`, 39,407 x 57 |
| per-family CSVs | 7, each 39,407 rows, historical column order |
| runtime | ~7.2 h (~2.2 min/plane; drifting gratings is 96 % of it) |
| sidecars | `subject.json`, `data_description.json`, `processing.json` — **inside** the asset dir |

The asset covers a complete 5x5 grid: columns 1-5 x volumes 1-5, all 2-photon, 6 planes
each. No 3-photon sessions and no letter volumes, so the "harden for missing stimuli"
work never became blocking.

### Numbers a rerun must reproduce

- **Low-confidence ROIs: 1,038 (2.63 %)**, every one in column 4 / volume 1 — 67 % of
  that session's 1,550. Identify by `pika_roi_confidence <= 0.5`.
- **Depth lattice** `50 + 96*(volume-1) + 16*plane`, 50-514 um, 30 distinct depths.
- **`roi_unique_id` collides ~2.9x**: 13,555 distinct strings for 39,407 rows. Use
  `roi_key`, or join on `(column, volume, plane, roi)`.
- **RF centres within +-32.55 deg altitude / +-60.45 deg azimuth** — the *corrected*
  bounds. Seeing +-28.481 / +-56.132 means the historical scale bug shipped by mistake.
- Validation integrity **57/57**.

To compare a fresh run against the shipped one:

```bash
python code/validation/diff_runs.py --last /results 409828_V1DD_stimulus_metrics
```

Nothing in a clean rerun should move. If a metric moved, suspect a config change before
suspecting the data.

---

## NWB access reference

Verified from stored output on session 794964451 (column 1 / volume 3). Every plane
processing module carries the same eight interfaces:

```python
['demixed', 'dff', 'events', 'image_segmentation', 'images',
 'neuropil_corrected', 'neuropil_fluorescence', 'raw']
```

```python
mod = nwbfile.processing[f'plane-{plane}']

dff        = mod['dff'].data[:]        # (n_frames, n_rois), dF/F in %
events     = mod['events'].data[:]     # same shape/timebase, deconvolved (L0), a.u.
timestamps = mod['dff'].timestamps[:]  # ~6.1 Hz
roi_table  = mod['dff'].rois.to_dataframe()
```

`events` is what every metric family uses **except receptive fields**, which use `dff`.
If you compare your own analysis against the shipped tables, match the trace type — the
two give visibly different responsiveness fractions on the same cells.

Interface *names* are verified. The pipeline *ordering* below is inferred from the
standard Allen ophys pipeline, not read from file metadata — print `mod[k].description`
to settle it:

`raw` -> `demixed` -> (`neuropil_fluorescence` as the contaminant) -> `neuropil_corrected`
-> `dff` -> `events`. `image_segmentation` holds the ROI masks; `images` holds summary
projections. Neither is a timeseries.

Three structural facts that differ from the old `allen_v1dd` client:

1. **Traces are `(n_frames, n_rois)`** — transposed relative to `get_traces`. This module
   keeps NWB's orientation because that is what `trial_responses.prefix_sums` wants.
2. **One `intervals['stimulus_table']`, not seven per-stimulus frames.** Every stimulus is
   concatenated with a union of columns and NaN where a parameter does not apply — so
   "blank sweep" can no longer be detected as "any NaN in the row". Pass the parameter
   columns explicitly.
3. **Running speed is already differentiated**, cm/s, on its own ~59 Hz timebase.

The asset is **mixed format**: 23 NWB-Zarr directories (`*.nwb.zarr`) and 2 plain HDF5
files (`*.nwb`). Use `v1dd_nwb.find_sessions()`, never a bare glob — globbing one suffix
drops the other silently, and the symptom is a shorter session list rather than an error.

---

## Open work

Ordered by what the next fresh run should capture, since those are the items that
otherwise force a second 7-hour run.

### 1. Capture the windowed-grating window geometry

`ssi_*` compares a windowed grating response against a full-field one, but **nothing in
the asset records where the window was or how big it was**. So the index cannot be
separated from a targeting miss: a cell whose receptive field falls outside the window
reads as "suppressed" when it was simply not stimulated.

- `preflight.py`, `_stimulus_coverage`: collect `center_azimuth` / `center_elevation`
  beside `direction` / `spatial_frequency`. Two lines, and it answers the per-column
  question for all 25 sessions permanently. **Nothing we built reads those columns
  today**, which is why this is still unknown.
- `stimulus_metrics.surround_suppression_metrics`: carry the centre out as
  `dgw_center_azimuth` / `dgw_center_elevation` (per-session constants).

Known so far: one fixed window position per session (24 conditions = 12 directions x 2
SFs, no location factor). In **column 1, volumes 3 and 5** it is azimuth -8.9 deg,
elevation -12.4 deg; full-field rows carry a `(0.0, 0.0)` placeholder, which is how you
tell them apart. **Only those 2 of 25 sessions were ever checked**, and both are column 1
— so they test volume-to-volume stability and nothing else. Acquisition notes say the
position was tuned per session. **Diameter is absent from the stimulus table entirely**
and must come from acquisition-side metadata.

### 2. Retain the per-cell 2D receptive-field map

The full per-ROI subfield map is already built in `receptive_field_metrics` and thrown
away on return:

```python
rf = frac.reshape(plane.n_rois, 2, n_rows, n_cols)   # dim 1: 0 = ON, 1 = OFF
```

Each pixel is the fraction of that pixel's presentations producing a response above that
ROI's own bootstrapped spontaneous 95th percentile. Exporting it is **a retention change,
not an analysis step — no extra computation**, and the shipped centres already derive
from this array.

- Export from **before** `frac[frac < rf_frac_thresh] = 0.0` (threshold 0.25). The
  post-threshold version is recoverable from the continuous one in a line; the reverse is
  not, and the continuous version is what a familiar RF figure looks like.
- It does not fit the per-ROI CSV schema: 2 x 8 x 14 = 224 floats per ROI, ~35 MB as
  float32 over 39,407 ROIs. Wants an `.npz` holding `(n_rois, 2, 8, 14)` plus a `roi_key`
  index.
- Three things must travel with it or it is uninterpretable: `altitudes`/`azimuths` from
  the `lsn` dict; the **seed**, since per-pixel significance is bootstrapped; and the fact
  that `frac[~plane.is_valid] = 0.0` means **a blank map is "excluded", not "no RF"**.

### 3. Tune the response windows

The windows reproduce a pipeline built for slow calcium transients; this one runs on
deconvolved events, which are far sparser. Matching first was the only way to make the
port checkable. **The most defensible thing to tune first is natural images**, already
expressed as `ni_response_frames = 2`, so alternatives are integers rather than a
duration to re-derive.

Response windows here behave as **discrete sample counts, not continuous durations** —
two windows differing by less than a sample interval are often identical; two that
straddle a sample boundary differ a lot. Tune in units of `dt`, and use a scale-invariant
metric (like `lifetime_sparseness`) to detect a varying sample count.

### 4. Efficiency — two unimplemented wins, ~2x

Drifting gratings is 96 % of runtime, and inside it the cost is `vonmises_two_peak_fit`
(~1,220 fits per plane at ~134 ms).

- **Fit only the SF that gets read (~2x, output identical).** `tuning_params` has exactly
  one consumer, `surround_suppression_metrics`, which reads one SF per ROI per grating
  type — so with `n_sf = 2`, **half of every fit is discarded**. Windowed can self-select
  (`pref_cond_index` is computed before the fit block); full-field needs a new
  `fit_sf_index` argument fed from `dgw.pref_cond_index[:, 1]`. Verify bit-for-bit.
- **A data-derived `p0` (maybe 2-3x, but changes numbers).** The fixed
  `p0=(0.1, 1, 180, 0.01, 1, 0.001)` starts `scale_1` one to two orders of magnitude
  above event amplitudes. This can move `ssi_tuning_fit` into a different local minimum,
  so it needs revalidation, not a diff. **Do the first one first, and separately.**

### 5. Smaller

- `ssi_tuning_fit` has **no seed-to-seed noise floor** — `fit_tuning_curves` is skipped on
  seed B, so the most fit-dependent metric is the only one with no control.
- The `_ssa` vs `_ssa_v2` gotcha is currently absent from every notebook.
- The access notebook's asset resolver could take a repo-relative fallback so it runs
  outside the capsule.

### Not open — do not reopen without new evidence

The two corrections (`rf_center_scale_bug=False`, `pref_cond_fillna=False`) and the
natural-images switch to a frame count are settled and validated.

---

## Deliberate imperfections

These are shipped on purpose. Do not "fix" them unprompted.

- **`ssi_tuning_fit` evaluates the fitted curve including its baseline offset**, while the
  preferred direction feeding it is chosen with the baseline subtracted. Inherited and
  reproduced deliberately.
- **`roi_key` is only in the wide feather**, not the per-family CSVs, which keep the
  historical column set. Adding it is a schema change and another 7 h rerun.
- **`pref_img` for natural movie is approximate by construction** — the window spans
  several 1/30 s frames, so read it as "around here in the clip".
- **`_ratio` returns 0 on a zero denominator while `_metric_index` returns NaN.** Two
  lines, opposite conventions, both inherited. Unifying them would change published
  columns for no measured benefit.

---

## Methodology worth reusing

**The two-seed control is the load-bearing idea.** Everything runs twice with different
seeds, and the report shows agreement-with-published *beside* agreement-with-itself. It
repeatedly turned alarming numbers into non-issues — drifting-gratings
`frac_responsive_trials` max difference 0.5 looked broken until the seed column showed the
identical 0.5; natural-images `frac_responsive_trials` r = 0.78 **is** its noise floor
(r vs seed is 0.775, i.e. we match the original better than we match ourselves).
Conversely it is what proved a real defect: `seed_med = 0.0` against `pub_med = 2.9e-3`
means systematic, not noise.

**Judge threshold metrics structurally, not numerically.** Receptive fields can never be
exact, so M6 was judged by regressing our centres on published (slope 0.963-0.988 pins
grid size and the `(n-1)/n` scale bug at once; 1.143 would have meant we corrected it by
accident), by comparing booleans against the seed floor rather than 100 %, and by
measuring centre disagreement in **pixels** — 75.6 % identical, 87 % within one pixel.
Because the centroid is unweighted, single-pixel wobble is the *expected* failure mode,
and seeing wobble rather than spread is what says the maps agree.

**The generalisation that cost the most to learn:** every one of the three provenance
defects in the first reproducible run **passed its tests and failed in the capsule**,
because the tests ran in a different shape than production — a fixture built inside the
results directory, a `SkipTest` raised at module scope, a git state that a reproducible
run does not have. When a test constructs a directory layout, an attached dataset, or a
git state, ask whether that is the one the run actually has.

---

## Traps that have already cost time

- **`.gitattributes` has `*.ipynb -merge`.** On conflict git leaves **our** side in the
  working tree and marks it conflicted, so `git add`ing it silently keeps ours — and
  because the merge commit still records the upstream parent, `git merge` reports
  *Already up to date* forever after. The divergence becomes invisible. Always pick a
  side explicitly with `git checkout --theirs|--ours <notebook>`, and verify afterwards
  with `git rev-parse HEAD:<path> HEAD^1:<path> HEAD^2:<path>`.
- **`pd.read_csv` loses 1 ULP by default** — the written CSV text is exact, pandas' fast
  parser is not, so ~31 % of float values differ in the last bit from the feather until
  you pass `float_precision="round_trip"`. "The two artifacts disagree" is a costly false
  alarm.
- **Volume is a string throughout** (volumes run 1-9 and a-f). A CSV round-trip re-infers
  int for an all-numeric column, so both sides of any comparison must agree.
- **Two mouse forms.** Reference tables say `M409828`; `roi_unique_id` is built off the
  bare number. Keep `mouse_id` and `mouse_label` separate or you get `MM409828`.
- **A documented flag is not an implemented one.** `pref_cond_fillna` was declared and
  documented from the start but never read, so flipping it would have been a silent no-op.
  Anything with a `*_bug` flag needs a test that exercises both settings and asserts they
  differ.
- **`Code.parameters` must be a plain dict**, never `GenericModel(**kwargs)` — the latter
  sends aind-data-schema's AssetPath walker, which has no cycle guard, into unbounded
  recursion and surfaces as `RecursionError` at `write_standard_file`, nowhere near the
  cause.

---

## An open question for someone else

**Why is 67 % of column 4 / volume 1 low-confidence?** 1,038 of 1,550 ROIs at
`pika_roi_confidence <= 0.5`, where every other session is unremarkable. Nothing else
about the session stands out — 6 planes, mid-range ROI count, `dt` in the middle of the
spread, all six stimuli present. It has the second-newest filtering date in the asset
(`2026-04-16` where most are `2026-04-09`), which is a thread worth pulling but a sample
of one.

This is a question for whoever produced the filtered NWB asset, not something the metrics
pipeline can settle. Until then, a population average over all ROIs is weighted by one
session's segmentation quality — the access notebook says so, and the column makes it
filterable.

---

## Where the rest lives

`.claude/memory/` holds the full notes this file summarises. `MEMORY.md` indexes them.
Start there when a section above is too terse:

| File | For |
|---|---|
| `v1dd-functional-metrics-fork.md` | remotes, merging from upstream, contributing back |
| `v1dd-metrics-asset-sanity-checks.md` | the full rerun checklist and the three provenance defects |
| `v1dd-metrics-open-questions.md` | every deferred decision, at length |
| `v1dd-metrics-refactor-decisions.md` | why the pipeline is shaped the way it is; P0 preflight results |
| `v1dd-stimulus-metrics-port-status.md` | the M1-M7 validation record |
| `v1dd-metrics-speedups.md` | the profile and the two speedups |
| `response-window-deferred-tuning.md` | recovered window values and the discrete-sample reasoning |
| `aind-metadata-for-derived-assets.md` | the metadata recipe and aind-data-schema 2.8.1 gotchas |
| `co-reproducible-run-blockers.md` | the three CO blockers; all resolved here, #2 open elsewhere |
| `user-handles-commits.md` | the commit convention |

**Keep this file current.** A session running in the capsule that learns something
durable should write it here and push, since that is the only channel back to sessions
running anywhere else.
