# Handoff — V1DD stimulus metrics

Written 2026-08-31, updated 2026-09-02. Carries context between sessions running in
different places (desktop, Code Ocean, Claude Code on the web). Those environments clone this repo and see nothing of each other's machines, so
anything they need has to live here.

**Read this first, then `.claude/memory/` for depth.** This file is the map; the memory
files are the territory, and each one goes deeper than the section that points at it.

---

## What this repo is

A fork of `AllenSWDB/SWDB_2026_Connectomics` carrying a port of the `allen_v1dd`
stimulus-analysis pipeline onto the published NWB assets. The original read a private
Isilon HDF5 tree that no longer exists; this reproduces its seven metric tables from the
NWB-Zarr sessions mounted in Code Ocean.

**The port is complete and has shipped two assets**, the most recent on 2026-09-01.

**The third run (2026-09-03) computed every metric and then failed writing the tuning
curves.** All the new columns are now observed rather than promised, and the checklist
numbers all reproduced — but the two new array archives and the provenance file were lost
with the crash, so a fourth run is due. The bug is fixed. See
[The 2026-09-03 run](#the-2026-09-03-run--everything-landed-except-the-two-array-archives)
before launching, and [Pending the next run](#pending-the-next-run) for the decisions the
launch still needs.

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
    stimulus_metrics.py        the metric families + MetricConfig + OUTPUT_COLUMNS
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
    probe_window_center.py   one-off: why two sessions record no aperture centre
    tests/        19 files, 677 checks, no pytest — run_all.py + a check() harness
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

As of 2026-09-03 this reports **677 checks** across 19 files in a checkout, and **674 in
the capsule** — three checks in `test_entrypoint.py` cannot run without a git repository
and now skip rather than fail. **Expect zero failures in both.** `test_reference_tables.py`
skips whenever the `data_frames` asset is not attached.

**Skip is not failure**, at two granularities: `harness.py` installs a `sys.excepthook` so
a file-level `SkipTest -> exit 2` holds even at module scope, and `harness.skip(name,
reason)` marks a single inapplicable check inside a file whose other checks still apply.
Neither counts toward `run_all.py`'s pass or fail tally.

---

## The shipped asset

**Current: `results/409828_V1DD_stimulus_metrics_2026-09-01_07-37-53/`.** The
`2026-08-16_19-40-03` run is kept beside it as the baseline `diff_runs.py` compares
against.

| | |
|---|---|
| sessions / planes / ROIs | 25 / 150 / **39,407** |
| wide table | `stimulus_metrics_M409828.feather` |
| per-family CSVs | 7, each 39,407 rows, historical column order |
| **new in this run** | `rf_maps_M409828.npz` (6.2 MB) — the per-ROI RF maps |
| runtime | **4.83 h** (17,400.6 s), down from ~7.2 h |
| seed | 0; no seed-B recomputation this run |
| sidecars | `subject.json`, `data_description.json`, `processing.json` — **inside** the asset dir |

The asset covers a complete 5x5 grid: columns 1-5 x volumes 1-5, all 2-photon, 6 planes
each. No 3-photon sessions and no letter volumes, so the "harden for missing stimuli"
work never became blocking.

### What the 2026-09-01 run added

- **`dgw_center_azimuth` / `dgw_center_elevation`** in the SSI table and the wide feather.
- **`rf_maps_M409828.npz`** — `rf_maps` (39,407 x 2 x 8 x 14 float32, dim 1 is ON/OFF),
  `roi_key`, `altitudes`, `azimuths`, `seed`. Exported pre-threshold, as planned.
- **Speedup 1** (fit only the SF that gets read), which is why the run is ~2.4 h shorter.
  Predicted ~3.6 h; came in at 4.83 h, so the fit is a smaller share of the total than
  the single-plane profile implied.
- `format` is now populated per session in provenance (it was null before).

### What this run did NOT check

`VALIDATION_SESSIONS` was empty and the `data_frames` reference asset was not attached, so
**no fidelity comparison against published tables and no seed-to-seed floor ran**. The
integrity checks (57/57) and the unit tests did run and cover every row. The agreement
numbers quoted anywhere in this file are still from the M1-M7 validation, not from this
run.

### The windowed-grating window position, now known for all 25 sessions

The open question — does the window move per column? — is answered. **It is fixed per
column and constant across that column's volumes:**

| column | azimuth | elevation |
|---|---|---|
| 1 | -8.9 | -12.4 |
| 2 | -19.6 (volume 2: **-19.8**) | -10.0 |
| 3 | +1.8 | -9.7 |
| 4 | -15.4 | -16.4 |
| 5 | +9.9 | -14.4 |

**Measured across all 25 sessions on 2026-09-03** (the aperture-centre pre-pass, not
inferred from a sample). Per-column spread of the sessions that record a centre:

| column | donors | azimuth spread | elevation spread | distinct azimuths |
|---|---|---|---|---|
| 1 | 5 | **0.0** | 0.0 | 1 |
| 2 | 4 | **0.2** | 0.0 | 2 |
| 3 | 5 | **0.0** | 0.0 | 1 |
| 4 | 4 (volumes 2-5) | **0.0** | 0.0 | 1 |
| 5 | 5 | **0.0** | 0.0 | 1 |

Three things to know before using it:

- **Only column 2 has any spread at all.** Columns 1, 3, 4 and 5 are *exactly* constant
  across their volumes — one distinct value each, to the float. So "fixed per column" is
  not an approximation for four of the five columns, and the earlier warning not to assume
  equality within a column applies specifically to **column 2 / volume 2, which sits
  0.2 deg off** the other three. Retargeting jitter, not a different window.
- That makes the two fills unequal in strength. **Column 4 / volume 1 is filled from four
  donors that agree exactly**, so its imputed centre is a unanimous value, not a
  compromise. Column 2 / volume 5 is filled from four donors of which three agree, so the
  median lands on the modal value rather than being pulled to a position no session used —
  which is why `n_donors` and `spread_*` are in the provenance: with two disagreeing
  donors a median would invent a midpoint, and the reader needs to be able to see that.
- **Two sessions ship no centre at all** — column 2 / volume 5 (906 ROIs) and column 4 /
  volume 1 (1,550 ROIs). Both have complete SSI and DGW data, so the stimulus ran; only
  the recorded position is missing. **2,456 ROIs therefore could not be filtered for RF
  containment** until the imputation. Column 4 / volume 1 is also the
  67 %-low-confidence session, which is probably coincidence — the two problems have
  nothing mechanically in common. No session records one coordinate without the other
  (`partial_sessions` is empty across all 25).

**The diameter is 30 degrees (15 degree radius)** — from the V1DD white paper, not from the
NWB, which records no size. The paper also gives the reason for the per-column position:
the window was placed to align with each column's population receptive field.

So containment is now computable, and the answer is in two halves. **Two thirds of cells
with a measured RF were probed by a window centred more than 15 degrees away** (67.1 % in
column 1), leaving only 970 ROIs — 2.5 % of the asset — where `ssi` is cleanly
interpretable. **But `ssi` does not track that distance** (r = -0.03, binned means flat to
beyond 37 degrees), and a targeting miss would push it negative, which never appears.

The distance test is probably just too blunt: RF pixels are 9.3 degrees, so a 30-degree
window spans about three of them and a centre two pixels off still overlaps. Worked
through in the access notebook.

**Both measures are now computed by the pipeline** — `sm.window_containment` emits
`dgw_rf_distance_on/off` and `dgw_rf_overlap_on/off` into the SSI table (not yet in a
shipped asset; needs the next run). Overlap is weighted by the **post-threshold** map:
the continuous pre-threshold map averages 0.086 overlap against the 0.073 a uniform
random map gives, so it mostly measures the window's share of the screen. Overlap is the
better measure — right sign, r = +0.07 against `ssi`, and far more permissive (1,572
cells at a 0.05 cut versus 970 for the centre test; 190 versus 110 in the coregistered
sessions). **Neither is used as a filter**, deliberately: the correlation is weak and the
binned profile is non-monotonic, so gating would discard most of the data on thin
evidence.

Receptive fields now run **first** in the per-plane loop so the maps exist when surround
suppression is assembled. That reorder is numerically free because `rng` is a factory
returning a freshly seeded generator per family — but confirm it with `diff_runs.py` on
the next run rather than trusting it: the expected diff is the four new columns added and
**zero changed columns**.

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

## The 2026-09-03 run — everything landed except the two array archives

The third reproducible run (`bc940fc`, results copied to
`data/results-V1DD_stimulus_metrics_2026-09-03_06-45-25/`) computed **every metric
correctly** and then **died in the tuning-curve writer cell**, on an assertion that turned
out to be wrong about the data. The processing loop had already finished, so the failure
cost the run its last three outputs and nothing else.

| landed | lost |
|---|---|
| 8 per-family CSVs, 39,407 rows each | `tuning_curves_M409828.npz` |
| `stimulus_metrics_M409828.feather`, **39,407 x 81** | `condition_means_M409828.npz` |
| `rf_maps_M409828.npz`, (39,407, 2, 8, 14) | `stimulus_metrics_provenance.json` |

**The lost archives cannot be rebuilt from the partial outputs.** They come from
trial-level arrays that live only in the processing loop's accumulators, and the CSVs are
reductions of exactly those arrays. Recovering them means rerunning — there is no
post-pass.

Every number on the [checklist](#numbers-a-rerun-must-reproduce) reproduced from the
partial outputs: 25 / 150 / 39,407, 1,038 low-confidence ROIs (2.63 %) all in column 4 /
volume 1, the depth lattice exact, 13,555 distinct `roi_unique_id` for 39,407 unique
`roi_key`, RF centres at the corrected +-32.55 / +-60.45 bounds, and the imputation
filling **exactly 2,456 ROIs** in column 2 / volume 5 and column 4 / volume 1 with their
columns' medians (column 2 showing its two distinct azimuths, -19.6 and -19.8).

### Blank sweeps are ragged, and 192 is the total, not the gratings

The assertion that killed the run said planes must agree on their blank-sweep count. They
do not, and this file had the sweep arithmetic backwards: **192 is the TOTAL number of
drifting-gratings sweeps per type, blank sweeps included** — not the non-blank total. A
session with more grey sweeps therefore shows *fewer* grating trials, since 12 x 2 x 8
condition slots only ever receive 184-187 of them.

Measured over all 25 sessions (2026-09-03, stimulus tables only):

| | `drifting_gratings_full` | `drifting_gratings_windowed` |
|---|---|---|
| total sweeps | 192 in all 25 | 192 in all 25 |
| blank sweeps | 7 (8 sessions), 8 (17) | 5 (1), 6 (1), 7 (9), 8 (14) |
| grating sweeps | 184-185 | 184-187 |

Only **13 of 25** sessions run the same number of blanks for both grating types, so this
is a per-stimulus property, not a per-session one. It also means the `trials` array is
legitimately 5-8 NaN slots short of full for every plane, which is why it was NaN-padded
from the start.

**Fixed by padding rather than raising.** `dg{w,f}_blank` is now padded to the widest
plane (8) with NaN, and `dg{w,f}_n_blank` `(n_planes,)` records the true width, keyed by
`plane_key` like the running speeds. `np.nanmean(blank, axis=1)` gives the right baseline
without consulting it. The raise was doing its job — it is how this was discovered — and
`test_tuning_export.py` now asserts the padding is NaN and not zero, that real sweeps
survive it, and that the recorded widths are the pre-pad ones. Trials, params and running
shapes still raise, because a disagreement there really would be a stimulus that ran
differently.

### One thing to decide before the rerun

The failure class is not fixed, only this instance of it. **A raise in either array-writer
cell still costs the whole run its provenance**, because provenance and the manifest run
after them and never execute. `condition_means`'s three assertions have still never
touched real data. Wrapping the two writer cells the way the per-session loop is already
wrapped — record the failure, print it loudly, carry on to provenance — would have turned
this five-hour loss into one missing file and a `!!` line. Not done here: it changes the
notebook's failure contract and `test_tuning_export.py` asserts against the current one,
so it is your call.

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

The asset was **mixed format** — 23 NWB-Zarr directories (`*.nwb.zarr`) and 2 plain HDF5
files (`*.nwb`) — and **as of 2026-09-03 the mount is 25 Zarr and 0 HDF5.** Still use
`v1dd_nwb.find_sessions()`, never a bare glob: globbing one suffix drops the other
silently. But do not trust the old symptom description — the format change turned
`find_sessions` itself into an infinite crawl (fixed; see
[Still open](#still-open--dgw_center_inferred-now-unblocked)), so **the symptom can be a
hang rather than a shorter list.** Re-check the format mix before relying on either count.

---

## Pending the next run

### Before you launch — the checklist

The three provenance defects are **fixed** (detail below). What is left is four decisions
and one manual step, none of which the code can make for you.

| | |
|---|---|
| **1. Bump the Dockerfile SHA** | Pinned to `0df1cf7228401b8024edf8378e985e801379f31a`, necessarily one commit behind whatever records this work. **Nothing derives it**, and a stale value makes the asset claim a commit it was not built from. `test_entrypoint.py` [7] checks the line's shape and that the commit exists — deliberately *not* that it equals HEAD, since that would be red on every commit and get turned off. |
| **2. Attach `data_frames` and set `VALIDATION_SESSIONS`** | The 2026-09-01 run did **neither**, so no fidelity comparison and no seed-to-seed floor have run since the M1-M7 work. Every agreement number in this file is from that older validation. This is the largest open gap in the asset's evidence, and it is free to close — it costs a dataset attachment and one variable. |
| **3. Decide `fit_all_sf`** | `False` (default) is ~2x faster and changes no published column, but leaves the unread SF's `dgw_params` / `dgf_params` **NaN by design, not by failed fit** in the exported `tuning_curves`. `True` roughly doubles the drifting-gratings time, which is ~96 % of the run. |
| **4. Decide `impute_dgw_center`** | `True` (default) fills the two sessions' aperture centre from their column median — see [Still open](#still-open--dgw_center_inferred-now-unblocked). `False` reproduces the 2026-09-01 behaviour with those 2,456 ROIs NaN. |
| **5. Know what `diff_runs.py` should say** | **Added columns, plus `dgw_center_azimuth` / `dgw_center_elevation` moving from NaN to a value for 2,456 ROIs.** That second part is the imputation and is expected. **Every `ssi` column must be unmoved** — the centre is metadata no index reads. Anything else moving means a config change, not a data change. |

Budget an extra **~6 minutes** for the new aperture-centre pre-pass (~14 s per session,
stimulus table only, no traces).

---

The three provenance defects are done and verified in both shapes — a checkout and a copy
of `code/` with no `.git`. What changed:

1. **`environment/Dockerfile`** now reads `ENV SWDB_CODE_VERSION=<sha>`, no spaces, with
   the reason in a comment above it. The legacy `ENV <key> <value>` form took everything
   after the first space as the value, which is how `"= 17cacea..."` shipped.
2. **`utils/provenance.git_sha()`** reads `SWDB_CODE_VERSION` first, verbatim, then falls
   back to short-`HEAD` as before. `stimulus_metrics_provenance.json` and
   `processing.json` now carry the same string in a capsule instead of `null` beside a
   version.
3. **`test_entrypoint.py`** skips `resolves from git here` and `looks like a full sha`
   when `git rev-parse` finds no repository, via a new per-check `harness.skip()` —
   `SkipTest` skips a whole file, which is the wrong granularity for 2 checks out of 14.
   A skipped check prints `SKIP` and counts as neither pass nor fail in `run_all.py`.

And two additions the defects argued for:

4. **`resolve_code_version()` now checks the shape** — 7-40 hex, so a short sha set by
   hand still works. The old guard rejected empty and whitespace but not malformed, which
   is exactly why the bad value passed. Its error message names the Docker `ENV` trap.
5. **`test_entrypoint.py` reads the Dockerfile.** No test read it last time, which is how
   the malformed line got out — the same failure class as the other two. It asserts the
   `ENV` line is well-formed and pins a commit that exists (not that it equals HEAD, which
   would be red on every commit).

Test counts after this work: **666 checks in a checkout, 663 in the capsule** (3 skips),
**0 failures in either** — 677 / 674 once the blank-sweep fix added its checks.
`test_run_dirs.py` also lost a ~5 % flake that took two live
`run_stamp()` calls 50 ms apart and asserted they matched — false whenever they straddled
a second boundary, and it printed the same "unit tests failed" banner over a clean asset.

### What is new

Built and unit-tested 2026-09-02. **The 2026-09-03 run produced all of it** — every
column below now exists in
`data/results-V1DD_stimulus_metrics_2026-09-03_06-45-25/`, at the promised shapes, and the
wide table came out **39,407 x 81** as the arithmetic below predicts. What that run did
*not* produce is the two array archives and the provenance file; see
[The 2026-09-03 run](#the-2026-09-03-run--everything-landed-except-the-two-array-archives).
The 2026-09-01 asset still has none of it.

Do them in **one rerun**, not several — each is a schema change and the run is ~5 h.

### New columns

| table | columns | notes |
|---|---|---|
| `natural_images`, `natural_images_12`, `natural_movie` | `reliability`, `reliability_dff` | mean pairwise between-trial correlation. **No threshold, no bootstrap** — the only responsiveness measure here that does not depend on a cut against a spontaneous null |
| `surround_supression_index` | `dgw_rf_distance_on/off`, `dgw_rf_overlap_on/off` | how much of each RF the grating aperture covered. **Reported, never used to filter.** Non-NaN exactly where `has_rf_on`/`has_rf_off` is true (7,068 / 6,657 ROIs), and they reproduce the access notebook's ad-hoc numbers: 67.1 % of column 1's RF-on cells beyond 15 deg, r = -0.02 against `ssi` |
| `surround_supression_index` | `dgw_center_inferred` | bool. True for the 2,456 ROIs whose aperture centre came from their column's median rather than from their own session |
| **`roi_summary`** (new family) | `snr`, `signal_power`, `noise_power`, `run_frac`, `spont_run_frac`, `spont_rate`, `spont_rate_run`, `spont_rate_stat`, `run_mod_dgf`, `run_mod_dgw`, `run_mod_spont` | per-ROI measures not tied to any visual stimulus |

`roi_summary` was briefly called `locomotion`; it was renamed once `snr` and `spont_rate`
joined, because those are not about locomotion. Prefix `""` in the wide table.

### New array archives

Both alongside `rf_maps_M409828.npz`, both registering themselves in
`provenance.outputs.arrays` and the manifest's missing-file check — **which `rf_maps`
previously did not**, so that asset shipped a 6 MB file provenance never mentioned.

- **`tuning_curves_M409828.npz`** — the full `(n_rois, 12, 2, 8)` per-trial grating
  responses for both types, plus blank sweeps, von Mises parameters, and per-plane running
  speeds. ~60 MiB raw. Blank sweeps are **NaN-padded to 8 columns** with the true
  per-plane count in `dg{w,f}_n_blank` — see
  [Blank sweeps are ragged](#blank-sweeps-are-ragged-and-192-is-the-total-not-the-gratings). The published grating columns are six numbers standing in for 192;
  this is what they were computed from, so `preferred_dir` and `osi` become checkable by
  eye. `roi_key` joins to the wide table; `plane_key` is `roi_key` minus its trailing
  `_{roi}` and indexes the running speeds, which have no ROI axis.
- **`condition_means_M409828.npz`** — `ni_mean` (39,407 x 118) and `ni12_mean`
  (39,407 x 12) trial-mean responses, with their image ids. ~20 MiB. Natural movie is
  deliberately absent: (39,407 x 3,600) is ~541 MiB even after averaging. This is the
  `(n_neurons, n_conditions)` matrix `functional_similarity.signal_correlation` expects
  and previously had nothing to read. See
  [[population-sparseness-from-condition-means]].

### A knob to decide before launching

**`MetricConfig.fit_all_sf`** (default `False`). `False` fits only the spatial frequency
surround suppression reads — ~2x faster over a full run, and it changes no published
column. But `tuning_curves` exports `dgw_params` / `dgf_params`, and under `False` the
unread SF is **NaN by design, not by failed fit**.

Set it `True` for a completeness run and expect roughly double the drifting-gratings time,
which is ~96 % of the total. `REFERENCE_CONFIG` sets it `True` because the original fitted
every SF, so a fast run now shows in `differs_from_reference_config` — see the count change
below.

### What to expect from the run

- **`differs_from_reference_config` is 5 entries, not 4 and not 3.** Three kinds of thing
  live in that set, and the distinction matters when reading a provenance file:
  `rf_center_scale_bug` / `pref_cond_fillna` / `ni_response_frames` are **corrections** —
  the original was wrong; `impute_dgw_center` is an **addition** — the original computed
  nothing there, so there is no defect to correct; `fit_all_sf` is a **performance**
  choice that changes no published column. A run reporting 3 or 4 is out of date.
- `diff_runs.py` against 2026-09-01 should show **only added columns**, with the existing
  ones unmoved — **except `dgw_center_azimuth` / `dgw_center_elevation`, which change from
  NaN to a value for 2,456 ROIs** (column 2 / volume 5 and column 4 / volume 1). That is
  the imputation and it is expected. Every `ssi` column must be unmoved: the centre is
  metadata that no index reads, which `test_drifting_gratings.py` [5b] asserts directly. Receptive fields now run **first** in the per-plane loop so their maps
  exist when surround suppression is assembled; that reorder is numerically free because
  `rng` is a factory returning a freshly seeded generator per family — but **confirm it
  with the diff rather than trusting it**.
- The two new archives and `rf_maps` must all appear in `provenance.outputs.arrays`.

### Still open — `dgw_center_inferred`, now unblocked

**The probe has been run (2026-09-03) and the answer is (a): imputation is justified.**
`probe_window_center.py` distinguished **(a)** the column is absent from the NWB stimulus
table, **(b)** present but all NaN, or **(c)** values exist and our extraction loses them
— only (a) and (b) justify filling in, because **(c) would be our bug** and filling would
bury it.

| session | verdict |
|---|---|
| column 4 / volume 1 | **(a)** `center_azimuth` and `center_elevation` **absent from the stimulus table entirely** |
| column 2 / volume 5 | **(a)** same — both columns absent |
| column 3 / volume 3 (control) | carries them: azimuth **1.8**, elevation **-9.7**, on all 192 rows |

Two things the probe settled beyond the verdict:

- **The columns are absent for `drifting_gratings_full` too**, not just windowed. So this
  is not "the aperture position went unrecorded" but "the centre columns are missing from
  those two sessions' stimulus tables" — an export-level omission, which is what makes it
  cleanly (a). Nothing to recover; there is no value hiding anywhere in the file.
- **The control's values match this file's per-column table exactly** (column 3 -> +1.8 /
  -9.7), which is an independent check that the table is right.
- Full-field records **0.0** in both centre columns where present — the `(0, 0)`
  placeholder `stimulus_metrics` documents, confirmed rather than assumed.

**Built 2026-09-03.** Fills from the median of the column's donors — column 2 to azimuth
-19.6 / elevation -10.0, column 4 to -15.4 / -16.4 — with a per-ROI `dgw_center_inferred`
flag, and the donor counts and spreads in provenance rather than in more columns. The
median rather than "the column's value" because column 2 / volume 2 sits 0.2 deg off the
rest of its column: the position was re-entered per session rather than shared by
construction, so asserting equality would fail on real data.

| piece | where |
|---|---|
| `sm.window_center(trials)` | the centre for one session, from **non-blank rows only** — shared by the pre-pass and `drifting_gratings_metrics`, so they cannot diverge |
| `sm.infer_window_centers(observed, config=)` | the column medians, the flags, and the provenance block |
| `MetricConfig.impute_dgw_center` | default `True`; `REFERENCE_CONFIG` `False` |
| notebook cell 12 (new) | the pre-pass, ~14 s per session, ~6 min against a ~5 h run |
| `surround_suppression_metrics(center=, center_inferred=)` | applies it per plane |

**Why a pre-pass and not a post-pass:** filling from the column needs every session in
that column, and the per-plane loop cannot know them when it reaches the first one. It
also has to happen *before* the loop rather than as a fix-up on the assembled table,
because `window_containment` is computed inside the loop from the centre — a post-pass
would fill `dgw_center_*` and leave all four `dgw_rf_*` columns NaN for exactly the 2,456
ROIs the exercise is for.

**Verified against all 25 real sessions on 2026-09-03**, running the pre-pass standalone:
`n_sessions 25, n_measured 23, n_inferred 2, n_unfilled 0`, `partial_sessions []`, and the
two fills exactly the values above. So the numbers this will produce are observed, not
predicted — unlike the rest of [Pending the next run](#pending-the-next-run). What has
*not* run is the imputation inside a full pipeline pass, which is what the rerun proves.

**A column with no donor is left NaN, not borrowed from another column.** The whole
justification is that a column agrees with itself; across columns the positions genuinely
differ. Provenance reports that as `n_donors: 0` and `n_unfilled`, so it reads as a gap
rather than a success. Same for a session recording one coordinate and not the other: not
a donor, and filled wholly from the donors rather than mixed with its own half.

#### Three bugs found on the way to that answer — all fixed 2026-09-03

The probe could not run, and neither could anything else that discovers sessions. All
three are the house failure class: **a check running in a different shape than
production.**

1. **`find_sessions()` did not return at all on this mount.** It had per-format `or`
   fallbacks:
   ```python
   zarr_paths = sorted(root.glob("*/*.nwb.zarr")) or sorted(root.rglob("*.nwb.zarr"))
   hdf5_paths = sorted(root.glob("*/*.nwb"))      or sorted(root.rglob("*.nwb"))
   ```
   **The asset is now 25 Zarr and 0 HDF5, not the "23 + 2" recorded above.** So
   `*/*.nwb` is legitimately empty, its `or` fires, and `rglob("*.nwb")` crawls 25 Zarr
   chunk trees — over 30 s to yield five results and never finishing. **The symptom is a
   hang, not the "shorter session list" this file warns about.** Fixed: the fallback fires
   only when *neither* format is found shallowly, and it is now a pruned `os.walk`
   (`_walk_sessions`) that never descends into a `.nwb.zarr`. Pruning is free — the old
   code already discarded everything found inside a store. Discovery: **never -> 0.04 s
   for 25 sessions.** Regression-tested in `test_find_sessions.py` [8] and [9], which
   spy on `os.scandir` — the one hook both `rglob` and `os.walk` go through, so the
   assertion is about the traversal rather than about which helper was called. **First
   attempt at those tests passed against the buggy code**: they asserted `_walk_sessions`
   and `os.walk` were not called, and the buggy version called neither, so both checks
   were vacuous. Verified by reverting the fix and confirming they go red.
2. **`probe_window_center.py` had never been executed.** It used
   `with vn.open_session(path)`, but that returns `(nwbfile, io)`; the context manager is
   `vn.session(path)`. It died on the first session with a `TypeError`. It now also reads
   the table via `vn.load_stimulus_table`, the same call the pipeline makes.
3. **The probe's own (c) test had a blind spot.** `stimulus_metrics` reads the centre from
   `trials.loc[~is_blank]`, but the probe counted over *all* trials. A centre recorded
   only on blank sweeps would have passed both its "present" counts while the asset still
   showed NaN — clearing a bug of ours as an absence in the data, the one conclusion the
   script exists to prevent. It now counts `n_non_nan_non_blank`, where the pipeline
   looks, and reports that case as (c). (On the control, 192 raw vs 184 non-blank differs
   by exactly the 8 blank sweeps, so blanks do carry the value here.)

---

## Open work

### 1. Three provenance defects from the 2026-09-01 run — FIXED 2026-09-03

**What was done is under [Pending the next run](#pending-the-next-run); the reasoning that
motivated each fix is kept here, because it is the reasoning and not the diff that
generalises.**

Every metric in that asset is fine. All three were *around* the numbers, and all three
were the same failure class as the run before: **a check that runs in a different shape
than production.** Kept in the past tense below because that generalisation is the point,
and the next defect of this class will not look like any of these three.

**(a) The Dockerfile `ENV` line was malformed, and the bad value shipped.**

```dockerfile
ENV SWDB_CODE_VERSION = 17cacea5a61c6b596324d6911a879b15f3ed98c4
```

Docker's legacy `ENV <key> <value>` form takes *everything after the first space* as the
value, so the variable is set to `"= 17cacea..."` — with the equals sign and space
included. `processing.json` in the shipped asset records exactly that:

```json
"version": "= 17cacea5a61c6b596324d6911a879b15f3ed98c4"
```

**Fixed:** the spaces are gone. The entry point's guard rejected *empty* and *whitespace*
values but not a malformed one, so it passed — `resolve_code_version()` now also requires
7-40 hex, and `test_entrypoint.py` reads the Dockerfile line itself, which no test did
before.

Still true, and unfixable in code: the SHA is **hardcoded in the Dockerfile** and must be
bumped by hand every time the code changes, or the asset claims a commit it was not built
from. It was correct for the 2026-09-01 run only because the commit after it touched
nothing but the Dockerfile. The test asserts the pinned commit *exists* rather than that
it is HEAD — equality would be red on every commit, so it would be turned off.

**(b) `stimulus_metrics_provenance.json` shipped `git_sha: null`** while `processing.json`
in the same directory carried the version. `utils/provenance.git_sha()` shelled out to git
and never consulted `SWDB_CODE_VERSION`, so in a capsule (no `.git`) it always returned
None. Two sidecars in one asset disagreeing about the same fact is worse than either
answer alone. **Fixed:** `git_sha()` has the same env-var-first ladder `metadata.py` has,
and a test asserts the two return the same string when git cannot answer.

**(c) `test_entrypoint.py` failed 2 of 14 checks in the capsule** — `resolves from git
here` and `looks like a full sha`. They assert that the *git fallback* works, which it
cannot in an environment with no `.git` — the exact environment the variable exists to
serve. The consequence was not cosmetic: the validation notebook printed **"unit tests
failed -- fix these before reading anything below"** over an otherwise-clean asset, which
is precisely the defect (a skipped-or-inapplicable check reported as failure) that the
previous run's `SkipTest` fix was meant to end. **Fixed:** both skip when `git rev-parse`
finds no repository, through a new per-check `harness.skip()`.

The lesson that outlived the fix: `SkipTest` was built for whole files, so when 2 checks
out of 14 became inapplicable there was no way to say so and they were left to fail. **A
suite that can only skip at one granularity will report inapplicable checks as failures at
the other**, and a suite that cries wolf gets ignored exactly when it is right.

### 2. Tune the response windows

The windows reproduce a pipeline built for slow calcium transients; this one runs on
deconvolved events, which are far sparser. Matching first was the only way to make the
port checkable. **The most defensible thing to tune first is natural images**, already
expressed as `ni_response_frames = 2`, so alternatives are integers rather than a
duration to re-derive.

Response windows here behave as **discrete sample counts, not continuous durations** —
two windows differing by less than a sample interval are often identical; two that
straddle a sample boundary differ a lot. Tune in units of `dt`, and use a scale-invariant
metric (like `lifetime_sparseness`) to detect a varying sample count.

### 3. Efficiency — one win left

**Speedup 1 is done** (2026-09-01): `drifting_gratings_metrics` takes a `fit_sf_index`,
windowed self-selects, and the notebook computes windowed first so full-field can be
passed `dgw.pref_cond_index[:, 1]`. Output is unchanged; the run went 7.2 h -> 4.83 h.

Remaining: **a data-derived `p0` (maybe 2-3x, but changes numbers).** The fixed
`p0=(0.1, 1, 180, 0.01, 1, 0.001)` starts `scale_1` one to two orders of magnitude above
event amplitudes. This can move `ssi_tuning_fit` into a different local minimum, so it
needs revalidation against the reference tables, not a bit-for-bit diff.

A caveat now visible: speedup 1 predicted ~3.6 h and delivered 4.83 h. The single-plane
profile that said drifting gratings was 96 % of runtime over-weighted the fit relative to
a full 25-session run. **Re-profile before promising a multiplier for speedup 2.**

### 4. Smaller

- `ssi_tuning_fit` has **no seed-to-seed noise floor** — `fit_tuning_curves` is skipped on
  seed B, so the most fit-dependent metric is the only one with no control.
- **Re-run the fidelity comparison.** The 2026-09-01 run attached no reference tables and
  recomputed no seed B, so the corrections and agreement statistics have not been checked
  against published data since the M1-M7 work. Attach `data_frames` and set
  `VALIDATION_SESSIONS` on the next run.
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

## An independent derivation exists — where it agrees, and where it diverges

`workshop2 - extended version.ipynb` (a teaching notebook on verifying LLM-generated code,
not a production pipeline) computes orientation tuning, OSI/gOSI and tuning width from the
same `409828_V1DD_Filtered` sessions, derived independently of this work. It covers
`drifting_gratings_full` only. Compared 2026-09-01.

**It independently confirms two things.** It found `is_soma == False` for **1,038 ROIs, all
in one session**, and concluded they should be filtered — exactly our 1,038
`pika_roi_confidence <= 0.5` in column 4 / volume 1. Two derivations reaching the same
number is the strongest evidence we have that those two criteria agree row-for-row. It also
joins coregistration on `(column, volume, plane, roi)` and calls `.drop_duplicates()`,
independently hitting the duplicate-row inflation documented here.

**The root divergence is trace type, and everything else follows from it.** They use `dff`
with a 0.5 s pre-stimulus baseline subtracted; we use `events` with no baseline. So:

- Their response window is placed at the trial-averaged PSTH peak to absorb the ~0.5-1 s
  indicator lag. Ours is a fixed 2.0 s from onset, which is only defensible **because
  deconvolution already removed that lag**.
- Their responses are signed, so their OSI can leave [0, 1] — they hit this and moved to a
  rectified gOSI. Ours does not rectify, which is safe only because events are
  non-negative. That invariant is now commented at the `gosi` computation in
  `stimulus_metrics.py`.
- They collapse direction to orientation (`% 180`) and average over TF **and SF**; we keep
  12 directions x 2 SFs and report metrics at the preferred SF. Their OSI is therefore a
  relative of ours, not the same number, and they cannot report `dsi` at all.

**They have one thing we do not: a tuning width.** Their von Mises fit is 5-parameter with
a shared kappa bounded at 50, which yields FWHH in degrees — and a real artifact they
caught, a floor at **19.1 deg** where the bound binds. Ours is 6-parameter with two
independent kappas and no bounds, so it has no such floor but is unstable enough to need
the 2,000 -> 10,000 evaluation retry. We already run this fit for `ssi_tuning_fit`, so FWHH
would be cheap to add; it is a schema change, so it has not been.

**Their verification technique worth stealing:** inspecting the argmin/argmax rows of each
metric on real data and hand-checking the arithmetic. That is how they found their defect,
and our aggregate-statistical validation would not have surfaced it.

## The 2019 white paper, and where we disagree with it

`V1DD_WhitePaper_v6.pdf` (Abbasi-Asl et al., Aug 2019) describes this dataset; our mouse is
its `Slc2`. Compared against the shipped asset 2026-09-01.

**The headline divergence is trace type, and it is the inverse of ours.** The paper says
"except the receptive field mapping, all the other analysis is performed using the **df/f**
traces. For the receptive field mapping, **events** ... are used." Our config is exactly
opposite: events for the gratings and natural stimuli, dF/F for locally sparse noise.

Our inversion is not a mistake on our side. The port reproduces the published
`data_frames/*_M409828.csv` tables to ~1e-9 on every deterministic metric, so those tables
were themselves built from events — meaning `allen_v1dd` had departed from the paper before
we touched it. **We are bug-compatible with the code, not with the paper.**

What reproduces, computed from the shipped CSVs:

| paper | paper value | ours |
|---|---|---|
| responsive to full-field gratings | 26 % | 28.3 % |
| responsive to windowed gratings | 30 % | 27.4 % (**ordering reversed**) |
| OSI full vs. windowed corr., responsive, col 1 | 0.54 | **0.555** |
| same, all unique cells | 0.27 | 0.321 |
| LSN responsiveness peak | >40 % at 306 um | 41.2 % at 274 um |
| RF centres outside the window, col 1 | "over half" | 67.1 % |
| full-field ∩ windowed responsive | ~15 % | 12.1 % |

The OSI correlation is the sharpest of these: the paper defines OSI as `|sum R e^2i0|/|sum R|`,
which is our `gosi` (0.555 vs their 0.54), not our `osi` (0.451). The metric whose
definition matches is the metric that agrees.

**Where the paper is stale or wrong:**

- Table 1 lists Slc2's Col.1/Vol.5 and Col.2/Vol.1 as having failed pre-processing. **Both
  are in our asset with full data** (965 and 914 ROIs) — recovered by reprocessing since.
- The natural-movie repeat count **contradicts itself**: 8 times on page 4, 10 times on
  page 21. Our M1 measured 9. Neither stated value is right.
- The gDSI formula is a typo — `(Rpref - Rnull)/(Rpref - Rnull)`, identically 1.
- **Cell counts differ ~3x**: 12,836 valid / 9,365 unique for Slc2 versus our 39,407 /
  30,864, with low confidence accounting for only 1,038. A segmentation-generation
  difference, and it means our population fractions are not strictly comparable to the
  paper's even where the percentages agree.

**Useful things only the paper records:** the 30-degree window diameter; gratings at 1 Hz
TF and 80 % contrast, 2 s on + 1 s grey with blank sweeps intermingled; LSN spots 9 degrees
at ~3 Hz; and the 118-image set shown as **two seeded orders, four presentations each** —
which is the origin of the `_ssa` / `_ssa_v2` distinction documented nowhere else.

**Metrics the paper has that we do not:** response reliability (mean pairwise between-trial
correlation, its Figure 18 and its natural-movie responsiveness criterion), population
sparseness, running modulation index, and event SNR. The first three are all computable from
arrays our metric functions already build and discard.

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
| `population-sparseness-from-condition-means.md` | how to compute it from the new condition-means archive, and why it is not a column |
| `co-reproducible-run-blockers.md` | the three CO blockers; all resolved here, #2 open elsewhere |
| `user-handles-commits.md` | the commit convention |

**Keep this file current.** A session running in the capsule that learns something
durable should write it here and push, since that is the only channel back to sessions
running anywhere else.
