---
name: v1dd-metrics-asset-sanity-checks
description: "What to verify when a full stimulus-metrics run returns — expected numbers, the diff that should appear, and what each of the three reproducible runs exposed (2026-08-16, 2026-09-01, and the 2026-09-03 run that lost its array archives)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-09-01T00:00:00.000Z
---

Checklist for the V1DD stimulus-metrics pipeline. Baseline: the `02d0fca` run
(2026-08-16 04:16). **The first reproducible run (2026-08-16 19:40) passed every number
below** — the diff against the baseline was exactly `+pika_roi_confidence` on all seven
tables with zero changed columns at `atol=0`. What it did *not* pass is in
"What the first reproducible run got wrong" at the bottom; all three are fixed and none
touches a metric.

## The 2026-09-03 run: every metric right, the array archives lost

The third reproducible run (`bc940fc`) computed all eight families and then **raised in
the tuning-curve writer cell**, which assumed every plane ran the same number of blank
sweeps. See [[dg-blank-sweeps-are-ragged]] — fixed by NaN-padding.

**Landed** (in `data/results-V1DD_stimulus_metrics_2026-09-03_06-45-25/`): 8 per-family
CSVs at 39,407 rows, the wide feather at **39,407 x 81**, `rf_maps` at
(39,407, 2, 8, 14). **Lost:** `tuning_curves`, `condition_means`,
`stimulus_metrics_provenance.json` — and **not rebuildable from the partial outputs**,
because they come from trial-level accumulators the CSVs are reductions of. A fourth run
is the only route.

Every number in this file reproduced from the partial outputs. The new columns are
therefore observed, not promised: `roi_summary` (11), the six `reliability` columns, the
four `dgw_rf_*`, and `dgw_center_inferred`. `dgw_rf_distance_*` is non-NaN exactly where
`has_rf_on`/`has_rf_off` is true (7,068 / 6,657) and reproduces the access notebook's
exploratory figures — 67.1 % of column 1's RF-on cells beyond 15 deg, r = -0.02 against
`ssi`.

**The failure class is not closed.** Both array-writer cells run *before* provenance and
the manifest, so a raise in either still forfeits the run's provenance. `condition_means`
still has three assertions no real data has tested.

## Expected shape

| | |
|---|---|
| sessions / planes / ROIs | **25 / 150 / 39,407** |
| wide table | **39,407 x 81** as of 2026-09-03 (59 before the new families; 57 before `dgw_center_azimuth`/`dgw_center_elevation`) |
| per-family CSVs | 7, each 39,407 rows |
| runtime | ~7.2 h (~2.2 min/plane; drifting gratings is 96 % of it) |
| `complete_asset` | `true`, `failed_sessions` empty |
| `differs_from_reference_config` | **exactly 4 entries** from 2026-09-02 (was 3) |

## The diff that should appear

```bash
python code/validation/diff_runs.py --last /results 409828_V1DD_stimulus_metrics
```

**`+dgw_center_azimuth` and `+dgw_center_elevation` on `surround_supression_index`,
zero other changed columns.** These are the new columns from the 2026-09-01 changes.
No existing metric values should move — these are new additions only.

**Note on `roi_key`:** The format changed 2026-09-01 (column and volume now
underscore-separated). If diffing against a pre-2026-09-01 asset, `roi_key` values in the
wide feather will not match; this is expected and cosmetic.

## Numbers that should reproduce exactly

* **Low-confidence ROIs: 1,038 (2.63 %), every one in column 4 volume 1** — 67 % of that
  session's 1,550. They should now be identifiable directly by `pika_roi_confidence <= 0.5`
  rather than inferred, and the two must agree row for row (P5 checks this).
* **Depth lattice** `50 + 96*(volume-1) + 16*plane`, spanning 50–514 um, 30 distinct depths.
* **`roi_unique_id` collides ~2.9x**: 13,555 distinct strings for 39,407 rows. `roi_key`
  is unique — format `M{mouse}_{column}_{volume}_{plane}_{roi}` (e.g. `M409828_1_3_2_0`;
  column and volume separated by underscore, changed 2026-09-01 from `M409828_13_2_0`).
  (Neither is in the per-family CSVs except `roi_unique_id`.)
* **Receptive-field centres within ±32.55° altitude / ±60.45° azimuth** — the *corrected*
  bounds. Seeing ±28.481 / ±56.132 would mean the historical scale shipped by mistake.
* Validation integrity: **57/57** (54 plus the three confidence checks).
* Unit tests: **17 files, 453 checks**, 16 passed + 1 skipped. `test_reference_tables.py`
  skips whenever `data_frames` is not attached — which is normal, and **skip is not
  failure**. (448 before the 5 new rf_map shape/dtype/value checks added 2026-09-01.)

## Metadata sidecars

Three files **inside** the asset directory, not beside it:
`subject.json`, `data_description.json`, `processing.json`.

* `data_description.data_level` = `derived`, `name` = the asset directory name,
  `source_data` = all 25 input session names, `modalities` = pophys + behavior (not
  behavior-videos), `tags` = the union of the inputs' Column/Volume tags.
* `processing.json` = 2 processes (metrics, then validation) if validation ran, else 1.
  `output_path` must be `"."`.
* `subject.json` inherited verbatim, subject_id `409828`, breeding info intact.

## What the first reproducible run got wrong

Every metric was right; everything that failed was *around* the numbers. All three shared
one cause — **they were invisible to the checks because the checks ran in a different
shape than production**.

1. **A skipped test reported as a failure.** `data_frames` is not attached to the capsule,
   so `require_dataset` raised `SkipTest` — at module scope, above anything `harness.main`
   could wrap. An uncaught exception exits 1, `run_all.py` read that as FAIL, and the
   validation notebook told the reader not to trust a clean asset. Fixed with a
   `sys.excepthook` in `harness.py`, so `SkipTest -> exit 2` now holds wherever it is
   raised. It must use `os._exit(2)`: raising `SystemExit` from an excepthook prints
   "Error in sys.excepthook" and still exits 1.
2. **`processing.json` recorded one process instead of two.** Two independent bugs, either
   alone sufficient. `metadata.py` ran *before* the validation notebook, so there was
   nothing to record; and `_validation_summary` looked under `--results-dir` while the
   notebook writes to `/scratch` on purpose. `test_metadata.py` built validation artifacts
   inside the results dir — a layout production never has — so it passed throughout. Fixed
   by a `--validation-dir` flag, reordering validation before metadata, and a `[2b]` test
   case in the real shape.
3. **`code.url` pointed at upstream and `commit_hash` was null.** The URL was hardcoded
   from before the fork; upstream has none of this code. The null hash is structural: a
   reproducible run copies `code/` without `.git`, so `git rev-parse` returns nothing.
   `SWDB_CODE_VERSION` and `SWDB_CODE_URL` now override.

   **Set `SWDB_CODE_VERSION` in the capsule environment before launching** — to
   `git -C <fork> rev-parse HEAD`. `run_stimulus_metrics.py` now *refuses to start*
   without it, so forgetting costs a second rather than seven hours. There is no bypass
   flag on purpose: setting the variable to a value you choose is already the escape
   hatch, and it is an honest one. The entry point is strict where `metadata.py` is
   lenient — writing `null` and warning is right for a library, wrong for a published
   asset. The CO capsule id is linked to the data asset, so nothing else CO injects needs
   capturing.

## The 2026-09-01 run

Second reproducible run, asset `409828_V1DD_stimulus_metrics_2026-09-01_07-37-53`, built
from `17cacea`. **Every metric is fine; three new provenance defects, all the same class
as last time — a check running in a different shape than production.**

Shape confirmed: 25 sessions / 150 planes / **39,407 ROIs**, `complete_asset` true,
`failed_sessions` empty, `differs_from_reference_config` exactly 3 (**4 from 2026-09-02**), integrity **57/57**,
low-confidence session still 1,038 invalid (its 512 non-null SSI rows are 1,550 - 1,038).
Runtime **4.83 h** (17,400.6 s), down from 7.2 h after speedup 1. New in the asset:
`rf_maps_M409828.npz` and `dgw_center_*`. `format` is now populated per session.

**What did not run:** `VALIDATION_SESSIONS` was empty and `data_frames` was not attached,
so there was **no fidelity comparison and no seed-to-seed floor**. Every agreement number
on record still comes from the M1-M7 work. Attach the reference and set
`VALIDATION_SESSIONS` next time.

### The three new defects — all fixed 2026-09-03

Fixed in code, **not yet exercised by a run.** The next asset is what proves them.

1. **`ENV SWDB_CODE_VERSION = <sha>` in the Dockerfile is malformed.** Docker's legacy
   `ENV <key> <value>` form takes everything after the first space as the value, so the
   variable became `"= 17cacea..."` and `processing.json` shipped that verbatim. Write it
   as `ENV SWDB_CODE_VERSION=<sha>`, no spaces. The entry-point guard rejects empty and
   whitespace values but not malformed ones — **add a shape check (hex, 7-40 chars)**,
   since a version you cannot resolve is the thing the variable exists to prevent. The
   SHA is also hardcoded, so it needs bumping by hand on every code change.
   **Done:** spaces dropped, shape check added (7-40 hex), and `test_entrypoint.py` now
   parses the Dockerfile `ENV` line — no test read it before, which is how it got out.
2. **`stimulus_metrics_provenance.json` ships `git_sha: null`** while `processing.json`
   beside it carries the version. `utils/provenance.git_sha()` shells out to git and never
   reads `SWDB_CODE_VERSION`; `metadata.py` does. Give both the same ladder — two sidecars
   in one asset disagreeing is worse than either answer alone.
   **Done:** `git_sha()` reads the variable first, verbatim, then falls back to short-HEAD.
3. **`test_entrypoint.py` fails 2 of 14 checks in the capsule** (`resolves from git here`,
   `looks like a full sha`). Those assert the git *fallback* works, in the one environment
   that has no `.git` — the environment the variable exists for. The validation notebook
   therefore printed "unit tests failed -- fix these before reading anything below" over a
   clean asset. **This is the same failure mode as the `SkipTest` defect of 2026-08-16,
   recurring in a test written to fix that class of problem.** Skip both when
   `git rev-parse` finds no repository.
   **Done**, via a new per-check `harness.skip(name, reason)`. The deeper lesson:
   `SkipTest` only skipped whole *files*, so when 2 checks out of 14 became inapplicable
   there was no way to express it and they were left to fail. **A suite that can only skip
   at one granularity reports inapplicable checks as failures at the other.**

Suite as of 2026-09-03: **18 passed, 1 skipped, 0 failed** — **677 checks in a checkout,
674 in the capsule** (3 skips), after the blank-sweep padding checks. It was 666 / 663
before those. Verified in both shapes by running it against a copy of
`code/` with no `.git`, which is the shape a reproducible run actually has. A ~5 % flake in
`test_run_dirs.py` — two live `run_stamp()` calls 50 ms apart asserted equal, false across
a second boundary — was made deterministic at the same time, since it produced the same
false "unit tests failed" banner.

## Aperture centres, measured across all 25 sessions (2026-09-03)

The pre-pass run standalone: **23 measured, 2 inferred, 0 unfilled, no partial sessions.**

**Only column 2 has any within-column spread.** Columns 1, 3, 4, 5 are exactly constant
across their volumes — one distinct float each. Column 2 spans 0.2 deg in azimuth because
volume 2 sits off the other three; elevation is constant in every column.

Consequence for the imputation: **column 4 / volume 1 is filled from four donors that
agree exactly** (a unanimous value), while column 2 / volume 5 is filled from four donors
of which three agree, so the median lands on the modal value rather than a midpoint. That
is exactly why `n_donors` and `spread_*` go into provenance — with two *disagreeing*
donors a median would invent a position no session used, and nothing in the column itself
would reveal it.

## `differs_from_reference_config` is 5 entries from 2026-09-03

`impute_dgw_center` joined `fit_all_sf`, `rf_center_scale_bug`, `pref_cond_fillna` and
`ni_response_frames`. **A run reporting 3 or 4 is now out of date, not clean.**

The five are not the same kind of thing, and a provenance file reads wrong if they are
lumped together:

- **corrections** — `rf_center_scale_bug`, `pref_cond_fillna`, `ni_response_frames`. The
  original was wrong. These change published numbers.
- **additions** — `impute_dgw_center`. The original computed nothing here, so there is no
  defect being corrected; it fills `dgw_center_*` for 2,456 ROIs that were NaN and the
  four `dgw_rf_*` columns derived from them. **No `ssi` column moves.**
- **performance** — `fit_all_sf`. Changes no published column, but leaves half of the
  exported `tuning_curves` `*_params` NaN, so it belongs in the block once those ship.

It is a different kind of entry from the other three, and the distinction matters when
reading the block. Those three are deliberate *corrections of defects* and each changes
published numbers. `fit_all_sf=False` is the fit-only-the-SF-that-gets-read speedup: it
changes **no published column**, because `ssi_tuning_fit` reads one SF per ROI either way.

It is in the block because it stopped being invisible. `tuning_curves_M409828.npz` exports
`dgw_params` / `dgf_params`, and under the speedup the unread SF is NaN — which reads as a
failed fit unless you know it was never attempted. `REFERENCE_CONFIG` sets `fit_all_sf=True`
because the original fitted every SF.

**Set `fit_all_sf=True` for a completeness run**, and expect roughly double the drifting-
gratings time, which is ~96 % of the total. `differs_from_reference_config` then drops back
to 3 entries — that is the one legitimate way to see 3 again.

## What no automated check covers

1. ~~Whether `aind-data-schema` installed.~~ **Resolved** — it resolved from
   `environment/pyproject.toml` despite being absent from `uv.lock`, and all three sidecars
   landed inside the asset directory.
2. **Whether col4/vol1's 67 % low-confidence rate is expected.** See
   [[v1dd-metrics-open-questions]] — this is a question for whoever produced the filtered
   asset, not something the pipeline can answer.
3. **Whether the checks run in production's shape.** The generalisation from defects 1-3:
   all three passed their tests and failed in the capsule. When a test constructs a
   directory layout, an attached dataset, or a git state, ask whether that is the one the
   reproducible run actually has.

Related: [[v1dd-metrics-refactor-decisions]] for why the corrections are what they are,
[[v1dd-metrics-speedups]] before considering a faster rerun.
