---
name: v1dd-metrics-asset-sanity-checks
description: "What to verify when a full stimulus-metrics run returns — expected numbers, the diff that should appear, and the provenance defects each reproducible run has exposed (three in 2026-08-16, three more in 2026-09-01)."
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

## Expected shape

| | |
|---|---|
| sessions / planes / ROIs | **25 / 150 / 39,407** |
| wide table | **39,407 x 59** (57 before adding `dgw_center_azimuth`/`dgw_center_elevation`) |
| per-family CSVs | 7, each 39,407 rows |
| runtime | ~7.2 h (~2.2 min/plane; drifting gratings is 96 % of it) |
| `complete_asset` | `true`, `failed_sessions` empty |
| `differs_from_reference_config` | exactly 3 entries |

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
`failed_sessions` empty, `differs_from_reference_config` exactly 3, integrity **57/57**,
low-confidence session still 1,038 invalid (its 512 non-null SSI rows are 1,550 - 1,038).
Runtime **4.83 h** (17,400.6 s), down from 7.2 h after speedup 1. New in the asset:
`rf_maps_M409828.npz` and `dgw_center_*`. `format` is now populated per session.

**What did not run:** `VALIDATION_SESSIONS` was empty and `data_frames` was not attached,
so there was **no fidelity comparison and no seed-to-seed floor**. Every agreement number
on record still comes from the M1-M7 work. Attach the reference and set
`VALIDATION_SESSIONS` next time.

### The three new defects

1. **`ENV SWDB_CODE_VERSION = <sha>` in the Dockerfile is malformed.** Docker's legacy
   `ENV <key> <value>` form takes everything after the first space as the value, so the
   variable became `"= 17cacea..."` and `processing.json` shipped that verbatim. Write it
   as `ENV SWDB_CODE_VERSION=<sha>`, no spaces. The entry-point guard rejects empty and
   whitespace values but not malformed ones — **add a shape check (hex, 7-40 chars)**,
   since a version you cannot resolve is the thing the variable exists to prevent. The
   SHA is also hardcoded, so it needs bumping by hand on every code change.
2. **`stimulus_metrics_provenance.json` ships `git_sha: null`** while `processing.json`
   beside it carries the version. `utils/provenance.git_sha()` shells out to git and never
   reads `SWDB_CODE_VERSION`; `metadata.py` does. Give both the same ladder — two sidecars
   in one asset disagreeing is worse than either answer alone.
3. **`test_entrypoint.py` fails 2 of 14 checks in the capsule** (`resolves from git here`,
   `looks like a full sha`). Those assert the git *fallback* works, in the one environment
   that has no `.git` — the environment the variable exists for. The validation notebook
   therefore printed "unit tests failed -- fix these before reading anything below" over a
   clean asset. **This is the same failure mode as the `SkipTest` defect of 2026-08-16,
   recurring in a test written to fix that class of problem.** Skip both when
   `git rev-parse` finds no repository.

Suite now reports **15 passed, 1 skipped, 1 failed, 451 checks**.

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
