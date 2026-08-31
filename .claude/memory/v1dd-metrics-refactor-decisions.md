---
name: v1dd-metrics-refactor-decisions
description: "Design decisions for splitting the V1DD stimulus-metrics port into a pipeline notebook plus a separate validation notebook — settled 2026-08-14, before implementation."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-15T16:39:34.382Z
---

Decisions the user made for the **final** shape of the stimulus-metrics work, once the
M1–M7 port validated. Settled 2026-08-14; implementation not started. Supersedes the
milestone-driven layout in [[v1dd-stimulus-metrics-port-status]].

The goal: **three notebooks** — a processing notebook producing a CodeOcean data asset, a
validation notebook holding the evidence, and an access notebook that loads the asset and
plots it. The processing notebook's markdown describes the analysis and its reasoning
rather than the porting exercise.

## Settled

| Decision | Choice |
|---|---|
| Session scope | **All sessions**, not just the two coregistered. Removes CAVE from this pipeline entirely. |
| Known defects | **Correct both** — `rf_center_scale_bug=False`, `pref_cond_fillna=False`. Flags stay so the old behaviour is reachable. |
| Validation's seed A | **Read the shipped asset**; compute only seed B for the noise floor. |
| Unit tests | **Standalone scripts** with the existing `check()` harness, no pytest dependency. |
| `mat_version` | **Drops out** — CAVE-specific. The mounted asset name becomes the single configured input. |
| Mouse | **Derived from the NWB**, never hard-coded. Other mice are expected later. |
| Imaging depth | **Add `depth_um` to the ROI identity block if the NWB carries it** — P0 confirms first. |
| Access notebook | Coregistration is an **optional gated section** (runs offline without it); lives in `supplement/` but written workshop-ready — self-contained, assuming no context from the other two. Inherits the old plot set plus a few additions. |

## Consequences that are easy to miss

**The unit tests are not in the repository.** `git ls-files` returns zero test files and
there is no pytest config anywhere (the only `pyproject.toml` is `environment/`, the
capsule env). Everything written for M2–M6 lives in a session scratchpad. Landing them is
new work, not a move.

**All-sessions introduces per-family stimulus coverage as a real requirement.**
`stimulus_trials` returns an *empty frame* for a missing stimulus rather than raising, so
a session lacking a family flows silently into the metric functions. Both validated
sessions are 2p with all seven stimuli; the 3p sessions (letter volumes a–f) have never
been checked. `spontaneous_block` and `load_lsn_template`'s 2x2 uniformity assert are the
other two places a different session shape breaks.

**Correcting the RF scale makes validation sharper, not weaker.** Run bug-compatible to
prove fidelity against the reference tables, then assert the shipped centres equal those
*exactly* x n/(n-1). A constant-factor relationship is a stronger claim than an agreement
statistic. Watch for `pref_cond_fillna=False` changing `preferred_dir`/`preferred_sf`
dtype, since all-NaN rows may now be NaN.

**`nwb.subject` is unverified.** The M1 schema report captured `session_id`, `column`,
`volume` and the session name — nothing about the subject. Use a ladder that never
silently defaults: `nwb.subject.subject_id`, else the leading token of the session name
(`409828_2018-12-13_...`), else raise. The coverage pre-flight must dump the subject block
before this is load-bearing.

**Two mouse forms.** Reference tables say `M409828`; `roi_unique_id` is
`M{409828}_{volume}_{plane}_{roi}` off the bare number. Keep `mouse_id` (bare) and
`mouse_label` (M-prefixed) separate or you get `MM409828`.

**`depth_um` breaks column-identity with the historical tables.** Adding it to the identity
block changes `OUTPUT_COLUMNS`, so the per-family CSVs are no longer column-for-column the
old schema and the column-order test must be updated to expect it. Joins are unaffected —
they key on `(column, volume, plane, roi)` and ignore extra columns.

**The old README's SSI definitions are accurate but incomplete**, in three ways the code
documents and the README does not: a trial at exactly 1 cm/s is neither running nor
stationary (both comparisons strict); `ssi_running`/`ssi_stationary` need >=3 qualifying
trials in *both* stimuli while the `*_avg_at_pref_sf` variants have **no** minimum, so the
former are NaN far more often; and `ssi_tuning_fit` evaluates the von Mises curve
*including* the fitted baseline while the preferred-direction selection feeding it excludes
it. Inherit the list, append these. Do **not** inherit its "GLM framework" claim — there is
no regression anywhere in the receptive-field code.

**The old access notebook's coreg counts are inflated by duplicate rows.** `coreg_df`
contains exact duplicates (e.g. `864691132770893729 / 1 / 3 / 3 / 98` twice), the merge
propagates them, and `221 / 571 (38.70 %)` therefore counts some ROIs more than once.
Deduplicate on the join key.

## P0 pre-flight results (ran 2026-08-14, 47 min over 25 sessions)

The asset is a **complete 5x5 grid**: columns 1-5 x volumes 1-5, all 2p, 6 planes each,
**39,407 ROIs** (10.7x the validated 3,673). 23 Zarr + 2 HDF5. Every session has all six
families with *identical* sweep counts (DGF/DGW 192, NI 944, NI12 480, NM 29,700, LSN
1,705), 12 directions, 2 SFs, one spontaneous block. **No 3p sessions, no letter volumes** —
so the "harden for missing stimuli" work is not blocking. LSN `native_shape` is already
`[8, 14]`, so the 16x28 downsample risk never materialised.

`nwb.subject.subject_id` = `"409828"` in all 25 and agrees with the directory name.

**Depth lives in `ImagingPlane.location` as a string** (`"50 um"`), forming a clean lattice:
`depth_um = 50 + 96*(volume-1) + 16*plane` — 6 planes 16 um apart, volumes 96 um apart,
spanning **50-514 um**. (`origin_coords` exists but the probe stringified the lazy array
instead of reading it; `location` is sufficient.)

### The 0.33 s natural-images window is wrong for 23 of 25 sessions

`dt` spans **0.16123-0.16671**, 22 distinct values. The window yields exactly two samples
per trial only where 2*dt sits just above 0.33 — i.e. dt 0.16504/0.16506, which are
*precisely the two sessions the window was recovered on*. Elsewhere it catches 3 samples on
up to 4.68 % of trials (dt 0.16123) or 1 sample on up to 2.05 % (dt 0.16671). That varying
count is the exact defect that broke M5.

**Fix: `ni_response_frames=2`**, which is dt-independent. Expected to be near-identical on
the validated pair (they differ on <=0.07 % of trials) — verify by diffing against the M5
output rather than assuming. Natural movie and LSN are already frame-based; drifting
gratings' 2.0 s spans 12.0-12.4 samples and is phase-dependent within every session, but
that is inherited behaviour and belongs to [[response-window-deferred-tuning]].

### For the P7 sidecars

`genotype` Slc17a7-IRES2-Cre;Camk2a-tTA;Ai94, `sex` male, `species` Mus musculus, `age`
126, institution Allen Institute. **`date_of_birth`, `weight` and `strain` are null** and
will need supplying or omitting.

## P3: `pref_cond_fillna` was never wired up

The flag was declared and documented from the start but **never read** — the `fillna(-1)`
behaviour was hard-coded, so flipping the default would have been a silent no-op. It only
surfaced because P3 wrote a test for the flipped behaviour.

Worse, switching definitions alone would not have fixed anything: the nan-skipping argmax
fills with `-inf`, so it *also* returns index 0 for an ROI with no finite response. Both
definitions invent a preference. The real fix marks those ROIs `-1` (→ NaN), which also
stops surround suppression keying off a fabricated preferred condition.

**Lesson: a documented flag is not an implemented one.** Anything with a `*_bug` flag
needs a test that exercises both settings and asserts they differ.

## Defects found in the M7 run, to fix in the refactor

**The provenance headline ranks by a tolerance that measures float noise.**
`frac_within_tol` at rtol 1e-9 reported drifting gratings as 0.1 % agreement while
`preferred_dir` was *exactly* identical, `osi` agreed to 4e-8 (100 % at rtol 1e-6) and
`pref_dir_mean` to 1e-4 on a 0-360 scale (99.6 % at rtol 1e-6). The port agrees to ~1e-6
relative — summation order, not defect. Headline at **rtol 1e-6**, keep 1e-9 secondary.

**`ssi_tuning_fit` has no noise floor.** `fit_tuning_curves=(seed == SEEDS[0])` skips the
fit on seed B as a runtime optimization, so `n_both_finite = 0` in the seed comparison and
the most fit-dependent metric in the set is the only one with no control. Fit both seeds
or state the gap.

**`format` was null in provenance** — `targets` came from the cached `session_index.csv`,
which predates the format column. Call `vn.nwb_format(path)` directly.

**`pd.read_csv` loses 1 ULP by default.** The written CSV text is exact; pandas' fast
parser is not, so ~31 % of float values differ in the last bit from the feather until you
pass `float_precision="round_trip"`. Worth a line in the access notebook — "the two
artifacts disagree" is a costly false alarm.

## Layout

`code/utils/` is pipeline-only and **must never import from `code/validation/`** — enforced
by a test that greps for it, because the boundary erodes otherwise. Moving out of utils:
`compare_to_published`/`load_published` (→ `validation/compare.py`, renamed
`compare_tables`/`load_reference`), `schema_report`, and `checkpoint()`. Staying:
`COLUMN_ORDER`/`to_published_schema` (it is the *output* schema, not a comparison — rename
to `OUTPUT_COLUMNS`/`to_output_schema`), plus `jsonable`/`git_sha` into a new
`utils/provenance.py`.

Assets to `/results/<asset>/`, validation artifacts to
`/scratch/v1dd_stimulus_metrics_validation/`. Metadata sidecars go *inside* the asset
directory, notebooks above it — the same choice made for the CCM asset, see
[[aind-metadata-for-derived-assets]].

## Build order

P0 coverage pre-flight (read-only; gates everything) → P1 mechanical refactor → P2 harden
for missing stimuli → P3 flip the flags → P4 processing notebook → P5 validation notebook
→ P6 sidecars and full run.

**P1's gate is a bit-for-bit diff** of the seven CSVs against the current `scratch/`
outputs. A pure move must change nothing, which separates "I moved code" from "I changed
behaviour" — so when P3's numbers shift, it is unambiguously P3.

Response-window tuning for deconvolved events stays out of scope; see
[[response-window-deferred-tuning]].
