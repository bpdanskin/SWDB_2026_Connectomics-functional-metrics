---
name: cell-cell-correlations-stay-separate
description: "Decided 2026-09-03: cell-cell correlations will NOT fold into the stimulus-metrics pipeline; they become their own pipeline. Why, and where the duplicated-code inventory lives."
metadata:
  node_type: memory
  type: project
  modified: 2026-09-03T00:00:00.000Z
---

**Decided 2026-09-03.** The cell-cell correlation analysis
(`code/supplement/Functional Data Cell-Cell Correlations.ipynb`) stays a **separate
analysis**, and the user intends to make it a **separate pipeline** rather than a family
inside the stimulus-metrics pipeline. Do not propose folding it in again without new
evidence.

**What settled it** — two facts, both measured:

1. **Wrong shape.** Every stimulus-metrics family is per-*plane*, one row per ROI.
   Correlations are per-*session* and cross-plane: `load_session_dff` interpolates all 6
   planes onto a reference plane's timebase, which is the whole point, since cross-plane
   pairs carry the connectivity story. Folding it in would change the processing loop, not
   add a function.
2. **Wrong size.** Within-session pairs across all planes are **37.4 M** (against 6.88 M
   within-plane), which at 7 stimulus columns is **1.05 GB** upper-triangle and 2.09 GB in
   the notebook's symmetric convention. The whole current asset is 128 MB.

Compute was never the objection — the correlations are matmuls, minutes against a 5.1-hour
run that is 96 % grating bootstraps.

## The reuse inventory lives in HANDOFF.md

Section **"Shared or duplicated? The inventory before the pipelines split"**. Read it
before migrating either side. The headline: **nothing is shared** — the correlations
notebook imports nothing from `code/utils/`, so all seven overlaps are second copies.

Three have already drifted, and the direction of adoption is **utils** in every case
except one:

* `peek_session` — the notebook does `int(volume)`, but volumes are **1..9 and a..f**.
  Latent only because M409828 is volumes 1-5.
* session discovery — the notebook does not dedup a session present in both formats;
  `find_sessions` takes `prefer` for exactly that.
* `spectral_snr` vs `estimate_snr_white_noise_model` — a deliberate port, documented as
  agreeing in distribution but not in the last digit, because of the per-plane vs
  common-timebase difference. **Do not reconcile these.**

**The one exception**: `functional_similarity.py` generalises the notebook's
`session_correlation_table`, and the notebook does not use it — only the `Module_2b`
workshops do. There, adopt the util *into* the new pipeline. The blocker is which id the
pair rows carry: the notebook's `column`/`volume`/`pre_plane`/`pre_roi`/…, the util's
configurable `pre_pt_root_id`/`post_pt_root_id` for coregistration merges, or the
pipeline's `roi_key`, which neither uses.

Also keep out of any reproducible pipeline: the **CAVE coregistration merge**. It needs a
token — see [[co-reproducible-run-blockers]].

Related: [[v1dd-functional-metrics-fork]], [[v1dd-metrics-refactor-decisions]].
