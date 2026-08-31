---
name: co-reproducible-run-blockers
description: Three known blockers preventing a Code Ocean Reproducible Run of the V1DD functional notebooks; deferred by the user as of 2026-08-07.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-08T03:34:36.932Z
---

**RESOLVED as of 2026-08-16 for the stimulus-metrics pipeline** (see
[[v1dd-functional-metrics-fork]]). Kept because the reasoning still applies to any other
capsule in this family, and because #2 remains open wherever CAVE is queried headlessly.

* **#1 entrypoint** — `code/run` now calls `code/run_stimulus_metrics.py`, which
  nbconverts the notebooks and then runs `metadata.py`. `run_capsule.py` was restored to
  upstream's stub to keep it off the merge surface.
* **#2 CAVE token** — **not applicable** to the metrics pipeline, which never queries
  CAVE. Only the access notebook's optional final section does, and it degrades
  gracefully when the import or token is missing. Still unsolved for the CCM capsule.
* **#3 output target** — the notebook now reads `SWDB_OUTPUT_TARGET`, defaulting to
  `scratch`; the entry point sets it to `results`. Interactive runs therefore cannot
  accidentally produce something that looks like a captured asset.

---

The original note, as background:

As of 2026-08-07, `code/workshops/Functional Data Cell-Cell Correlations.ipynb` runs
cleanly in the CodeOcean *interactive* session but a **Reproducible Run** would not work.
The user knows about these and deferred them ("straightforward to solve when I am ready") —
do not treat them as bugs to fix unprompted.

1. **Entrypoint is a no-op.** `code/run` calls `code/run_capsule.py`, whose `run()` is
   `pass`. A reproducible run builds the image, mounts data, executes nothing, captures an
   empty `/results`. Fix: have it execute the notebook, e.g.
   `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir /results <nb>`
   (nbconvert arrives via `jupyterlab`; papermill is not a dependency).

2. **`CAVEclient.setup_token(...)` is interactive.** Headless runs have no stdin/browser,
   so it hangs or raises. Needs the token as a capsule secret injected via env var plus a
   non-interactive auth path. Module_1's cell 5 markdown links the DataBook CodeOcean
   token-setup page — follow that, since it's what students are told.

3. **`output_target = "scratch"`** in cell 3. `/scratch` is discarded; only `/results` is
   captured. The knob exists, it just needs flipping for a capture run.

**Resolved / non-issues** (do not re-raise):
- CAVE materialization version is now pinned: `CAVEclient(datastack_name="v1dd_public", version=mat_version)`.
- Memory: the capsule has 8 GB; the notebook peaks around 1 GB and has never come close.
- Wall time (~20+ min, dominated by a 5 min session-index glob) is acceptable for a one-off.

**Optional improvement, not a blocker:** a reproducible run needs determinism, not
discovery. Reading a cached `session_index.csv` instead of globbing all 23 sessions would
cut ~5 min and remove drift if the functional asset gains sessions. The interactive
notebook can keep the glob.

See [[v1dd-functional-metrics-fork]] — commit locally, never push.
