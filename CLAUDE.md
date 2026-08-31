# SWDB 2026 Connectomics — V1DD functional metrics (fork)

**Read [HANDOFF.md](HANDOFF.md) before starting work.** It covers what this repo is, how
to run the pipeline, what the shipped asset should contain, what is deliberately left
open, and the traps that have already cost debugging time. `.claude/memory/` holds the
longer notes it summarises, indexed by `.claude/memory/MEMORY.md`.

The essentials, repeated here because they are easy to get wrong:

- **This is a fork.** `origin` is `bpdanskin/SWDB_2026_Connectomics-functional-metrics` and
  may be pushed to freely. `upstream` is `AllenSWDB/SWDB_2026_Connectomics` and is
  **PR-only, never pushed to**. A near-identical clone of the shared workshop repo exists
  on the same machine — check `git remote -v` before anything git-shaped.
- **Do not run `git commit`.** Leave finished work uncommitted, say what changed and in
  which files, and stop. The user inspects diffs and commits themselves.
- **`SWDB_CODE_VERSION` must be set** before `code/run_stimulus_metrics.py` — it refuses
  to start without it, on purpose.
- **Notebooks:** edit cell `source` only; leave outputs and execution counts alone unless
  asked. Do not change run order. Do not rewrite descriptive text beyond what a
  formatting change requires without checking first — several notebooks have other
  authors.
- The pipeline lives in `code/utils/`, which **must never import from
  `code/validation/`**. A test enforces this.
