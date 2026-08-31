---
name: v1dd-functional-metrics-fork
description: "The stimulus-metrics work now lives in a fork — where it is, which remote is which, how to merge from upstream and how to contribute back."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-16T18:57:29.678Z
---

Forked 2026-08-16. **Supersedes the old "never push" rule**: pushing to `origin` (the
fork) is now normal. It is `upstream` that must never be pushed to directly — contribute
there by PR only.

## Where things are

| | |
|---|---|
| **fork (work here)** | `C:\Users\bethanny.danskin\Documents\GitHub\SWDB_2026_Connectomics-functional-metrics`, branch `main` |
| `origin` | `github.com/bpdanskin/SWDB_2026_Connectomics-functional-metrics` — push freely |
| `upstream` | `github.com/AllenSWDB/SWDB_2026_Connectomics` — **PR only, never push** |
| old clone | `…\SWDB_2026_Connectomics` still exists with `origin` = AllenSWDB |

**The old clone is a trap.** Same-looking directory, different remotes. Check
`git remote -v` before doing anything git-shaped; committing the metrics work there pushes
at the shared repo. [[user-handles-commits]] still applies in both.

## Collision surface with upstream

Deliberately reduced to almost nothing:

* `code/run` — **one line**, pointing at `run_stimulus_metrics.py`. The pipeline entry
  point was moved out of the shared `run_capsule.py` (restored to upstream's stub) so
  upstream can develop it freely.
* `environment/pyproject.toml` — two added dependency lines. Trivial to merge.
* Everything else the fork adds is new files: `code/utils/{v1dd_nwb,trial_responses,
  stimulus_metrics,provenance}.py`, all of `code/validation/`, `code/metadata.py`,
  `code/run_stimulus_metrics.py`, three notebooks.

The correlations notebook **no longer diverges** — its fix was upstreamed (PR #12) and the
fork then adopted upstream's implementation verbatim.

## Merging from upstream

```bash
git fetch upstream
git merge upstream/main
python code/validation/tests/run_all.py     # <- the step that matters
```

Git only sees textual conflicts. The dangerous case is upstream changing something like
`code/utils/paths.py` in a way that merges cleanly and quietly breaks data-root
resolution. 450 checks in seconds is what makes "pull commits that don't collide" a
verifiable claim rather than a hopeful one. `test_import_boundary.py` additionally proves
the pipeline still imports with `code/validation` off the path.

### The `.gitattributes` trap — this already cost a debugging cycle

`.gitattributes` has `*.ipynb -merge`, so notebooks use the binary driver: on conflict git
leaves **our** side in the working tree and marks it conflicted. `git add`ing it therefore
**silently keeps ours**, and because the merge commit records the upstream parent anyway,
`git merge` reports *Already up to date* forever afterwards. The divergence becomes
invisible.

Always pick a side explicitly:

```bash
git checkout --theirs "<notebook>"   # take upstream's
git checkout --ours   "<notebook>"   # keep ours
```

To check afterwards which side a merge actually kept, compare blobs against the parents:
`git rev-parse HEAD:<path> HEAD^1:<path> HEAD^2:<path>`.

## Working notes are public here, and that is fine

Confirmed 2026-08-31: the user is comfortable with candid working notes living in the
public fork — `HANDOFF.md`, `CLAUDE.md` and the copied `.claude/memory/` are all committed
there on purpose, including assessments of inherited code and open questions aimed at
other teams. **Do not re-raise this as a concern for this repo.** They said the caution
was worth having once; repeating it is noise.

It was still right to flag it before the first push. The judgement that changed is about
this repo's contents, not about the general habit of checking before publishing something
outward-facing.

## Contributing back to upstream

```bash
git fetch upstream
git checkout -b <topic> upstream/main
```

**Keep contributions self-contained.** Upstream has none of the fork's modules, so a fix
touching a shared notebook must not reach for `v1dd_nwb` — write the few lines inline
instead. The mixed-format PR was 9 lines in one notebook; including `v1dd_nwb.py` would
have turned it into a 500-line module contribution *and* handed upstream ownership of the
file the whole pipeline depends on. Edit only cell `source`; leave outputs and execution
counts untouched so the diff is the change and nothing else.

Push the topic branch to `origin` and open the PR from the fork.
