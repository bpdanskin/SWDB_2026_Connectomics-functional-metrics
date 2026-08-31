---
name: aind-metadata-for-derived-assets
description: "Working recipe for emitting AIND subject/data_description/processing JSON from a Code Ocean pipeline, with the aind-data-schema 2.8.1 API gotchas that cost time to discover."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-17T04:22:01.326Z
---

Ran successfully in Code Ocean on 2026-08-13 to produce the V1DD CCM asset. **Reference
implementation: `SWDB_2026_Connectomics_v1dd_ccm_reprun`, branch `add-metadata`,
`code/metadata.py`** — copy it as the starting point for the next derived asset.

## Shape of the solution

- `code/metadata.py` — a plain module, not a notebook (no exploratory value, easier to test).
- `code/run` computes ONE timestamp (`RUN_EPOCH`) and derives from it both the ISO
  `--start-time` and the `YYYY-MM-DD_HH-MM-SS` asset-name stamp, so the directory name,
  `data_description.name`, and `processing.json` timings cannot disagree.
- Layout that worked: **metadata sidecars go INSIDE the asset subdirectory**
  (`/results/<asset-name>/`) next to the data; executed notebooks and any provenance
  sidecar stay at the `/results` top level, outside the asset. This means the script needs
  two paths — `--output-dir` (where JSONs are written) and `--results-dir` (where it READS
  run artifacts from). Easy to conflate; the tests assert the split both ways.
- **Inherit, never invent.** `subject.json` is copied from the input asset;
  `institution` / `funding_source` / `investigators` / `project_name` are lifted from the
  input's `data_description.json`. The script raises naming the missing field rather than
  guessing. Nothing in an EM/connectomics pipeline knows which animal the volume came from.

## aind-data-schema 2.8.1 gotchas

Pin choice: `exclude-newer` in `environment/pyproject.toml` was 2026-07-30 and 2.9.0
shipped 2026-08-01, so 2.8.1 was the newest release inside the freeze. **Always check
`exclude-newer` before picking a version.**

- The file is **`data_description.json`**, not `description.json`.
- **`DerivedDataDescription` does not exist in 2.x.** A derived asset is a plain
  `DataDescription` with `data_level=DataLevel.DERIVED` + `source_data=[<input name>]`.
- Required: `Subject`{subject_id, subject_details}; `MouseSubject`{sex, date_of_birth,
  strain, species, genotype, source}; `DataDescription`{creation_time, institution,
  funding_source, data_level, investigators, project_name, modalities};
  `DataProcess`{process_type, stage, code, experimenters, start_date_time}.
- `experimenters` is `List[str]` but `investigators` is `List[Person]` — don't mix them.
- **`MouseSubject.source = Organization.AI` additionally requires `breeding_info`**
  ("bred in house"). Use an external org (e.g. `Organization.JAX`) in test fixtures to avoid it.
- `DataProcess.output_path` is an `AssetPath` and **must be relative to the metadata
  directory**, not absolute. `"."` when the data shares the metadata's directory.
- `write_standard_file(output_directory=...)` does **not** create the directory — mkdir first.
- The **name regex is only enforced when `name is None` AND `data_level == RAW`**. An
  explicit name on a derived asset is never validated, so a non-conformant name passes
  silently. `DataRegex.DATA` = `<label>_<YYYY-MM-DD_HH-MM-SS>`; the readable compromise
  `409828_V1DD_CCM_materialization_1196_<stamp>` satisfies it.
- `ProcessName` has 47 terms, none connectomics-specific; `"Analysis"` is the closest fit
  (alternatives `"Other"`, `"Pipeline"`).
- **Pass `Code.parameters` as a plain dict, never `GenericModel(**kwargs)`.** The field
  coerces either, but a directly-constructed GenericModel makes
  `aind_data_schema.utils.validators.recursive_check_paths` — the AssetPath walker, which
  follows any object exposing `__dict__` and has no cycle guard — recurse until it blows
  the stack. `RecursionError` at `write_standard_file`, nowhere near the cause. A dict
  lets pydantic build the model itself and the walker terminates.
- Multi-session input assets: sidecars live *inside each session directory* beside the
  `.nwb.zarr`/`.nwb`, so detect sessions by the presence of `data_description.json`, not
  by data-file extension. Read subject and institutional fields once and **cross-check
  the rest**; `source_data` takes every session name, and `tags` the union across them.

## Recording dependency and database provenance

- Package pinned to a **git branch** is not reproducible — branches move. Read PEP 610
  `direct_url.json` from the installed dist (`importlib.metadata.distribution(x)
  .read_text("direct_url.json")`) to get `url`, `requested_revision` and the resolved
  **`commit_id`**. Verified this works for VCS installs.
- `Code.core_dependency` is a `Software(name, version)` with **no URL field** — put the
  git URL/branch/commit in `Code.parameters` (a `GenericModel`, `extra="allow"`, so
  arbitrary keys validate). `Code.input_data` is `List[DataAsset]` for input declarations.
- For a live database query (CAVE), have the ETL notebook write a small provenance sidecar
  (`/results/cave_provenance.json`) with datastack, server, version AND timestamp;
  `metadata.py` reads it afterwards. `client.materialize.get_timestamp(version)` returns a
  datetime in caveclient 8.2.1. Record both version and timestamp — the version is what
  filenames key on, the timestamp is what actually survives renumbering.

## Two things a reproducible run breaks that an interactive run does not

Both shipped in the first V1DD stimulus-metrics reproducible run and only showed up in the
published sidecars — see [[v1dd-metrics-asset-sanity-checks]].

- **`git rev-parse` returns nothing.** A reproducible run copies `code/` without `.git`, so
  a best-effort `_git_commit()` yields `None` and the asset ships `commit_hash: null`.
  aind-data-schema warns about it and continues. Take the version from an env var
  (`SWDB_CODE_VERSION`) first and **set it before launching**, or accept an unstamped asset.
- **Order the steps so metadata is last.** Anything `processing.json` reports on —
  validation especially — must have already run. Metadata written before validation can
  only ever describe a validation that had not happened, and it fails silently: one process
  where there should be two, with no error.

Also: a `Code.url` hardcoded to the repo the work *started* in goes stale the moment it is
forked. Point it at the repository that actually holds the code, and make it overridable.

## Testing without the real data

Synthesize the upstream asset using the schema itself (build `Subject` / `DataDescription`
and `write_standard_file` them into a temp dir), then assert the outputs round-trip. Also
cover the degraded paths: missing sidecar, non-validating upstream subject, missing
institutional fields. On Windows keep the fixture root SHORT — a long asset name plus a
deep temp path blows past MAX_PATH (260) and surfaces as a baffling `FileNotFoundError`
on write.

See [[co-reproducible-run-blockers]] — this fork also demonstrates fixes for two of those:
`code/run` uses `jupyter nbconvert --to notebook --execute --output-dir /results` as the
entrypoint, and CAVE auth is non-interactive via `.codeocean/secrets.json` +
`os.environ["CUSTOM_KEY"]` passed as `auth_token=`. Also [[v1dd-functional-metrics-fork]].
