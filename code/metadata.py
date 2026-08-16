"""Emit AIND metadata sidecars for the derived V1DD stimulus-metrics asset.

Writes three files into the asset directory, beside the tables:

* ``subject.json``          -- inherited from the input asset, unchanged
* ``data_description.json`` -- the input's institutional fields, re-stamped as derived
* ``processing.json``       -- what this run did, and with which settings

Run after the processing notebook::

    python -u /code/metadata.py --asset-dir "$ASSET_DIR" \
        --input-asset /data/409828_V1DD_Filtered --results-dir /results

Written against aind-data-schema 2.8.1.

**Inherit, never invent.** Every institutional field — who funded the work, which
institution, which investigators — is copied from the input asset's own sidecars. Nothing
in a metrics pipeline knows those things, and a plausible guess in a metadata record is
worse than a missing one, because it looks authoritative. If a required field is absent
upstream this raises and names it rather than filling it in.

The input here is a *multi-session* asset: 25 session directories, each carrying its own
sidecars. They describe the same animal and the same study, so the subject and the
institutional fields are read once and cross-checked against the rest; disagreement is an
error rather than something to resolve by picking one.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema.core.subject import Subject
from aind_data_schema_models.data_name_patterns import DataLevel
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.process_names import ProcessName

log = logging.getLogger("metadata")

#: Optical physiology, and behaviour for the running-speed signal that splits the
#: surround-suppression trials into running and stationary. Deliberately *not*
#: BEHAVIOR_VIDEOS, which the input asset also declares: nothing here reads video.
MODALITIES = [Modality.POPHYS, Modality.BEHAVIOR]

REPO_URL = "https://github.com/AllenSWDB/SWDB_2026_Connectomics"

DATA_SUMMARY = (
    "Per-ROI stimulus-response metrics for the V1DD two-photon sessions: orientation and "
    "direction tuning from full-field and windowed drifting gratings, surround "
    "suppression, selectivity across natural images and a natural movie, and receptive "
    "field position from locally sparse noise. One table per stimulus family plus a wide "
    "table joining them, keyed on (column, volume, plane, roi)."
)


# ------------------------------------------------------------------ helpers


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"expected {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _git_commit(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() or None
    except Exception:                                             # noqa: BLE001
        return None


def input_sessions(input_asset: Path) -> List[Path]:
    """Session directories carrying AIND sidecars, sorted by name.

    The sidecars sit beside the ``.nwb.zarr`` / ``.nwb`` object inside each session
    directory, so their presence is what identifies a session rather than the data file,
    whose extension varies with storage format.
    """
    if not input_asset.is_dir():
        raise FileNotFoundError(f"input asset not found: {input_asset}")
    found = sorted(d for d in input_asset.iterdir()
                   if d.is_dir() and (d / "data_description.json").is_file())
    if not found:
        raise FileNotFoundError(
            f"no session directories with a data_description.json under {input_asset}. "
            "The sidecars are expected beside each session's NWB object."
        )
    return found


def _consistent(sessions: List[Path], filename: str, keys: Tuple[str, ...]) -> Dict[str, Any]:
    """Read `filename` from the first session, asserting the others agree on `keys`.

    A silent mismatch here would attribute the whole asset to whichever session happened
    to sort first, so it is an error rather than a resolution.
    """
    first = _read_json(sessions[0] / filename)
    for other in sessions[1:]:
        raw = _read_json(other / filename)
        for key in keys:
            if raw.get(key) != first.get(key):
                raise ValueError(
                    f"{filename} disagrees between {sessions[0].name} and {other.name} "
                    f"on {key!r}: {first.get(key)!r} vs {raw.get(key)!r}. These sessions "
                    "are supposed to describe one animal and one study."
                )
    return first


# ------------------------------------------------------------------ builders


def build_subject(sessions: List[Path], strict: bool) -> Tuple[Optional[Subject], Path]:
    """Inherit subject.json, cross-checking that every session names the same animal.

    Returns a validated model where possible, so writing it back normalises the file to
    the pinned schema. If the upstream predates that schema it will not validate; unless
    ``strict``, return None so the caller copies it verbatim — preserving real provenance
    beats emitting nothing.
    """
    raw = _consistent(sessions, "subject.json", ("subject_id",))
    source = sessions[0] / "subject.json"
    try:
        return Subject.model_validate(raw), source
    except Exception as exc:                                      # noqa: BLE001
        msg = (f"upstream subject.json does not validate against the pinned "
               f"aind-data-schema: {type(exc).__name__}: {exc}")
        if strict:
            raise ValueError(msg) from exc
        log.warning("%s", msg)
        log.warning("copying subject.json verbatim instead (schema version may differ)")
        return None, source


def build_data_description(sessions: List[Path], creation_time: datetime,
                           asset_name: str) -> DataDescription:
    """Derive the data description, inheriting everything institutional.

    aind-data-schema 2.8.1 has no ``DerivedDataDescription``; a derived asset is a plain
    ``DataDescription`` with ``data_level="derived"`` and ``source_data`` naming its
    inputs.
    """
    raw = _consistent(sessions, "data_description.json",
                      ("institution", "project_name", "subject_id", "funding_source"))

    required = ("institution", "funding_source", "investigators", "project_name")
    missing = [k for k in required if raw.get(k) in (None, [], "")]
    if missing:
        raise ValueError(
            f"upstream data_description.json is missing {missing}. These name the "
            "institution and the people behind the data and are not safe to invent — "
            "supply them explicitly or fix the input asset."
        )

    inherited = ("institution", "funding_source", "investigators", "project_name",
                 "subject_id", "license", "group", "restrictions")
    kwargs = {k: raw[k] for k in inherited if raw.get(k) is not None}

    # Every input session, by the name it calls itself, so the chain back is exact.
    source_data, tags = [], set()
    for session in sessions:
        sraw = _read_json(session / "data_description.json")
        source_data.append(sraw.get("name") or session.name)
        tags.update(sraw.get("tags") or [])

    kwargs.update(
        name=asset_name,
        creation_time=creation_time,
        data_level=DataLevel.DERIVED,
        modalities=MODALITIES,
        source_data=source_data,
        data_summary=DATA_SUMMARY,
        # The union of the inputs' tags: the derived asset spans every column and volume
        # its inputs covered, so inheriting one session's pair would be wrong.
        tags=sorted(tags) or None,
    )
    try:
        return DataDescription(**kwargs)
    except Exception as exc:                                      # noqa: BLE001
        raise ValueError(
            f"could not build data_description.json from the inherited fields "
            f"({type(exc).__name__}: {exc}). The upstream file may use an older schema "
            "whose field shapes differ from the pinned version."
        ) from exc


def build_processing(asset_dir: Path, sessions: List[Path], input_asset: Path,
                     experimenters: List[str], start_time: Optional[datetime],
                     results_dir: Optional[Path], repo: Path) -> Processing:
    """One DataProcess for the metrics run, plus one for validation if it ran.

    Settings come from ``stimulus_metrics_provenance.json``, which the pipeline already
    writes — re-deriving them here would create a second source of truth that could
    disagree with the asset it describes.
    """
    prov = _read_json(asset_dir / "stimulus_metrics_provenance.json")

    end_time = datetime.fromisoformat(prov["generated_utc"])
    if start_time is None:
        # The provenance records when it was written and how long the loop took.
        start_time = end_time - timedelta(seconds=float(prov.get("wall_seconds") or 0.0))

    # A plain dict, not GenericModel(**...). The field coerces either, but constructing
    # the model directly makes `recursive_check_paths` -- the schema's AssetPath walker,
    # which follows any object exposing __dict__ and has no cycle guard -- recurse until
    # it blows the stack. Passing a dict lets pydantic build the model on its own terms
    # and the walker terminates. Structured and queryable either way, so the whole config
    # travels rather than a summary of it.
    parameters = dict(
        seed=prov.get("seed"),
        n_sessions=prov.get("n_sessions"),
        n_planes=prov.get("n_planes"),
        n_rois=prov.get("n_rois"),
        complete_asset=prov.get("complete_asset"),
        session_filter=prov.get("session_filter"),
        metric_config=prov.get("config"),
        # The three settings that depart from the historical pipeline, and what they were.
        differs_from_reference_config=prov.get("differs_from_reference_config"),
        packages=(prov.get("environment") or {}).get("packages"),
    )

    processes = [DataProcess(
        name="Stimulus response metrics",
        process_type=ProcessName.ANALYSIS,
        stage=ProcessStage.ANALYSIS,
        experimenters=experimenters,
        start_date_time=start_time,
        end_date_time=end_time,
        code=Code(
            url=REPO_URL,
            version=_git_commit(repo) or prov.get("git_sha"),
            parameters=parameters,
            input_data=[DataAsset(name=s.name, url=f"file://{s}") for s in sessions],
        ),
        # Relative to the metadata directory: the tables sit beside these sidecars.
        output_path=".",
        notes=(
            f"Computed from {prov.get('n_sessions')} sessions / {prov.get('n_planes')} "
            f"imaging planes. Deconvolved events for every family except receptive "
            f"fields, which use dF/F with a subtracted baseline. Responsiveness is "
            f"bootstrapped against each ROI's own spontaneous block, so the seed is "
            f"load-bearing and is recorded in the parameters."
        ),
    )]

    validation = _validation_summary(results_dir)
    if validation is not None:
        processes.append(DataProcess(
            name="Validation",
            process_type=ProcessName.OTHER,
            stage=ProcessStage.ANALYSIS,
            experimenters=experimenters,
            start_date_time=end_time,
            code=Code(url=REPO_URL, version=_git_commit(repo),
                      parameters=dict(validation)),
            output_path=".",
            notes=("Unit tests against synthetic data, and integrity checks over every "
                   "row of the asset. Artifacts are not part of this asset; they stay "
                   "in the validation output directory."),
        ))

    return Processing(data_processes=processes)


def _validation_summary(results_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Headline numbers from the validation run, if one is present beside the asset."""
    if results_dir is None:
        return None
    checks = results_dir / "v1dd_stimulus_metrics_validation" / "checks"
    verdict, tests = checks / "validation.json", checks / "tests.json"
    if not verdict.is_file():
        return None
    out: Dict[str, Any] = {}
    try:
        v = _read_json(verdict)
        integrity = v.get("integrity", {})
        out["integrity_checks_passed"] = integrity.get("n_passed")
        out["integrity_checks_total"] = integrity.get("n_checks")
        out["integrity_failed"] = integrity.get("failed") or []
        out["n_rois_recomputed"] = v.get("n_rois_recomputed")
    except Exception as exc:                                      # noqa: BLE001
        log.warning("could not read %s: %s", verdict, exc)
        return None
    if tests.is_file():
        try:
            t = _read_json(tests)
            out["unit_tests_passed"] = t.get("n_pass")
            out["unit_tests_failed"] = t.get("n_fail")
            out["unit_test_checks"] = t.get("checks_passed")
        except Exception as exc:                                  # noqa: BLE001
            log.warning("could not read %s: %s", tests, exc)
    return out


# ------------------------------------------------------------------ entry point


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset-dir", required=True, type=Path,
                    help="the run directory holding the metrics tables; sidecars go here")
    ap.add_argument("--input-asset", required=True, type=Path,
                    help="mounted NWB asset root, e.g. /data/409828_V1DD_Filtered")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="run directory above the asset, read for validation artifacts")
    ap.add_argument("--start-time", default=None,
                    help="ISO start of the run; derived from the provenance if omitted")
    ap.add_argument("--experimenters", nargs="*", default=None,
                    help="who ran it; defaults to the upstream investigators' names")
    ap.add_argument("--strict", action="store_true",
                    help="fail rather than copying a non-validating subject.json verbatim")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    asset_dir: Path = args.asset_dir
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"asset directory not found: {asset_dir}")
    asset_dir.mkdir(parents=True, exist_ok=True)      # write_standard_file will not

    sessions = input_sessions(args.input_asset)
    log.info("input: %d session(s) under %s", len(sessions), args.input_asset)

    creation_time = datetime.now(timezone.utc)
    start_time = datetime.fromisoformat(args.start_time) if args.start_time else None
    repo = Path(__file__).resolve().parents[1]

    experimenters = args.experimenters
    if not experimenters:
        raw = _read_json(sessions[0] / "data_description.json")
        experimenters = [p.get("name") for p in (raw.get("investigators") or [])
                         if p.get("name")]
        log.info("experimenters not given; inheriting investigators %s", experimenters)

    # --- subject
    subject, subject_src = build_subject(sessions, args.strict)
    if subject is None:
        shutil.copyfile(subject_src, asset_dir / "subject.json")
        log.info("wrote subject.json (verbatim copy)")
    else:
        subject.write_standard_file(output_directory=asset_dir)
        log.info("wrote subject.json (subject_id %s)", subject.subject_id)

    # --- data description
    dd = build_data_description(sessions, creation_time, asset_dir.name)
    dd.write_standard_file(output_directory=asset_dir)
    log.info("wrote data_description.json (%s, %d source asset(s))",
             dd.name, len(dd.source_data))

    # --- processing
    proc = build_processing(asset_dir, sessions, args.input_asset, experimenters,
                            start_time, args.results_dir, repo)
    proc.write_standard_file(output_directory=asset_dir)
    log.info("wrote processing.json (%d process(es))", len(proc.data_processes))

    for name in ("subject.json", "data_description.json", "processing.json"):
        path = asset_dir / name
        log.info("  %-26s %8.1f KB", name, path.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
