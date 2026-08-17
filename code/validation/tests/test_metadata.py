"""code/metadata.py against a synthetic multi-session input asset.

The fixture is built from the shape of the real V1DD sidecars, so the inheritance is
exercised on the field layout that actually exists rather than an idealised one. Nothing
here needs the NWB data — only the JSON beside it.

Skips when aind-data-schema is absent: it is a capsule dependency, not one this test
suite can assume.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from harness import REPO, SkipTest, check, fails, summary

try:
    import aind_data_schema  # noqa: F401
except ImportError:
    print("  SKIP  aind-data-schema not installed (capsule dependency)")
    raise SystemExit(2)

sys.path.insert(0, str(REPO / "code"))
import metadata as md                                             # noqa: E402

SUBJECT = {
    "describedBy": "https://raw.githubusercontent.com/AllenNeuralDynamics/aind-data-schema/main/src/aind_data_schema/core/subject.py",
    "notes": None, "object_type": "Subject", "schema_version": "2.0.12",
    "subject_id": "409828",
    "subject_details": {
        "alleles": [], "object_type": "Mouse subject",
        "breeding_info": {"breeding_group": "Slc17a7-IRES2-Cre;Camk2a-tTA;Ai94(V1DD)",
                          "maternal_genotype": "", "maternal_id": "372850",
                          "object_type": "Breeding info",
                          "paternal_genotype": "Slc17a7-IRES2-Cre/wt",
                          "paternal_id": "359727"},
        "date_of_birth": "2018-07-03",
        "genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai94(TITL-GCaMP6s)/wt",
        "housing": None, "restrictions": None, "rrid": None, "sex": "Male",
        "source": {"abbreviation": "AI", "name": "Allen Institute",
                   "registry": "Research Organization Registry (ROR)",
                   "registry_identifier": "03cpe7c52"},
        "species": {"common_name": "House mouse", "name": "Mus musculus",
                    "registry": "National Center for Biotechnology Information (NCBI)",
                    "registry_identifier": "NCBI:txid10090"},
        "strain": {"name": "C57BL/6J", "registry": "Mouse Genome Informatics (MGI)",
                   "registry_identifier": "MGI:3028467", "species": "Mus musculus"},
        "wellness_reports": []},
}


def data_description(name, column, volume):
    return {
        "creation_time": "2026-04-09T04:59:00.471674Z", "data_level": "derived",
        "data_summary": None, "object_type": "Data description",
        "describedBy": "https://raw.githubusercontent.com/AllenNeuralDynamics/aind-data-schema/main/src/aind_data_schema/core/data_description.py",
        "schema_version": "2.3.3", "name": name, "subject_id": "409828",
        "project_name": "V1 Deep Dive", "license": "CC-BY-4.0",
        "group": None, "restrictions": None,
        "funding_source": [{"fundee": None, "grant_number": None,
                            "object_type": "Funding",
                            "funder": {"abbreviation": "AI", "name": "Allen Institute",
                                       "registry": "Research Organization Registry (ROR)",
                                       "registry_identifier": "03cpe7c52"}}],
        "institution": {"abbreviation": "AIBS",
                        "name": "Allen Institute for Brain Science",
                        "registry": "Research Organization Registry (ROR)",
                        "registry_identifier": "00dcv1019"},
        "investigators": [
            {"name": "Saskia de Vries", "object_type": "Person",
             "registry": "Open Researcher and Contributor ID (ORCID)",
             "registry_identifier": "0000-0002-3704-3499"},
            {"name": "Clay Reid", "object_type": "Person",
             "registry": "Open Researcher and Contributor ID (ORCID)",
             "registry_identifier": None}],
        "modalities": [{"abbreviation": "pophys", "name": "Planar optical physiology"},
                       {"abbreviation": "behavior-videos", "name": "Behavior videos"}],
        "source_data": [f"{name}_raw"],
        "tags": [f"Column {column}", f"Volume {volume}"],
    }


def build_input(root, n=3, break_it=None):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        col, vol = 1 + i, 1 + (i % 2)
        name = f"409828_2018-11-0{i + 1}_14-02-59_filtered_2026-04-09_04-59-0{i}"
        d = root / name
        d.mkdir()
        subj = json.loads(json.dumps(SUBJECT))
        dd = data_description(name, col, vol)
        if break_it == "subject_mismatch" and i == 1:
            subj["subject_id"] = "999999"
        if break_it == "institution_mismatch" and i == 1:
            dd["institution"] = {"abbreviation": "XX", "name": "Somewhere else",
                                 "registry": "Research Organization Registry (ROR)",
                                 "registry_identifier": "00dcv1019"}
        if break_it == "no_investigators":
            dd["investigators"] = []
        (d / "subject.json").write_text(json.dumps(subj), encoding="utf-8")
        (d / "data_description.json").write_text(json.dumps(dd), encoding="utf-8")
        (d / f"{name}.nwb.zarr").mkdir()
    return root


def build_asset(root, n_sessions=3):
    root.mkdir(parents=True, exist_ok=True)
    (root / "stimulus_metrics_provenance.json").write_text(json.dumps({
        "generated_utc": "2026-08-16T04-16-35".replace("_", ":"), "seed": 0,
        "n_sessions": n_sessions, "n_planes": n_sessions * 6, "n_rois": 39407,
        "wall_seconds": 25832.0, "complete_asset": True, "session_filter": None,
        "git_sha": "abc1234", "config": {"ni_response_frames": 2, "dg_n_boot": 2500},
        "differs_from_reference_config": {
            "rf_center_scale_bug": {"used": False, "historical": True}},
        "environment": {"packages": {"numpy": "2.5.1"}},
    }).replace("2026-08-16T04-16-35", "2026-08-16T04:16:35"), encoding="utf-8")
    (root / "natural_movie_M409828.csv").write_text("roi\n1\n", encoding="utf-8")
    return root


def run(asset, inp, extra=(), env=None):
    return subprocess.run(
        [sys.executable, str(REPO / "code" / "metadata.py"),
         "--asset-dir", str(asset), "--input-asset", str(inp), *extra],
        capture_output=True, text=True,
        env=None if env is None else {**os.environ, **env})


def build_validation(checks: Path) -> None:
    """The two artifacts the validation notebook leaves behind."""
    checks.mkdir(parents=True, exist_ok=True)
    (checks / "validation.json").write_text(json.dumps(
        {"integrity": {"n_passed": 57, "n_checks": 57, "failed": []},
         "n_rois_recomputed": 0}), encoding="utf-8")
    (checks / "tests.json").write_text(json.dumps(
        {"n_pass": 15, "n_fail": 0, "checks_passed": 413}), encoding="utf-8")


print("[1] the happy path writes three valid sidecars")
tmp = Path(tempfile.mkdtemp(prefix="md_"))
inp = build_input(tmp / "in")
asset = build_asset(tmp / "409828_V1DD_stimulus_metrics_2026-08-16_04-16-35")
proc = run(asset, inp)
check("exits cleanly", proc.returncode == 0, (proc.stderr or "").strip()[-200:])
for name in ("subject.json", "data_description.json", "processing.json"):
    check(f"{name} written", (asset / name).is_file())

if (asset / "data_description.json").is_file():
    dd = json.loads((asset / "data_description.json").read_text(encoding="utf-8"))
    check("data_level is derived", dd["data_level"] == "derived", dd["data_level"])
    check("name is the asset directory name", dd["name"] == asset.name, dd["name"])
    check("every input session is listed in source_data", len(dd["source_data"]) == 3,
          str(dd["source_data"]))
    check("institution inherited", dd["institution"]["abbreviation"] == "AIBS")
    check("investigators inherited", len(dd["investigators"]) == 2)
    check("project inherited", dd["project_name"] == "V1 Deep Dive")
    check("licence inherited", dd["license"] == "CC-BY-4.0")
    abbrev = sorted(m["abbreviation"] for m in dd["modalities"])
    check("modalities are pophys + behavior, NOT behavior-videos",
          abbrev == ["behavior", "pophys"], str(abbrev))
    check("tags are the union across sessions, not one session's pair",
          set(dd["tags"]) == {"Column 1", "Column 2", "Column 3", "Volume 1", "Volume 2"},
          str(dd["tags"]))

if (asset / "subject.json").is_file():
    sj = json.loads((asset / "subject.json").read_text(encoding="utf-8"))
    check("subject inherited verbatim", sj["subject_id"] == "409828")
    check("genotype survives", "Slc17a7" in sj["subject_details"]["genotype"])
    check("breeding info survives (required when source is AI)",
          sj["subject_details"]["breeding_info"]["maternal_id"] == "372850")

if (asset / "processing.json").is_file():
    pj = json.loads((asset / "processing.json").read_text(encoding="utf-8"))
    dp = pj["data_processes"][0]
    check("one process without a validation run", len(pj["data_processes"]) == 1)
    check("named", dp["name"] == "Stimulus response metrics", dp["name"])
    check("output_path is relative", dp["output_path"] == ".", str(dp["output_path"]))
    check("every input session declared", len(dp["code"]["input_data"]) == 3)
    check("the config travels in parameters",
          dp["code"]["parameters"]["metric_config"]["ni_response_frames"] == 2)
    check("the departures from the historical pipeline are recorded",
          "rf_center_scale_bug" in dp["code"]["parameters"]["differs_from_reference_config"])
    check("seed recorded", dp["code"]["parameters"]["seed"] == 0)
    check("experimenters default to the upstream investigators",
          dp["experimenters"] == ["Saskia de Vries", "Clay Reid"], str(dp["experimenters"]))
    check("start precedes end", dp["start_date_time"] < dp["end_date_time"])

print("\n[2] a validation run adds a second process")
tmp2 = Path(tempfile.mkdtemp(prefix="md2_"))
inp2 = build_input(tmp2 / "in")
asset2 = build_asset(tmp2 / "results" / "409828_V1DD_stimulus_metrics_2026-08-16_04-16-35")
build_validation(tmp2 / "results" / "v1dd_stimulus_metrics_validation" / "checks")
proc = run(asset2, inp2, ["--results-dir", str(tmp2 / "results")])
check("exits cleanly", proc.returncode == 0, (proc.stderr or "").strip()[-200:])
if (asset2 / "processing.json").is_file():
    pj = json.loads((asset2 / "processing.json").read_text(encoding="utf-8"))
    check("two processes now", len(pj["data_processes"]) == 2)
    v = pj["data_processes"][1]
    check("the second is the validation", v["name"] == "Validation")
    check("it carries the integrity result",
          v["code"]["parameters"]["integrity_checks_passed"] == 57)
    check("and the unit-test result", v["code"]["parameters"]["unit_test_checks"] == 413)

print("\n[2b] the real layout: validation writes to scratch, not beside the asset")
# The first full run shipped a one-process processing.json because [2] above is not the
# shape the capsule produces. There, `--results-dir` is /results and the validation
# notebook writes to /scratch, so appending the run name to the results directory could
# never find anything. These checks fail against the code that shipped.
tmp3 = Path(tempfile.mkdtemp(prefix="md3_"))
inp3 = build_input(tmp3 / "in")
asset3 = build_asset(tmp3 / "results" / "409828_V1DD_stimulus_metrics_2026-08-16_19-40-03")
vdir3 = tmp3 / "scratch" / "v1dd_stimulus_metrics_validation"
build_validation(vdir3 / "checks")

proc = run(asset3, inp3, ["--results-dir", str(tmp3 / "results")])
check("exits cleanly", proc.returncode == 0, (proc.stderr or "").strip()[-200:])
if (asset3 / "processing.json").is_file():
    pj = json.loads((asset3 / "processing.json").read_text(encoding="utf-8"))
    check("--results-dir alone cannot see a scratch validation (the shipped behaviour)",
          len(pj["data_processes"]) == 1, str(len(pj["data_processes"])))

proc = run(asset3, inp3, ["--results-dir", str(tmp3 / "results"),
                          "--validation-dir", str(vdir3)],
           env={"SWDB_CODE_VERSION": "deadbee", "SWDB_CODE_URL": "https://example.test/x"})
check("exits cleanly", proc.returncode == 0, (proc.stderr or "").strip()[-200:])
if (asset3 / "processing.json").is_file():
    pj = json.loads((asset3 / "processing.json").read_text(encoding="utf-8"))
    check("--validation-dir finds it", len(pj["data_processes"]) == 2,
          str(len(pj["data_processes"])))
    check("integrity result carried across directories",
          pj["data_processes"][1]["code"]["parameters"]["integrity_checks_passed"] == 57)
    code = pj["data_processes"][0]["code"]
    # A reproducible run copies code/ without .git, so `git rev-parse` returns nothing and
    # the asset ships `commit_hash: null` unless the version is supplied explicitly.
    check("SWDB_CODE_VERSION stamps the version", code["version"] == "deadbee",
          str(code["version"]))
    check("SWDB_CODE_URL overrides the repo url", code["url"] == "https://example.test/x",
          str(code["url"]))

check("default repo url is the fork, not upstream",
      md.REPO_URL.endswith("SWDB_2026_Connectomics-functional-metrics"), md.REPO_URL)

print("\n[3] it refuses to invent, and refuses to pick a side")
for defect, expect in [("no_investigators", "missing"),
                       ("subject_mismatch", "disagrees"),
                       ("institution_mismatch", "disagrees")]:
    t = Path(tempfile.mkdtemp(prefix=f"md_{defect}_"))
    i = build_input(t / "in", break_it=defect)
    a = build_asset(t / "409828_V1DD_stimulus_metrics_2026-08-16_04-16-35")
    p = run(a, i)
    hit = expect in (p.stderr or "") + (p.stdout or "")
    check(f"{defect}: fails and says why", p.returncode != 0 and hit,
          (p.stderr or "").strip().splitlines()[-1][:120] if p.stderr else "no stderr")
    check(f"{defect}: writes nothing", not (a / "data_description.json").is_file())

print("\n[4] missing inputs are named, not guessed")
t = Path(tempfile.mkdtemp(prefix="md_empty_"))
(t / "in").mkdir(parents=True)
a = build_asset(t / "409828_V1DD_stimulus_metrics_2026-08-16_04-16-35")
p = run(a, t / "in")
check("an input asset with no sessions is an error",
      p.returncode != 0 and "no session directories" in (p.stderr or ""))

shutil.rmtree(tmp, ignore_errors=True)
summary()
