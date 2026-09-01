"""Read-only survey of every session in the functional asset.

This exists to answer three questions before the pipeline is widened from two sessions to
all of them, each of which is currently an assumption rather than a fact:

1. **Does every session carry all six stimuli?** `stimulus_trials` returns an *empty
   frame* for a missing stimulus rather than raising, so a session that never ran, say,
   locally sparse noise would flow silently into the metric functions and produce
   confident nonsense. The two validated sessions are both 2p and have all seven blocks;
   the 3p sessions (letter volumes) have never been looked at.

2. **Where does the mouse id actually live?** The metric tables hard-code `"M409828"`.
   `nwb.subject.subject_id` is the obvious source but nothing has ever read it, so this
   records what is there alongside the id parsed from the session name and whether the two
   agree.

3. **Do the imaging planes carry depth?** The ROI tables carry `column, volume, plane,
   roi` and no physical depth, so tuning-versus-depth is currently unaskable. If the
   `ImagingPlane` holds it, it is worth adding to the output schema.

Nothing here loads a trace array. `load_plane` pulls every sample for every ROI — roughly
400 MB per session — which is fine for two sessions and not for thirty-five. Shapes and
timestamps are metadata and cost nothing, so that is all this reads.

Every probe is wrapped: a session that cannot be opened, or a field that does not exist,
becomes a recorded string rather than an exception. A survey that dies on the first
surprise cannot survey anything.
"""

import json
import os
import re
import sys
import time
import traceback
from os.path import join as pjoin
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

for _c in [pjoin("..", "utils"), pjoin("code", "utils"), "utils",
           pjoin(os.path.dirname(os.path.abspath(__file__)), "..", "utils")]:
    if os.path.isdir(_c) and os.path.abspath(_c) not in sys.path:
        sys.path.append(os.path.abspath(_c))

import v1dd_nwb as vn  # noqa: E402

__all__ = ["FAMILIES", "probe_session", "run_preflight", "summarise"]

#: The six metric families, by their `stim_name` in the stimulus table.
FAMILIES = ["drifting_gratings_full", "drifting_gratings_windowed", "natural_images",
            "natural_images_12", "natural_movie", "locally_sparse_noise"]

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _probe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run `fn`, returning its value or a short error string. Never raises.

    The point of a survey is to come back with a description of what is there, including
    the parts that are broken. An exception anywhere would replace all of that with one
    traceback.
    """
    try:
        return fn()
    except Exception as exc:                                    # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        return msg[:200] if default is None else default


def _plain(value: Any) -> Any:
    """Scalars JSON can hold; anything else becomes its repr, truncated."""
    if value is None or isinstance(value, (bool, str, int, float)):
        return None if isinstance(value, float) and not np.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in list(value)[:16]]
    return str(value)[:200]


def _subject(nwb) -> Dict[str, Any]:
    """Everything the NWB says about the animal, plus the AIND-metadata-relevant fields.

    We are opening every file anyway, and the sidecar step later needs species, sex,
    genotype and date of birth. Recording them now costs nothing and saves a second pass.
    """
    s = getattr(nwb, "subject", None)
    if s is None:
        return {"present": False}
    out: Dict[str, Any] = {"present": True}
    for f in ("subject_id", "species", "sex", "age", "genotype", "strain",
              "date_of_birth", "description", "weight"):
        out[f] = _plain(_probe(lambda f=f: getattr(s, f, None)))
    return out


def _file_metadata(nwb) -> Dict[str, Any]:
    """Session-level fields, again with the eventual data_description sidecar in mind."""
    out = {}
    for f in ("session_id", "session_description", "experiment_description", "identifier",
              "lab", "institution", "session_start_time", "experimenter"):
        out[f] = _plain(_probe(lambda f=f: getattr(nwb, f, None)))
    out["intervals"] = _probe(lambda: sorted(nwb.intervals.keys()), [])
    out["acquisition"] = _probe(lambda: sorted(nwb.acquisition.keys()), [])
    return out


def _imaging_plane(series) -> Dict[str, Any]:
    """The ImagingPlane behind a RoiResponseSeries — the only plausible home for depth.

    `origin_coords` is the structured place for it; `location` and `description` are free
    text and may carry it informally. All three are recorded raw, plus any numbers found
    in the text, because guessing a parse before seeing the strings is how you write a
    regex that silently matches the wrong number.
    """
    ip = _probe(lambda: series.rois.table.imaging_plane)
    if isinstance(ip, str) or ip is None:
        return {"error": ip or "no imaging_plane"}
    out: Dict[str, Any] = {}
    for f in ("name", "description", "location", "imaging_rate", "excitation_lambda",
              "indicator", "origin_coords", "origin_coords_unit", "grid_spacing",
              "grid_spacing_unit", "reference_frame"):
        out[f] = _plain(_probe(lambda f=f: getattr(ip, f, None)))
    out["device"] = _plain(_probe(lambda: ip.device.name))
    text = " ".join(str(out.get(k) or "") for k in ("location", "description", "name"))
    out["numbers_in_text"] = [float(x) for x in _NUMBER.findall(text)][:8]
    return out


def _planes(nwb) -> Dict[str, Any]:
    """Per-plane shape, sampling and available trace types — without reading a trace.

    `series.data` is the lazy backing array for both Zarr and HDF5, so `.shape` is
    metadata. Slicing it is what would cost hundreds of megabytes.
    """
    keys = _probe(lambda: vn.list_planes(nwb), [])
    if isinstance(keys, str):
        return {"error": keys}
    detail: Dict[str, Any] = {}
    for k in keys:
        d: Dict[str, Any] = {}
        module = _probe(lambda k=k: nwb.processing[k])
        if isinstance(module, str):
            detail[k] = {"error": module}
            continue
        available = _probe(lambda: sorted(module.data_interfaces), [])
        d["interfaces"] = available
        d["has_events"] = "events" in available
        d["has_dff"] = "dff" in available
        pick = "events" if "events" in available else ("dff" if "dff" in available else None)
        if pick is None:
            d["error"] = "neither events nor dff"
            detail[k] = d
            continue
        series = _probe(lambda: module[pick])
        d["shape"] = _plain(_probe(lambda: tuple(series.data.shape)))
        ts = _probe(lambda: np.asarray(series.timestamps[:], dtype=np.float64))
        if isinstance(ts, str):
            d["timestamps_error"] = ts
        else:
            d["n_frames"] = int(ts.size)
            d["dt"] = float(np.median(np.diff(ts))) if ts.size > 1 else None
            d["duration_s"] = float(ts[-1] - ts[0]) if ts.size > 1 else None
        d["imaging_plane"] = _imaging_plane(series)
        detail[k] = d
    return {"n_planes": len(keys), "detail": detail}


def _roi_identity(nwb) -> Dict[str, Any]:
    """`column`/`volume` off the first plane's ROI table, as `peek_session` does."""
    keys = _probe(lambda: vn.list_planes(nwb), [])
    if isinstance(keys, str) or not keys:
        return {"error": keys or "no planes"}
    def _read():
        rois = nwb.processing[keys[0]]["dff"].rois.to_dataframe()
        return {"column": int(pd.to_numeric(rois["column"]).iloc[0]),
                "volume": vn._as_volume_str(rois["volume"].iloc[0]),
                "roi_columns": sorted(c for c in rois.columns if c != "pixel_mask")}
    got = _probe(_read)
    return got if isinstance(got, dict) else {"error": got}


def _stimulus_coverage(nwb) -> Dict[str, Any]:
    """Which of the six families this session actually ran, and their parameter ranges.

    The counts matter as much as the presence: a family with a handful of sweeps is not
    the same as an absent one, and only the first would survive `stimulus_trials` looking
    like real data.
    """
    out: Dict[str, Any] = {}

    epochs = _probe(lambda: vn.epoch_table(nwb))
    if isinstance(epochs, str):
        out["epochs"] = {"error": epochs}
    else:
        out["epochs"] = {
            "n_rows": int(len(epochs)),
            "stim_names": sorted(epochs["stim_name"].astype(str).unique().tolist()),
            "n_spontaneous_blocks": int((epochs["stim_name"].astype(str)
                                         == "spontaneous").sum()),
        }
    out["spontaneous_block"] = _plain(_probe(lambda: list(vn.spontaneous_block(nwb))))

    st = _probe(lambda: vn.load_stimulus_table(nwb))
    if isinstance(st, str):
        return {**out, "stimulus_table": {"error": st}}

    counts = st["stim_name"].astype(str).value_counts().to_dict()
    out["stimulus_table"] = {"n_rows": int(len(st)), "columns": sorted(st.columns.tolist()),
                             "sweeps_per_stimulus": {k: int(v) for k, v in counts.items()}}

    per: Dict[str, Any] = {}
    for fam in FAMILIES:
        rows = st.loc[st["stim_name"] == fam]
        entry: Dict[str, Any] = {"present": bool(len(rows)), "n_sweeps": int(len(rows))}
        if len(rows):
            if fam.startswith("drifting_gratings"):
                _, blank = _probe(lambda rows=rows: vn.stimulus_trials(
                    st, fam, vn.DG_PARAM_COLUMNS), (None, np.zeros(0, bool)))
                entry["n_blank_sweeps"] = int(np.sum(blank)) if blank is not None else None
                for c in ("direction", "spatial_frequency", "temporal_frequency"):
                    vals = _probe(lambda c=c: sorted(
                        pd.to_numeric(rows[c], errors="coerce").dropna().unique().tolist()))
                    entry[c] = _plain(vals)
                    if isinstance(vals, list):
                        entry[f"n_{c}"] = len(vals)
                if fam == "drifting_gratings_windowed":
                    for c in ("center_azimuth", "center_elevation"):
                        vals = _probe(lambda c=c: sorted(
                            pd.to_numeric(rows[c], errors="coerce").dropna().unique().tolist()))
                        entry[c] = _plain(vals)
            elif fam.startswith("natural_images"):
                vals = _probe(lambda: pd.to_numeric(rows["image_index"], errors="coerce")
                              .dropna().unique())
                entry["n_images"] = int(len(vals)) if not isinstance(vals, str) else None
            elif fam == "natural_movie":
                fr = _probe(lambda: pd.to_numeric(rows["frame"], errors="coerce").dropna())
                if not isinstance(fr, str) and len(fr):
                    entry["frame_min"] = float(fr.min())
                    entry["frame_max"] = float(fr.max())
                    entry["n_unique_frames"] = int(fr.nunique())
        per[fam] = entry
    out["per_family"] = per
    return out


def _lsn_template(nwb) -> Dict[str, Any]:
    """The locally-sparse-noise template, including the 2x2 block-uniformity assert.

    This is the gate on receptive fields: the NWB embeds a 16x28 grid where the metric
    code expects 8x14, and they are only the same stimulus if every 2x2 block is uniform.
    """
    got = _probe(lambda: vn.load_lsn_template(nwb))
    if isinstance(got, str):
        return {"error": got}
    return {"error": got.get("error"), "native_shape": _plain(got.get("native_shape")),
            "reduced_shape": _plain(_probe(lambda: tuple(got["images"].shape))),
            "pixel_on": _plain(got.get("pixel_on")), "pixel_off": _plain(got.get("pixel_off")),
            "pixel_gray": _plain(got.get("pixel_gray")),
            "pixel_values": _plain(got.get("pixel_values"))}


def probe_session(path) -> Dict[str, Any]:
    """Everything the survey wants from one session. Never raises."""
    started = time.time()
    rec: Dict[str, Any] = {"path": str(path), "name": os.path.basename(os.path.dirname(str(path))),
                           "format": _probe(lambda: vn.nwb_format(path)), "error": None}
    # The mouse as the directory name claims it, independent of what the file says.
    token = rec["name"].split("_", 1)[0]
    rec["mouse_from_name"] = token if token.isdigit() else None

    nwb = io = None
    try:
        nwb, io = vn.open_session(path)
        rec["file"] = _file_metadata(nwb)
        rec["subject"] = _subject(nwb)
        rec["roi_identity"] = _roi_identity(nwb)
        rec["planes"] = _planes(nwb)
        rec["stimulus"] = _stimulus_coverage(nwb)
        rec["lsn_template"] = _lsn_template(nwb)
        rec["running_speed"] = _probe(lambda: {
            "n_samples": int(len(vn.load_running_speed(nwb)[0])),
            "min": float(np.nanmin(vn.load_running_speed(nwb)[0])),
            "max": float(np.nanmax(vn.load_running_speed(nwb)[0])),
        })
    except Exception:                                            # noqa: BLE001
        rec["error"] = traceback.format_exc(limit=3)[:800]
    finally:
        if io is not None:
            try:
                io.close()
            except Exception:                                    # noqa: BLE001, S110
                pass
    rec["probe_seconds"] = round(time.time() - started, 2)
    return rec


def _row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """One flat row per session — the artifact that is actually readable at 35 rows."""
    subj = rec.get("subject") or {}
    ident = rec.get("roi_identity") or {}
    planes = rec.get("planes") or {}
    stim = rec.get("stimulus") or {}
    fam = stim.get("per_family") or {}
    lsn = rec.get("lsn_template") or {}
    detail = (planes.get("detail") or {})

    # `shape` is (n_frames, n_rois) in this asset, but `load_plane` tolerates the
    # transpose, so infer rather than assume: n_rois is whichever axis is not n_frames.
    def _n_rois(d):
        shape, n_frames = d.get("shape"), d.get("n_frames")
        if not isinstance(shape, list) or len(shape) != 2:
            return None
        if n_frames in shape:
            return shape[1] if shape[0] == n_frames else shape[0]
        return min(shape)

    counted = [_n_rois(d) for d in detail.values()]
    shapes = [n for n in counted if isinstance(n, int)]
    dts = [d.get("dt") for d in detail.values() if isinstance(d.get("dt"), float)]
    ip = next((d.get("imaging_plane") for d in detail.values()
               if isinstance(d.get("imaging_plane"), dict)), {})

    row = {
        "name": rec.get("name"), "format": rec.get("format"),
        "error": (rec.get("error") or "")[:120] or None,
        "session_id": (rec.get("file") or {}).get("session_id"),
        "column": ident.get("column"), "volume": ident.get("volume"),
        "mouse_subject": subj.get("subject_id"), "mouse_name": rec.get("mouse_from_name"),
        "n_planes": planes.get("n_planes"),
        "n_rois_total": int(sum(shapes)) if shapes else None,
        "dt_median": round(float(np.median(dts)), 6) if dts else None,
        "has_events": all(d.get("has_events") for d in detail.values()) if detail else None,
        "has_dff": all(d.get("has_dff") for d in detail.values()) if detail else None,
        "n_spont": (stim.get("epochs") or {}).get("n_spontaneous_blocks"),
        "ip_location": ip.get("location"), "ip_origin": _plain(ip.get("origin_coords")),
        "lsn_ok": (lsn.get("error") is None) if lsn else None,
        "lsn_reduced": _plain(lsn.get("reduced_shape")),
    }
    for f in FAMILIES:
        row[f"n_{_short(f)}"] = (fam.get(f) or {}).get("n_sweeps", 0)
    row["n_dirs"] = (fam.get("drifting_gratings_windowed") or {}).get("n_direction")
    row["n_sfs"] = (fam.get("drifting_gratings_windowed") or {}).get("n_spatial_frequency")
    return row


def _row_get(d, *keys):
    for k in keys:
        d = (d or {}).get(k)
    return d


def _short(family: str) -> str:
    return {"drifting_gratings_full": "dgf", "drifting_gratings_windowed": "dgw",
            "natural_images": "ni", "natural_images_12": "ni12",
            "natural_movie": "nm", "locally_sparse_noise": "lsn"}[family]


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The aggregate verdict: what varies across sessions, and what is missing."""
    rows = pd.DataFrame([_row(r) for r in records])
    ok = rows[rows["error"].isna()]
    fam_cols = [f"n_{_short(f)}" for f in FAMILIES]

    verdict: Dict[str, Any] = {
        "n_sessions": int(len(rows)),
        "n_readable": int(len(ok)),
        "formats": rows["format"].value_counts().to_dict(),
        "columns": ok["column"].dropna().astype(int).value_counts().to_dict(),
        "volumes": ok["volume"].dropna().astype(str).value_counts().to_dict(),
        "mouse_from_subject": ok["mouse_subject"].dropna().astype(str)
                                .value_counts().to_dict(),
        "mouse_from_name": ok["mouse_name"].dropna().astype(str).value_counts().to_dict(),
        "mouse_sources_agree": bool(
            (ok["mouse_subject"].astype(str).str.lstrip("M")
             == ok["mouse_name"].astype(str)).all()) if len(ok) else None,
        "n_planes": ok["n_planes"].dropna().astype(int).value_counts().to_dict(),
        "dt_range": [float(ok["dt_median"].min()), float(ok["dt_median"].max())]
                    if ok["dt_median"].notna().any() else None,
        "sessions_missing_a_family": {
            f: sorted(ok.loc[ok[f"n_{_short(f)}"].fillna(0) == 0, "name"].tolist())
            for f in FAMILIES},
        "sessions_with_all_six": int((ok[fam_cols].fillna(0) > 0).all(axis=1).sum()),
        # .eq(True), not .fillna(False): lsn_ok is object-dtype with None for sessions
        # that never got far enough to try, and fillna would downcast with a warning.
        "lsn_template_ok": int(ok["lsn_ok"].eq(True).sum()),
        "sessions_without_spontaneous": sorted(
            ok.loc[ok["n_spont"].fillna(0) == 0, "name"].tolist()),
        "imaging_plane_locations": ok["ip_location"].dropna().astype(str)
                                     .value_counts().to_dict(),
        "n_direction_values": ok["n_dirs"].dropna().astype(int).value_counts().to_dict(),
        "unreadable": sorted(rows.loc[rows["error"].notna(), "name"].tolist()),
    }
    return verdict


def run_preflight(functional_dir: str, save_dir: str, limit: Optional[int] = None,
                  verbose: bool = True) -> Dict[str, Any]:
    """Probe every session, write the summary CSV and the JSON report, return the verdict."""
    from checkpoints import git_sha, jsonable

    paths = vn.find_sessions(functional_dir)
    if limit:
        paths = paths[:limit]
    os.makedirs(save_dir, exist_ok=True)

    records = []
    started = time.time()
    for i, p in enumerate(paths, 1):
        rec = probe_session(p)
        records.append(rec)
        if verbose:
            fam = rec.get("stimulus", {}).get("per_family", {})
            have = "".join("." if (fam.get(f) or {}).get("present") else "x"
                           for f in FAMILIES)
            print(f"  [{i:>2}/{len(paths)}] {rec['name'][:44]:<44} {rec['format']:<5} "
                  f"{have}  {rec['probe_seconds']:>5.1f}s"
                  + ("  ERROR" if rec.get("error") else ""))

    rows = pd.DataFrame([_row(r) for r in records])
    csv_path = pjoin(save_dir, "preflight_summary.csv")
    rows.to_csv(csv_path, index=False)

    verdict = summarise(records)
    verdict["_provenance"] = {"git_sha": git_sha(), "functional_dir": str(functional_dir),
                              "wall_seconds": round(time.time() - started, 1),
                              "numpy": np.__version__, "pandas": pd.__version__}
    # Full detail for two representative sessions only: the whole set would be ~600 KB,
    # and the summary CSV plus the verdict is what actually gets read.
    by_col = {}
    for r in records:
        key = str((r.get("roi_identity") or {}).get("column"))
        by_col.setdefault(key, r)
    verdict["detail_samples"] = {k: v for k, v in list(by_col.items())[:3]}

    json_path = pjoin(save_dir, "preflight_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(verdict), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")

    if verbose:
        print(f"\nwrote {csv_path}  ({os.path.getsize(csv_path) / 1024:.1f} KB)")
        print(f"wrote {json_path}  ({os.path.getsize(json_path) / 1024:.1f} KB)")
    return verdict


if __name__ == "__main__":
    from paths import resolve_data_root, resolve_dataset_dir

    asset = sys.argv[1] if len(sys.argv) > 1 else "409828_V1DD_Filtered"
    out = sys.argv[2] if len(sys.argv) > 2 else "/scratch/v1dd_stimulus_metrics_validation"
    root = resolve_data_root(asset)
    run_preflight(resolve_dataset_dir(asset, root=root), out)
