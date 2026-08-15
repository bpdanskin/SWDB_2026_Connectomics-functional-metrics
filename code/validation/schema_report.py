"""Describing an NWB session without computing anything from it.

This is the artifact that answered whether the per-trial stimulus table matched the schema
reconstructed from the NWB writer script -- it had never been read off a real file. It
**reports rather than asserts**: every section is guarded, and a missing table or column
becomes an error string rather than an exception, because a function that raises on the
first surprise tells you much less than one that describes the whole file.

It lives on the validation side. The pipeline reads specific fields it needs; this reads
everything to find out what is there.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

import v1dd_nwb as vn

__all__ = ["schema_report"]


def _describe_column(s: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {"dtype": str(s.dtype), "n_null": int(s.isna().sum())}
    nn = s.dropna()
    if not len(nn):
        return out
    if pd.api.types.is_numeric_dtype(nn):
        out["min"] = float(nn.min())
        out["max"] = float(nn.max())
        uniq = nn.unique()
        if len(uniq) <= 20:
            out["unique"] = sorted(float(v) for v in uniq)
        else:
            out["n_unique"] = int(len(uniq))
    else:
        uniq = nn.unique()
        out["n_unique"] = int(len(uniq))
        if len(uniq) <= 20:
            out["unique"] = sorted(str(v) for v in uniq)
    return out


def schema_report(nwb, planes: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Describe a session without computing any metric. The M1 artifact.

    Every section is guarded: a missing table or column is recorded as an error string
    rather than raised, because the point is to learn what the file contains.
    """
    report: Dict[str, Any] = {}

    # --- epochs
    try:
        ep = vn.epoch_table(nwb)
        report["epochs"] = {
            "n_rows": int(len(ep)),
            "rows": ep[["stim_name", "start_time", "stop_time", "duration"]]
            .round(3).to_dict("records"),
            "stim_names": sorted(ep["stim_name"].unique().tolist()),
            "n_spontaneous_blocks": int((ep["stim_name"] == "spontaneous").sum()),
        }
    except Exception as exc:
        report["epochs"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- stimulus table: the unverified one
    try:
        st = vn.load_stimulus_table(nwb)
        cols = list(st.columns)
        report["stimulus_table"] = {
            "n_rows": int(len(st)),
            "columns": cols,
            "expected_columns": vn.STIMULUS_TABLE_COLUMNS,
            "missing_vs_expected": [c for c in vn.STIMULUS_TABLE_COLUMNS if c not in cols],
            "extra_vs_expected": [
                c for c in cols if c not in vn.STIMULUS_TABLE_COLUMNS and c != "id"
            ],
            "column_detail": {c: _describe_column(st[c]) for c in cols},
            "head": st.head(4).round(4).to_dict("records"),
            "sweeps_per_stimulus": {
                str(k): int(v) for k, v in st["stim_name"].value_counts().items()
            },
        }
        report["per_stimulus"] = _per_stimulus_report(st)
    except Exception as exc:
        report["stimulus_table"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- planes
    plane_keys = vn.list_planes(nwb)
    report["planes"] = {"keys": plane_keys, "detail": {}}
    wanted = plane_keys if planes is None else [f"plane-{p}" for p in planes]
    for key in wanted:
        try:
            module = nwb.processing[key]
            interfaces = sorted(module.data_interfaces)
            series = module["dff" if "dff" in interfaces else interfaces[0]]
            ts = np.asarray(series.timestamps[:], dtype=np.float64)
            rois = series.rois.to_dataframe()
            conf = rois.get("pika_roi_confidence")
            soma = rois.get("is_soma")
            entry = {
                "interfaces": interfaces,
                "shape": [int(x) for x in np.shape(series.data)],
                "n_frames": int(len(ts)),
                "dt": float(np.median(np.diff(ts))),
                "fs": float(1.0 / np.median(np.diff(ts))),
                "duration_s": float(ts[-1] - ts[0]),
                "roi_columns": list(rois.columns),
                "n_rois": int(len(rois)),
            }
            if conf is not None and soma is not None:
                entry["is_soma_matches_conf_gt_0.5"] = bool(
                    np.array_equal(soma.to_numpy().astype(bool), conf.to_numpy() > 0.5)
                )
                entry["n_valid"] = int((conf.to_numpy() > 0.5).sum())
            report["planes"]["detail"][key] = entry
        except Exception as exc:
            report["planes"]["detail"][key] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- running speed
    try:
        speed, ts = vn.load_running_speed(nwb)
        report["running_speed"] = {
            "n_samples": int(len(speed)),
            "fs": float(1.0 / np.median(np.diff(ts))),
            "min": float(np.nanmin(speed)),
            "max": float(np.nanmax(speed)),
            "frac_above_1_cm_s": float(np.nanmean(speed > 1.0)),
        }
    except Exception as exc:
        report["running_speed"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- stimulus templates, incl. the RF viability question
    try:
        report["stimulus_images"] = {
            k: int(len(getattr(nwb.stimulus[k], "images", {}) or {}))
            for k in nwb.stimulus
        }
    except Exception as exc:
        report["stimulus_images"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        lsn = vn.load_lsn_template(nwb)
        report["lsn_template"] = {
            "n_frames": len(lsn["frames"]),
            "native_shape": list(lsn["native_shape"]),
            "reduced": lsn["reduced"],
            "blocks_uniform": lsn["blocks_uniform"],
            "block_factor": list(lsn.get("block_factor", ())) or None,
            "final_shape": [int(x) for x in lsn["images"].shape[1:]],
            "azimuth_range": [float(lsn["azimuths"][0]), float(lsn["azimuths"][-1])],
            "altitude_range": [float(lsn["altitudes"][0]), float(lsn["altitudes"][-1])],
            "pixel_values": lsn.get("pixel_values"),
            "pixel_on": lsn.get("pixel_on"),
            "pixel_off": lsn.get("pixel_off"),
            "pixel_gray": lsn.get("pixel_gray"),
            "rf_viable": bool(lsn["reduced"] or lsn["native_shape"] == (8, 14)),
        }
        if "error" in lsn:
            report["lsn_template"]["error"] = lsn["error"]
    except Exception as exc:
        report["lsn_template"] = {"error": f"{type(exc).__name__}: {exc}"}

    return report


def _per_stimulus_report(st: pd.DataFrame) -> Dict[str, Any]:
    """Condition counts and inferred n_trials for each family, to check against the
    published denominators (DG 8, NI 8, NI12 40, NM 9)."""
    out: Dict[str, Any] = {}

    for name in ("drifting_gratings_full", "drifting_gratings_windowed"):
        trials, is_blank = vn.stimulus_trials(st, name, vn.DG_PARAM_COLUMNS)
        if not len(trials):
            continue
        grat = trials.loc[~is_blank]
        dirs = np.sort(grat["direction"].dropna().unique())
        sfs = np.sort(grat["spatial_frequency"].dropna().unique())
        counts = grat.groupby(["direction", "spatial_frequency"]).size()
        out[name] = {
            "n_sweeps": int(len(trials)),
            "n_blank": int(is_blank.sum()),
            "directions": [float(d) for d in dirs],
            "n_directions": int(len(dirs)),
            "spatial_frequencies": [float(s) for s in sfs],
            "temporal_frequencies": sorted(
                float(v) for v in grat["temporal_frequency"].dropna().unique()
            ),
            "n_conditions": int(len(counts)),
            "trials_per_condition": {
                "min": int(counts.min()), "max": int(counts.max()),
                "counts": {str(int(k)): int(v) for k, v in counts.value_counts().items()},
            },
            "n_trials_inferred": int(counts.max()),
            "sweep_duration_s": _duration_summary(trials),
        }

    for name in ("natural_images", "natural_images_12"):
        trials, _ = vn.stimulus_trials(st, name)
        if not len(trials):
            continue
        ids = np.sort(trials["image_index"].dropna().unique())
        counts = trials.groupby("image_index").size()
        out[name] = {
            "n_sweeps": int(len(trials)),
            "n_images": int(len(ids)),
            "image_index_range": [float(ids.min()), float(ids.max())],
            "image_indices": [int(v) for v in ids] if len(ids) <= 20 else None,
            "n_trials_inferred": int(counts.max()),
            "trials_per_image": {"min": int(counts.min()), "max": int(counts.max())},
            "sweep_duration_s": _duration_summary(trials),
        }

    trials, _ = vn.stimulus_trials(st, "natural_movie")
    if len(trials):
        frames = trials["frame"].dropna().to_numpy().astype(int)
        counts = np.bincount(frames)
        out["natural_movie"] = {
            "n_sweeps": int(len(trials)),
            "frame_range": [int(frames.min()), int(frames.max())],
            "frames_contiguous_from_zero": bool(
                np.array_equal(np.unique(frames), np.arange(frames.max() + 1))
            ),
            "n_repeats_inferred": int(counts.max()),
            "repeats_per_frame": {"min": int(counts.min()), "max": int(counts.max())},
            "sweep_duration_s": _duration_summary(trials),
        }

    trials, _ = vn.stimulus_trials(st, "locally_sparse_noise")
    if len(trials):
        frames = trials["frame"].dropna().to_numpy().astype(int)
        out["locally_sparse_noise"] = {
            "n_sweeps": int(len(trials)),
            "frame_range": [int(frames.min()), int(frames.max())],
            "n_unique_frames": int(len(np.unique(frames))),
            "sweep_duration_s": _duration_summary(trials),
        }

    trials, _ = vn.stimulus_trials(st, "spontaneous")
    out["spontaneous"] = {"n_sweeps": int(len(trials))}
    return out


def _duration_summary(trials: pd.DataFrame) -> Dict[str, float]:
    """Sweep duration and onset-to-onset period.

    The original took the stimulus duration from an NWB attribute (2.0 s for gratings)
    rather than from the trial rows. If `stop - start` disagrees, the response window is
    a decision, not a lookup — hence reporting both here.
    """
    d = (trials["stop_time"] - trials["start_time"]).to_numpy()
    onset = np.diff(trials["start_time"].to_numpy())
    out = {
        "median_stop_minus_start": float(np.median(d)),
        "min_stop_minus_start": float(np.min(d)),
        "max_stop_minus_start": float(np.max(d)),
    }
    if onset.size:
        out["median_onset_to_onset"] = float(np.median(onset))
    return out
