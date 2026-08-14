"""Reading V1DD two-photon sessions out of NWB-Zarr.

The `allen_v1dd` analysis code reached into a private Isilon HDF5 tree through
`OPhysClient`/`OPhysSession`, indexing literal paths like
`processing/l0_events_plane3/DfOverF/l0_events`. That tree is gone; the same sessions are
now published as NWB-Zarr. This module is the replacement seam — the only file in the
port that imports `hdmf_zarr` — so that when the NWB layout changes again, one file moves.

Three differences from the old client are worth knowing before you use this:

* **Traces are transposed.** NWB stores `(n_frames, n_rois)`; `get_traces` returned
  `(n_rois, n_time)`. This module keeps NWB's orientation, because that is what
  `trial_responses.prefix_sums` wants. Metric code transposes once, deliberately.
* **One stimulus table, not seven.** The old client returned a per-stimulus DataFrame
  with stimulus-specific columns. NWB concatenates every stimulus into
  `intervals['stimulus_table']` with a union of columns and NaN where a parameter does
  not apply. `stimulus_trials()` slices it back apart — and note that "blank sweep"
  can no longer be detected as "any NaN in the row", since every drifting-gratings row
  is NaN in the image and frame columns. Pass the parameter columns explicitly.
* **Running speed is already differentiated.** The old client read cumulative distance
  and took a central difference; NWB ships cm/s directly.

`schema_report()` deliberately reports rather than asserts. The stimulus-table schema was
reconstructed from the NWB *writer* script and has never been read back off a real file,
so the first job is to find out what is actually there — a function that raises on the
first surprise tells you much less than one that describes the whole file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PlaneData",
    "open_session",
    "list_planes",
    "load_plane",
    "load_stimulus_table",
    "stimulus_trials",
    "epoch_table",
    "spontaneous_block",
    "load_running_speed",
    "load_lsn_template",
    "schema_report",
    "STIMULUS_TABLE_COLUMNS",
    "DG_PARAM_COLUMNS",
]

#: Columns the NWB writer is expected to emit. Verified against a real file in M1;
#: `schema_report` reports what is actually present rather than assuming this.
STIMULUS_TABLE_COLUMNS = [
    "stim_name", "start_time", "stop_time", "temporal_frequency", "spatial_frequency",
    "center_azimuth", "center_elevation", "direction", "frame", "image_order",
    "image_index", "stimulus_condition_id",
]

#: The columns that define a drifting-gratings condition. Blank (gray) sweeps are rows
#: that are NaN in these — NOT rows that are NaN anywhere, which in the concatenated
#: table is every drifting-gratings row.
DG_PARAM_COLUMNS = ["temporal_frequency", "spatial_frequency", "direction"]


@dataclass
class PlaneData:
    """One imaging plane's traces and ROI identity."""

    column: int
    volume: str                       # str: volumes run 1..9 and a..f across the project
    plane: int
    roi: np.ndarray                   # (n_rois,) ROI ids from the file, not 0..n-1
    is_valid: np.ndarray              # (n_rois,) bool, pika_roi_confidence > 0.5
    timestamps: np.ndarray            # (n_frames,) seconds
    traces: Dict[str, np.ndarray]     # trace_type -> (n_frames, n_rois)
    roi_table: pd.DataFrame
    dt: float                         # median inter-frame interval, seconds

    @property
    def n_rois(self) -> int:
        return len(self.roi)

    @property
    def fs(self) -> float:
        return 1.0 / self.dt


def open_session(path):
    """Open an NWB-Zarr session read-only. Caller owns closing the returned io object.

    Returns `(nwbfile, io)`. Use as a context manager via `io` when looping over
    sessions; the notebook keeps one open for the interactive path.
    """
    from hdmf_zarr import NWBZarrIO

    io = NWBZarrIO(str(path), mode="r")
    return io.read(), io


def list_planes(nwb) -> List[str]:
    """Imaging-plane processing module names, ordered by plane number."""
    keys = [k for k in nwb.processing if k.startswith("plane-")]
    return sorted(keys, key=lambda k: int(k.split("-", 1)[1]))


def _as_volume_str(value) -> str:
    """Volumes are 1..9 and a..f; normalise to str once, here, and never re-parse."""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def load_plane(
    nwb,
    plane,
    trace_types: Sequence[str] = ("dff", "events"),
    valid_only: bool = False,
) -> PlaneData:
    """Load one plane's traces, timestamps and ROI table.

    `trace_types` are the data interfaces to pull — the shipped metrics use `events` for
    everything except receptive fields, which use `dff`. Loading both once is cheaper
    than reopening the plane.
    """
    key = f"plane-{int(plane)}" if not isinstance(plane, str) else plane
    module = nwb.processing[key]

    missing = [t for t in trace_types if t not in module.data_interfaces]
    if missing:
        raise KeyError(
            f"{key} has no {missing}; available: {sorted(module.data_interfaces)}"
        )

    first = module[trace_types[0]]
    timestamps = np.asarray(first.timestamps[:], dtype=np.float64)

    roi_table = first.rois.to_dataframe().reset_index(drop=False)
    roi_table = roi_table.rename(columns={"id": "nwb_roi_row_id"})
    roi_table = roi_table.drop(columns=[c for c in ("pixel_mask",) if c in roi_table])

    traces: Dict[str, np.ndarray] = {}
    for t in trace_types:
        series = module[t]
        data = np.asarray(series.data[:])
        if data.shape[0] != timestamps.shape[0]:
            if data.ndim == 2 and data.shape[1] == timestamps.shape[0]:
                data = data.T
            else:
                raise ValueError(
                    f"{key}/{t}: data {data.shape} incompatible with timestamps "
                    f"{timestamps.shape}"
                )
        if data.shape[1] != len(roi_table):
            raise ValueError(
                f"{key}/{t}: {data.shape[1]} trace columns vs {len(roi_table)} roi rows"
            )
        traces[t] = data

    if "pika_roi_confidence" in roi_table:
        is_valid = roi_table["pika_roi_confidence"].to_numpy() > 0.5
    elif "is_soma" in roi_table:
        is_valid = roi_table["is_soma"].to_numpy().astype(bool)
    else:
        is_valid = np.ones(len(roi_table), dtype=bool)

    if valid_only:
        traces = {t: d[:, is_valid] for t, d in traces.items()}
        roi_table = roi_table[is_valid].reset_index(drop=True)
        is_valid = np.ones(len(roi_table), dtype=bool)

    return PlaneData(
        column=int(pd.to_numeric(roi_table["column"]).iloc[0]),
        volume=_as_volume_str(roi_table["volume"].iloc[0]),
        plane=int(pd.to_numeric(roi_table["plane"]).iloc[0]),
        roi=pd.to_numeric(roi_table["roi"]).to_numpy(dtype=int),
        is_valid=is_valid,
        timestamps=timestamps,
        traces=traces,
        roi_table=roi_table,
        dt=float(np.median(np.diff(timestamps))),
    )


def load_stimulus_table(nwb) -> pd.DataFrame:
    """The flat per-trial stimulus table, chronological, index reset."""
    df = nwb.intervals["stimulus_table"].to_dataframe().reset_index(drop=False)
    return df.sort_values("start_time").reset_index(drop=True)


def epoch_table(nwb) -> pd.DataFrame:
    """Stimulus-block boundaries, sorted by start time."""
    df = nwb.intervals["epochs"].to_dataframe().reset_index(drop=False)
    if "duration" not in df:
        df["duration"] = df["stop_time"] - df["start_time"]
    return df.sort_values("start_time").reset_index(drop=True)


def stimulus_trials(
    stim_table: pd.DataFrame,
    stim_name: str,
    param_columns: Sequence[str] = (),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Rows for one stimulus, chronological, index reset, plus a blank-sweep mask.

    `param_columns` are the columns that define a condition for this stimulus. A sweep
    is blank when it is NaN in *those* columns — restricting the test is essential in the
    concatenated table, where every row is NaN in some column belonging to a different
    stimulus. Pass `()` for stimuli that have no blank sweeps.
    """
    trials = (
        stim_table.loc[stim_table["stim_name"] == stim_name]
        .sort_values("start_time")
        .reset_index(drop=True)
    )
    if len(param_columns):
        present = [c for c in param_columns if c in trials.columns]
        is_blank = trials[present].isna().any(axis=1).to_numpy() if present else np.zeros(len(trials), bool)
    else:
        is_blank = np.zeros(len(trials), dtype=bool)
    return trials, is_blank


def spontaneous_block(nwb, which: int = 0) -> Tuple[float, float]:
    """Start/stop of a spontaneous epoch, in seconds.

    The original drew its null distribution from `spont_stim_table.iloc[0]` — the first
    spontaneous block only. These sessions have exactly one, so first and only coincide,
    but `which` is explicit so that a session with more than one cannot silently change
    the answer.
    """
    ep = epoch_table(nwb)
    spont = ep.loc[ep["stim_name"] == "spontaneous"].reset_index(drop=True)
    if not len(spont):
        raise LookupError("no spontaneous epoch in this session")
    if which >= len(spont):
        raise IndexError(f"requested spontaneous block {which} of {len(spont)}")
    row = spont.iloc[which]
    return float(row["start_time"]), float(row["stop_time"])


def load_running_speed(nwb) -> Tuple[np.ndarray, np.ndarray]:
    """`(speed_cm_s, timestamps)`, on the behaviour clock — not the imaging clock.

    Already differentiated by the NWB writer. The old client computed a central
    difference of cumulative distance itself, so small disagreements here propagate into
    the running/stationary surround-suppression variants, which threshold at exactly
    1 cm/s.
    """
    series = nwb.processing["behavior"]["running_speed"]
    return (
        np.asarray(series.data[:], dtype=np.float64),
        np.asarray(series.timestamps[:], dtype=np.float64),
    )


def _images_to_array(images) -> Tuple[np.ndarray, List[int]]:
    """Stack an NWB `Images` container into (n, rows, cols), ordered by integer name."""
    named = getattr(images, "images", None) or {}
    keys = sorted(named, key=lambda k: int(k))
    stack = np.stack([np.asarray(named[k].data[:]) for k in keys])
    return stack, [int(k) for k in keys]


def load_lsn_template(
    nwb,
    downsample_to: Optional[Tuple[int, int]] = (8, 14),
    grid_size_deg: float = 9.3,
) -> Dict[str, Any]:
    """The locally-sparse-noise template, reduced to the grid the RF code expects.

    This is the one genuinely uncertain piece of the port. `locally_sparse_noise.py`
    loaded an 8x14 grid from `lsn_9deg_28degExclusion_jun_256.npy`, and has a
    commented-out block labelled "Incorrect stimulus" pointing at the 16x28 tif — which
    is what NWB embeds. They are almost certainly the same stimulus at 2x sampling, so
    we block-reduce and **assert every 2x2 block is uniform**. If that assert fails the
    two are not the same stimulus and receptive fields are not reproducible from this
    asset; better to find out here than to ship retinotopy on the wrong scale.

    Returns a dict with `images`, `frames`, `azimuths`, `altitudes`, `native_shape`,
    `reduced`, and `blocks_uniform`, so a caller can inspect the failure instead of only
    catching an exception.
    """
    images, frames = _images_to_array(nwb.stimulus["locally_sparse_noise"])
    native_shape = tuple(images.shape[1:])
    out: Dict[str, Any] = {
        "frames": frames,
        "native_shape": native_shape,
        "reduced": False,
        "blocks_uniform": None,
        "grid_size_deg": grid_size_deg,
    }

    if downsample_to is not None and native_shape != tuple(downsample_to):
        want_r, want_c = downsample_to
        have_r, have_c = native_shape
        if have_r % want_r or have_c % want_c:
            out["images"] = images
            out["error"] = (
                f"template {native_shape} is not an integer multiple of {downsample_to}; "
                "cannot reduce to the grid the RF code was written against"
            )
            out["azimuths"], out["altitudes"] = _grid_degrees(have_c, have_r, grid_size_deg)
            return out

        fr, fc = have_r // want_r, have_c // want_c
        blocks = images.reshape(len(images), want_r, fr, want_c, fc)
        uniform = bool(np.all(blocks == blocks[:, :, :1, :, :1]))
        out["blocks_uniform"] = uniform
        out["block_factor"] = (fr, fc)
        if not uniform:
            out["images"] = images
            out["error"] = (
                f"{fr}x{fc} blocks are not uniform, so the {native_shape} template is not "
                f"a {fr}x{fc} upsample of a {downsample_to} grid — receptive fields "
                "cannot be reproduced from this asset"
            )
            out["azimuths"], out["altitudes"] = _grid_degrees(have_c, have_r, grid_size_deg)
            return out

        images = blocks[:, :, 0, :, 0]
        out["reduced"] = True

    rows, cols = images.shape[1:]
    out["images"] = images
    out["azimuths"], out["altitudes"] = _grid_degrees(cols, rows, grid_size_deg)
    return out


def _grid_degrees(n_cols: int, n_rows: int, grid: float) -> Tuple[np.ndarray, np.ndarray]:
    """Screen-centred pixel-centre coordinates, matching `LocallySparseNoise._load_frames`."""
    azimuths = (np.arange(n_cols) - n_cols // 2 + 0.5) * grid
    altitudes = (np.arange(n_rows) - n_rows // 2 + 0.5) * grid
    return azimuths, altitudes


# --------------------------------------------------------------------- schema report


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
        ep = epoch_table(nwb)
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
        st = load_stimulus_table(nwb)
        cols = list(st.columns)
        report["stimulus_table"] = {
            "n_rows": int(len(st)),
            "columns": cols,
            "expected_columns": STIMULUS_TABLE_COLUMNS,
            "missing_vs_expected": [c for c in STIMULUS_TABLE_COLUMNS if c not in cols],
            "extra_vs_expected": [
                c for c in cols if c not in STIMULUS_TABLE_COLUMNS and c != "id"
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
    plane_keys = list_planes(nwb)
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
        speed, ts = load_running_speed(nwb)
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
        lsn = load_lsn_template(nwb)
        report["lsn_template"] = {
            "n_frames": len(lsn["frames"]),
            "native_shape": list(lsn["native_shape"]),
            "reduced": lsn["reduced"],
            "blocks_uniform": lsn["blocks_uniform"],
            "block_factor": list(lsn.get("block_factor", ())) or None,
            "final_shape": [int(x) for x in lsn["images"].shape[1:]],
            "azimuth_range": [float(lsn["azimuths"][0]), float(lsn["azimuths"][-1])],
            "altitude_range": [float(lsn["altitudes"][0]), float(lsn["altitudes"][-1])],
            "pixel_values": sorted(int(v) for v in np.unique(lsn["images"])),
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
        trials, is_blank = stimulus_trials(st, name, DG_PARAM_COLUMNS)
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
        trials, _ = stimulus_trials(st, name)
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

    trials, _ = stimulus_trials(st, "natural_movie")
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

    trials, _ = stimulus_trials(st, "locally_sparse_noise")
    if len(trials):
        frames = trials["frame"].dropna().to_numpy().astype(int)
        out["locally_sparse_noise"] = {
            "n_sweeps": int(len(trials)),
            "frame_range": [int(frames.min()), int(frames.max())],
            "n_unique_frames": int(len(np.unique(frames))),
            "sweep_duration_s": _duration_summary(trials),
        }

    trials, _ = stimulus_trials(st, "spontaneous")
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
