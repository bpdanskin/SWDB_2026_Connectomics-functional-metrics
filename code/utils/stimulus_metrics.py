"""Stimulus-response metrics, ported from `allen_v1dd` onto NWB-Zarr.

One family per function, each returning a tidy per-ROI DataFrame with the same column
names the historical `data_frames/*_M409828.csv` tables use, so a regenerated table
drops in where the old one was. Comparing against those tables is
`validation/compare.py`'s job, not this module's.

**The defaults compute the right thing; `REFERENCE_CONFIG` reproduces the historical
tables.** Every place the original did something defensible-but-wrong is named in a
`MetricConfig` flag, and both behaviours stay reachable.

That ordering was reversed during the port, deliberately. Matching first is what made the
port checkable: if corrections had gone in alongside the translation, every disagreement
with the historical tables would have been ambiguous between "we fixed a bug" and "we
introduced one", and there would have been no signal left to validate against. Only once
all seven families matched did the known defects get corrected — so each correction is a
single, isolated, measurable change rather than part of a fog.

The two that changed the numbers are documented on the flags below. Neither is subtle:
receptive-field centres were compressed by a constant factor, and preferred condition
reported condition 0 for ROIs that had no response at all.

Three facts about the original that are easy to get wrong, all verified against the data:

* Metrics use **deconvolved events**, except receptive fields, which use dF/F.
* `frac_responsive_trials` means three different things. Drifting gratings and natural
  images: the fraction of preferred-condition trials beating a bootstrapped spontaneous
  null at p < 0.05. Natural movie: the fraction of repeats with **any** response above
  zero — no null, no threshold, no stochasticity.
* Nothing in the original is seeded, so its published bootstrap numbers are one
  unreproducible draw. Pass an explicit `rng` and record the seed.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import trial_responses as tr

__all__ = [
    "MetricConfig",
    "DEFAULT_CONFIG",
    "REFERENCE_CONFIG",
    "OUTPUT_COLUMNS",
    "natural_movie_metrics",
    "natural_images_metrics",
    "receptive_field_metrics",
    "drifting_gratings_metrics",
    "surround_suppression_metrics",
    "locomotion_metrics",
    "window_containment",
    "DGResult",
    "SSI_COLUMNS",
    "CONTAINMENT_COLUMNS",
    "LOCOMOTION_COLUMNS",
    "vonmises_two_peak",
    "vonmises_two_peak_fit",
    "vonmises_pref_dir",
    "roi_frame",
    "to_output_schema",
    "absent_frame",
]


@dataclass(frozen=True)
class MetricConfig:
    """Knobs, defaulted to computing the right thing.

    Three defaults differ from what reproduces the historical tables — `rf_center_scale_bug`,
    `pref_cond_fillna` and `ni_response_frames` — and each is documented where it is
    declared. `REFERENCE_CONFIG` is the historical set, so both behaviours are one
    argument away and which one you asked for is written down rather than inferred.
    """

    # --- trace type per family; events everywhere except receptive fields
    trace_type: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({
        "drifting_gratings_full": "events",
        "drifting_gratings_windowed": "events",
        "natural_images": "events",
        "natural_images_12": "events",
        "natural_movie": "events",
        "locally_sparse_noise": "dff",
    }))

    # --- response windows
    #: Drifting-grating response window, seconds. The original read `duration_sec` (2.0)
    #: from an NWB attribute; the per-trial rows in this asset say stop-start is 1.985 s.
    #: 2.0 reproduces the published numbers. "per_trial" uses each sweep's own duration.
    dg_response_seconds: Any = 2.0
    #: Natural-images response window, seconds. **Recovered empirically, not read from
    #: the data**: the original took it from an NWB `duration_sec` attribute the current
    #: files no longer carry. Scanning it against the published table gives a sharp
    #: optimum at 0.33 s (median |diff| in lifetime_sparseness = 1e-16, i.e. exact),
    #: while 0.30 s gives 6e-3 and 0.35 s gives 2e-3.
    #:
    #: The reason is discrete. At dt = 0.165 s the samples after an onset sit at
    #: delta, delta+dt, delta+2dt with delta in [0, dt). A 0.33 s window is just under
    #: 2*dt = 0.33008, so it catches **exactly two samples on every trial**; a 0.30 s
    #: window catches two when delta <= 0.135 and one otherwise. That varying count
    #: rescales each trial differently, which `lifetime_sparseness` detects because it
    #: is invariant to a *global* scale but not a per-trial one.
    #:
    #: Used only when `ni_response_frames` is None. Kept because it is what reproduces
    #: the historical tables — see `REFERENCE_CONFIG` — and because the reasoning above
    #: is the evidence for `ni_response_frames = 2`.
    #:
    #: **It does not generalise.** The margin is 8e-5 s, and the pre-flight found dt
    #: spanning 0.16123-0.16671 across the 25 sessions. 0.33 s catches exactly two
    #: samples only where 2*dt lands just above it — dt 0.16504 and 0.16506, which are
    #: precisely the two sessions the value was recovered on. Elsewhere it takes three
    #: samples on up to 4.7 % of trials or one on up to 2.1 %, reintroducing exactly the
    #: per-trial rescaling described above.
    ni_response_seconds: Any = 0.33
    #: Natural-images response window as a FIXED number of imaging samples from each
    #: onset. When set (the default), `ni_response_seconds` is ignored.
    #:
    #: Two samples, because that is what the 0.33 s window was doing on the sessions it
    #: was tuned against — but expressed in the units the intent actually lives in, so it
    #: cannot drift with the sampling rate. A time window varies with onset phase; a
    #: frame count does not. Since per-trial rescaling changes the relative pattern across
    #: images, scale-invariant metrics like `lifetime_sparseness` can tell the two apart,
    #: which is how the discrepancy was found in the first place.
    ni_response_frames: Optional[int] = 2
    #: Natural-movie and LSN windows are counted in *imaging* frames, so they depend on
    #: the plane's own sampling period rather than on the stimulus.
    nm_response_frames: int = 3
    lsn_response_frames: int = 4

    # --- bootstrap
    dg_n_boot: int = 2500
    other_n_boot: int = 10_000
    sig_p_thresh: float = 0.05

    # --- responsiveness thresholds (different per family in the original)
    dg_frac_thresh: float = 0.50
    ni_frac_thresh: float = 0.25
    rf_frac_thresh: float = 0.25

    # --- surround suppression
    running_threshold_cm_s: float = 1.0
    running_pad_seconds: float = 0.10
    #: Radius of the windowed-grating aperture, degrees. **Not recorded anywhere in the
    #: NWB** — there is no size column in the stimulus table. It comes from the V1DD
    #: white paper (Abbasi-Asl et al. 2019), which states a 30 degree diameter twice.
    #: Used only by `window_containment`; no metric that reproduces the historical
    #: tables depends on it.
    dgw_window_radius_deg: float = 15.0
    ssi_min_trials: int = 3

    # --- expensive extras, absent from every published table
    permutation_test_shuffles: int = 0
    chisq_shuffles: int = 0
    fit_tuning_curves: bool = True
    #: Fit the von Mises curve at EVERY spatial frequency, not just the one surround
    #: suppression reads. False is roughly 2x faster over a full run, because drifting
    #: gratings dominate the runtime and half of every fit was being discarded.
    #:
    #: It changes no published column — `ssi_tuning_fit` is the only consumer and it
    #: reads one SF per ROI — but it is **not** invisible in the asset: `tuning_curves`
    #: exports `dgw_params` / `dgf_params`, and under False the unread SF is NaN. That
    #: reads as a failed fit unless you know better. Set True for a completeness run.
    #:
    #: The original fitted every SF, so `REFERENCE_CONFIG` sets this True and a fast run
    #: therefore shows up in `differs_from_reference_config`. That is deliberate: the
    #: block answers "what did this run do differently", and once the parameters ship,
    #: this qualifies.
    fit_all_sf: bool = False

    # --- historical compatibility. These default to the CORRECTED behaviour; set both
    # True — or just use REFERENCE_CONFIG — to reproduce the historical tables exactly.
    #: `point_to_alt_azi` in the original divides the centre-to-centre *range* by `n`
    #: rather than `n - 1`, so its degree scale is compressed by `(n-1)/n`: 12.5 % in
    #: altitude (8 rows) and 7.1 % in azimuth (14 columns). The historical tables
    #: therefore span ±28.481° and ±56.132° where the screen actually spans ±32.55° and
    #: ±60.45°. Shipping the compressed scale means anyone who plots retinotopy plots it
    #: wrong, so the default is the true mapping. The two differ by exactly `n/(n-1)`,
    #: which is what makes the correction verifiable rather than merely asserted.
    rf_center_scale_bug: bool = False
    #: The original takes `preferred_dir`/`preferred_sf` from `fillna(-1).argmax`, so an
    #: ROI with no finite response at any condition reports condition **0** rather than
    #: "no preferred condition" — while every other metric in the same function uses a
    #: nan-skipping argmax. False makes the two agree and leaves those ROIs NaN. It
    #: touches only all-NaN rows, but surround suppression keys off the preferred
    #: condition, so a fabricated preference propagates.
    pref_cond_fillna: bool = False

    memory_budget_mb: float = 64.0


DEFAULT_CONFIG = MetricConfig()

#: The settings that reproduce the historical `data_frames` tables, defects and all.
#:
#: This exists so that "does this pipeline still match the original?" stays a question you
#: can ask in one line, after the defaults moved on to computing the right thing. The
#: validation notebook runs both: this one to prove the port is still faithful, the
#: defaults to produce what ships, and the difference between them is then a specific,
#: measurable quantity rather than an unexplained disagreement.
REFERENCE_CONFIG = MetricConfig(
    rf_center_scale_bug=True,      # centres compressed by (n-1)/n
    pref_cond_fillna=True,         # all-NaN ROIs report condition 0
    ni_response_frames=None,       # fall back to the recovered time window
    ni_response_seconds=0.33,
    fit_all_sf=True,               # the original fitted every SF, not just the read one
)


#: Published column names and order, per output file. `surround_supression_index` keeps
#: the misspelling from the published filename.
OUTPUT_COLUMNS: Dict[str, Sequence[str]] = {
    "drifting_gratings_full": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi", "depth_um",
        "pika_roi_confidence",
        "dsi", "frac_responsive_trials", "gosi", "is_responsive",
        "lifetime_sparseness", "osi", "preferred_dir", "preferred_sf", "pref_dir_mean"],
    "natural_images": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi", "depth_um",
        "pika_roi_confidence",
        "frac_responsive_trials", "lifetime_sparseness", "pref_img", "pref_response",
        "z_score", "reliability", "reliability_dff"],
    "surround_supression_index": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi", "depth_um",
        "pika_roi_confidence",
        "ssi", "ssi_avg", "ssi_avg_at_pref_sf", "ssi_running",
        "ssi_running_avg_at_pref_sf", "ssi_stationary",
        "ssi_stationary_avg_at_pref_sf", "ssi_tuning_fit",
        "dgw_center_azimuth", "dgw_center_elevation",
        "dgw_rf_distance_on", "dgw_rf_distance_off",
        "dgw_rf_overlap_on", "dgw_rf_overlap_off"],
    "locomotion": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi", "depth_um",
        "pika_roi_confidence",
        "run_frac", "spont_run_frac",
        "spont_rate", "spont_rate_run", "spont_rate_stat",
        "run_mod_dgf", "run_mod_dgw", "run_mod_spont"],
    "rf_metrics": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi", "depth_um",
        "pika_roi_confidence",
        "has_rf_on", "has_rf_off", "has_rf_on_or_off",
        "azimuth_rf_on", "altitude_rf_on", "azimuth_rf_off", "altitude_rf_off"],
}
OUTPUT_COLUMNS["drifting_gratings_windowed"] = OUTPUT_COLUMNS["drifting_gratings_full"]
OUTPUT_COLUMNS["natural_images_12"] = OUTPUT_COLUMNS["natural_images"]
OUTPUT_COLUMNS["natural_movie"] = OUTPUT_COLUMNS["natural_images"]


# --------------------------------------------------------------------- identity


def roi_frame(plane, mouse: Optional[str] = None) -> pd.DataFrame:
    """The identity columns every output table starts with.

    `roi_unique_id` reproduces the published format `M{mouse}_{volume}_{plane}_{roi}`,
    which **omits the column** and therefore collides across the five columns — 164,345
    published rows share only 56,449 distinct ids. It is emitted for drop-in
    compatibility; `roi_key` is the non-colliding version. **Join on
    `(column, volume, plane, roi)`, never on either string.**
    """
    n = plane.n_rois
    # The mouse comes from the file, not from a constant: this pipeline is expected to
    # run on other animals. `mouse` overrides only when a caller genuinely knows better.
    mouse_num = (mouse or "").lstrip("M") or getattr(plane, "mouse_id", "")
    if not mouse_num:
        raise ValueError(
            "no mouse id: pass mouse=, or load the plane with load_plane(), which reads "
            "it from nwb.subject via session_mouse()"
        )
    mouse = f"M{mouse_num}"
    return pd.DataFrame({
        "roi_unique_id": [f"M{mouse_num}_{plane.volume}_{plane.plane}_{r}" for r in plane.roi],
        "roi_key": [
            f"M{mouse_num}_{plane.column}_{plane.volume}_{plane.plane}_{r}" for r in plane.roi
        ],
        "mouse": [mouse] * n,
        "column": np.full(n, plane.column, dtype=int),
        "volume": [plane.volume] * n,
        "plane": np.full(n, plane.plane, dtype=int),
        "roi": plane.roi.astype(int),
        # Physical depth, which (column, volume, plane) only encodes implicitly. NaN when
        # the file does not carry it -- no metric depends on it.
        "depth_um": np.full(n, getattr(plane, "depth_um", None)
                            if getattr(plane, "depth_um", None) is not None else np.nan,
                            dtype=float),
        # Segmentation confidence, emitted so consumers can see which ROIs the pipeline
        # treated as unreliable. Without it, low-confidence ROIs are neither dropped nor
        # labelled: `preferred_dir`, the `ssi*` columns and every receptive-field column
        # are suppressed for them, but `osi`, `dsi`, `lifetime_sparseness` and the natural
        # scene metrics are populated as usual, so they enter any population average
        # unnoticed. `is_valid` is this column > 0.5.
        "pika_roi_confidence": _roi_confidence(plane),
    })


def _roi_confidence(plane) -> np.ndarray:
    """Per-ROI segmentation confidence, or NaN where the ROI table does not carry it."""
    table = getattr(plane, "roi_table", None)
    if table is None or "pika_roi_confidence" not in getattr(table, "columns", []):
        return np.full(plane.n_rois, np.nan, dtype=float)
    return pd.to_numeric(table["pika_roi_confidence"], errors="coerce").to_numpy(float)


def to_output_schema(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Reorder and dtype a metrics frame to the asset's output schema.

    The column order matches the historical `data_frames` tables, including the
    `surround_supression_index` misspelling, so a regenerated table drops in where the old
    one was.
    """
    cols = list(OUTPUT_COLUMNS[family])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{family}: missing published columns {missing}")
    out = df[cols].copy()
    out["column"] = out["column"].astype(int)
    out["volume"] = out["volume"].astype(str)
    out["plane"] = out["plane"].astype(int)
    out["roi"] = out["roi"].astype(int)
    if "pref_img" in out:                        # published uses int with a -1 sentinel
        out["pref_img"] = out["pref_img"].fillna(-1).astype(int)
    if "is_responsive" in out:                   # published writes float 0.0/1.0
        out["is_responsive"] = out["is_responsive"].astype(float)
    for c in ("has_rf_on", "has_rf_off", "has_rf_on_or_off"):
        if c in out:                             # published writes True/False
            out[c] = out[c].astype(bool)
    return out


def _reliability_on(plane, trace_key, starts, codes, *, n_trials, n_conditions,
                    window=None, frames=None):
    """`trial_reliability` recomputed on a second trace type, or NaN if it is absent.

    Reliability is reported twice — once on the trace the family's metrics use (events)
    and once on dF/F — because the two answer different questions and, on sparse events,
    disagree substantially. Events are exactly zero most of the time, so a repeat's
    response vector is mostly flat and its correlation with another repeat rests on a
    handful of frames; dF/F carries a continuous signal and is what the white paper's
    Figure 18 reports. Shipping both makes "how reproducible are the events every other
    metric is built on?" a question the asset can answer.

    Returns NaN for every ROI when `trace_key` is not loaded, so a plane loaded with a
    single trace type still produces a valid frame rather than raising.
    """
    traces = plane.traces.get(trace_key)
    if traces is None:
        return np.full(plane.n_rois, np.nan)
    if frames is not None:
        sweeps = tr.sweep_responses_frames(traces, plane.timestamps, starts, int(frames))
    else:
        sweeps = tr.sweep_responses(traces, plane.timestamps, starts, window, None)
    return tr.trial_reliability(
        tr.trial_array(sweeps, codes, n_trials=n_trials, n_conditions=n_conditions))


def absent_frame(plane, family: str, mouse: Optional[str] = None) -> pd.DataFrame:
    """Identity rows with no metrics, for a session that did not run this stimulus.

    The pre-flight found all six families present in all 25 sessions of this asset, so
    this is insurance rather than a code path in daily use. It exists because
    `stimulus_trials` returns an *empty frame* for a missing stimulus instead of raising:
    without a guard, an absent stimulus would flow into the metric functions and come out
    as confident nonsense rather than as an absence.

    Booleans are set False rather than left NaN — `to_output_schema` casts them with
    `astype(bool)`, and `bool(nan)` is **True**, which would report a receptive field for
    every ROI in a session that never saw the stimulus.
    """
    out = roi_frame(plane, mouse=mouse)
    for column in OUTPUT_COLUMNS[family]:
        if column in out:
            continue
        out[column] = False if column.startswith("has_rf_") else np.nan
    return out


# --------------------------------------------------------------------- helpers


def _lifetime_sparseness_chunked(ta: np.ndarray, block: int = 256) -> np.ndarray:
    """Lifetime sparseness over a (n_conditions, n_trials, n_rois) trial array.

    Same closed form as `trial_responses.lifetime_sparseness`, accumulated in blocks over
    the condition axis: natural movie has 3,600 conditions x 9 repeats x ~480 ROIs, and
    flattening that to call the reference implementation would copy 125 MB per plane for
    no reason.
    """
    n_cond, n_trials, n_rois = ta.shape
    n = np.zeros(n_rois)
    s1 = np.zeros(n_rois)
    s2 = np.zeros(n_rois)
    for s in range(0, n_cond, block):
        x = ta[s : s + block]
        finite = np.isfinite(x)
        x0 = np.where(finite, x, 0.0)
        n += finite.sum(axis=(0, 1))
        s1 += x0.sum(axis=(0, 1))
        s2 += np.einsum("ijk,ijk->k", x0, x0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean, mean_sq = s1 / n, s2 / n
        return (1.0 - mean**2 / mean_sq) / (1.0 - 1.0 / n)


def _nanmean(x: np.ndarray, axis) -> np.ndarray:
    """`np.nanmean` without the all-NaN-slice warning, which fires constantly here."""
    finite = np.isfinite(x)
    n = finite.sum(axis=axis)
    total = np.where(finite, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)


# --------------------------------------------------------------------- natural movie


def natural_movie_metrics(
    plane,
    trials: pd.DataFrame,
    spont: Sequence[float],
    *,
    config: MetricConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
    mouse: Optional[str] = None,
) -> pd.DataFrame:
    """Natural-movie metrics: one row per ROI.

    Every movie frame is a "trial" and every pass through the movie a "repeat". Note the
    response window spans ~3 imaging frames ≈ 0.49 s while movie frames are 1/30 s apart,
    so consecutive "trials" overlap heavily and are strongly autocorrelated. That is the
    original's design; it means `lifetime_sparseness` over 3,600 x 9 such values is not
    measuring what its name suggests.

    A sharper consequence, worth knowing before interpreting `pref_img`: because the
    window looks *forward* from each frame's onset, activity driven by frame f lands
    inside the windows of frames f-15 … f. The reported preferred frame can therefore
    **precede** the frame that actually drove the response by up to half a second, and
    among those overlapping windows the argmax is decided partly by how many imaging
    samples each happens to contain. Treat `pref_img` as locating a ~0.5 s neighbourhood,
    not a frame.

    `frac_responsive_trials` here is **not** a statistical test — it is the fraction of
    repeats whose mean response at the preferred frame is strictly greater than zero. No
    bootstrap is involved, which makes this the one fully deterministic end-to-end check
    against the published table.
    """
    if not len(trials):
        return absent_frame(plane, "natural_movie", mouse)
    rng = np.random.default_rng() if rng is None else rng
    traces = plane.traces[config.trace_type["natural_movie"]]
    window = (0.0, config.nm_response_frames * plane.dt)

    starts = trials["start_time"].to_numpy(dtype=np.float64)
    frames = trials["frame"].to_numpy()
    if np.isnan(frames).any():
        raise ValueError("natural_movie trials contain NaN frame indices")
    frames = frames.astype(int)

    frame_ids = np.arange(frames.max() + 1)
    if not np.array_equal(np.unique(frames), frame_ids):
        raise ValueError(
            "movie frames are not contiguous from 0; the original indexes them "
            "positionally and by label interchangeably, which only works if they are"
        )

    sweeps = tr.sweep_responses(traces, plane.timestamps, starts, window, None)
    n_repeats = int(np.bincount(frames).max())
    ta = tr.trial_array(sweeps, frames, n_trials=n_repeats, n_conditions=len(frame_ids))

    mean_resp = _nanmean(ta, axis=1)                       # (n_frames, n_rois)
    n_rois = plane.n_rois
    roi_ix = np.arange(n_rois)

    all_nan = np.all(~np.isfinite(mean_resp), axis=0)
    safe = np.where(np.isfinite(mean_resp), mean_resp, -np.inf)
    pref_idx = safe.argmax(axis=0)
    pref_response = np.where(all_nan, np.nan, mean_resp[pref_idx, roi_ix])
    pref_img = np.where(all_nan, -1, frame_ids[pref_idx])

    # fraction of repeats with any response at the preferred frame
    pref_trials = ta[pref_idx, :, roi_ix]                  # (n_rois, n_repeats)
    sig = np.where(np.isfinite(pref_trials), (pref_trials > 0).astype(float), np.nan)
    frac_responsive = _nanmean(sig, axis=1)

    null = tr.spontaneous_null(
        traces, plane.timestamps, spont[0], spont[1], window, None,
        n_boot=config.other_n_boot, n_means=n_repeats, rng=rng,
        memory_budget_mb=config.memory_budget_mb,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        z_score = (pref_response - null.mean(axis=1)) / null.std(axis=1)

    out = roi_frame(plane, mouse=mouse)
    out["frac_responsive_trials"] = frac_responsive
    out["lifetime_sparseness"] = _lifetime_sparseness_chunked(ta)
    out["reliability"] = tr.trial_reliability(ta)
    out["reliability_dff"] = _reliability_on(
        plane, "dff", starts, frames, n_trials=n_repeats,
        n_conditions=len(frame_ids), window=window)
    out["pref_img"] = pref_img
    out["pref_response"] = pref_response
    out["z_score"] = z_score
    return out


# --------------------------------------------------------------- drifting gratings


@dataclass
class DGResult:
    """Drifting-gratings metrics plus the intermediates surround suppression needs."""

    metrics: pd.DataFrame                 # per-ROI, published column names
    trial_responses: np.ndarray           # (n_rois, n_dir, n_sf, n_trials), NaN-padded
    trial_running_speeds: np.ndarray      # (n_dir, n_sf, n_trials) cm/s -- no ROI axis
    pref_cond_index: np.ndarray           # (n_rois, 2) [dir_idx, sf_idx], -1 if invalid
    tuning_params: np.ndarray             # (n_rois, n_sf, 6) von Mises, NaN if no fit
    dir_list: np.ndarray
    sf_list: np.ndarray
    blank_responses: np.ndarray           # (n_rois, n_blank)
    center: Tuple[float, float] = field(default_factory=lambda: (np.nan, np.nan))
    # (center_azimuth_deg, center_elevation_deg) of the grating aperture; NaN for full-field


def vonmises_two_peak(x, scale_1, k_1, x0, scale_2, k_2, b):
    """Two 180-degree-opposed von Mises bumps plus an offset. x is in degrees."""
    x = np.asarray(x, dtype=np.float64)
    return (scale_1 * np.exp(k_1 * np.cos(np.deg2rad(x - x0)))
            + scale_2 * np.exp(k_2 * np.cos(np.deg2rad(x - x0 - 180)))
            + b)


_VONMISES_BOUNDS = (
    (0, 0, 0, 0, 0, 0),
    (np.inf, np.inf, 360, np.inf, np.inf, np.inf),
)


def vonmises_two_peak_fit(x, y, p0=(0.1, 1, 180, 0.01, 1, 0.001),
                          max_fn_calls=(2000, 10000)):
    """Least-squares fit, or None if it never converges. Mirrors `fit_utils`."""
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(y)
    if good.sum() < 6:
        return None
    for maxfev in max_fn_calls:
        try:
            params, _ = curve_fit(vonmises_two_peak, x[good], y[good], maxfev=maxfev,
                                  bounds=_VONMISES_BOUNDS, p0=p0)
            return params
        except (RuntimeError, ValueError):
            continue
    return None


def vonmises_pref_dir(params) -> float:
    """Preferred direction of a fitted curve: whichever of the two peaks is taller.

    Peak height here is baseline-*subtracted* (f(x) - b), matching
    `vonmises_two_peak_get_amplitude`. Note that `ssi_tuning_fit` then evaluates the
    curve *including* b -- an inconsistency in the original that we reproduce.
    """
    x0 = float(params[2])
    x1 = (x0 + 180.0) % 360.0
    a0 = vonmises_two_peak(x0, *params) - params[-1]
    a1 = vonmises_two_peak(x1, *params) - params[-1]
    return x0 if a0 > a1 else x1


def _ratio(p, q):
    """The original's `ratio`: **0** when the denominator is 0, not NaN."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(q == 0, 0.0, p / np.where(q == 0, 1.0, q))


def _metric_index(a, b):
    """The SSI index: **NaN** when the denominator is 0, unlike `_ratio`."""
    s = np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s == 0, np.nan, (a - b) / np.where(s == 0, 1.0, s))


def _condition_codes(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Index of each value in `levels`.

    Exact lookup rather than tolerance-based: both sides come from the same column, so a
    float32 round-trip (0.04 stored as 0.039999999) matches itself.
    """
    idx = np.clip(np.searchsorted(levels, values), 0, len(levels) - 1)
    if not np.array_equal(levels[idx], values):
        raise ValueError("condition values are not exactly present in the level list")
    return idx


def drifting_gratings_metrics(
    plane,
    trials: pd.DataFrame,
    is_blank: np.ndarray,
    spont: Sequence[float],
    running: Optional[Sequence[np.ndarray]] = None,
    *,
    dg_type: str = "full",
    fit_sf_index: Optional[np.ndarray] = None,
    config: MetricConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
    mouse: Optional[str] = None,
) -> "DGResult":
    """Drifting-gratings metrics for one plane.

    `dg_type` is "full" or "windowed". The computation is identical for both; surround
    suppression is what compares them.

    Two preferred conditions are computed, deliberately. `preferred_dir`/`preferred_sf` in
    the published table come from an argmax over `fillna(-1)` responses, while the
    selectivity indices come from a NaN-skipping argmax with no fill. They disagree only
    for ROIs whose condition means are NaN. Both are kept, and a divergence warns, because
    surround suppression keys off the first and `osi`/`dsi` off the second -- a silent
    divergence would corrupt SSI without touching any drifting-gratings column.
    """
    if not len(trials):
        raise ValueError(
            f"no drifting_gratings_{dg_type} sweeps for column {plane.column} "
            f"volume {plane.volume} plane {plane.plane}. Surround suppression consumes "
            "this result, so there is no empty value that stays honest downstream -- "
            "skip the session instead."
        )
    import warnings

    rng = np.random.default_rng() if rng is None else rng
    family = f"drifting_gratings_{dg_type}"
    traces = plane.traces[config.trace_type[family]]

    if config.dg_response_seconds == "per_trial":
        raise NotImplementedError("per-trial windows need a per-sweep window API")
    window = (0.0, float(config.dg_response_seconds))

    starts = trials["start_time"].to_numpy(dtype=np.float64)
    sweeps = tr.sweep_responses(traces, plane.timestamps, starts, window, None)

    grat = trials.loc[~is_blank]

    # per-session grating-aperture centre (NaN for full-field, which uses (0, 0) placeholders)
    az_vals = grat["center_azimuth"].dropna().unique() if "center_azimuth" in grat.columns else []
    el_vals = grat["center_elevation"].dropna().unique() if "center_elevation" in grat.columns else []
    center: Tuple[float, float] = (
        float(az_vals[0]) if len(az_vals) else np.nan,
        float(el_vals[0]) if len(el_vals) else np.nan,
    )

    dir_list = np.sort(grat["direction"].dropna().unique())
    sf_list = np.sort(grat["spatial_frequency"].dropna().unique())
    if len(dir_list) != 12:
        raise ValueError(
            f"{family}: found {len(dir_list)} directions, expected 12. The orthogonal and "
            "null directions are hard-coded as (i +/- 3) % 12 and (i + 6) % 12, which "
            "compute silently wrong values for any other count."
        )
    n_dir, n_sf = len(dir_list), len(sf_list)

    d = _condition_codes(grat["direction"].to_numpy(), dir_list)
    s = _condition_codes(grat["spatial_frequency"].to_numpy(), sf_list)
    code = d * n_sf + s
    n_trials = int(np.bincount(code, minlength=n_dir * n_sf).max())

    ta = tr.trial_array(sweeps[~is_blank], code, n_trials=n_trials,
                        n_conditions=n_dir * n_sf)
    ta = ta.reshape(n_dir, n_sf, n_trials, plane.n_rois).transpose(3, 0, 1, 2)

    blank = sweeps[is_blank].T if bool(is_blank.any()) else np.empty((plane.n_rois, 0))

    mean_tr = _nanmean(ta, axis=3)                       # (n_rois, n_dir, n_sf)
    n_rois = plane.n_rois
    roi_ix = np.arange(n_rois)

    # Historical preferred condition: argmax over fillna(-1), C order.
    k_fill = np.nan_to_num(mean_tr, nan=-1.0).reshape(n_rois, -1).argmax(axis=1)
    pref_dir_fill, pref_sf_fill = np.divmod(k_fill, n_sf)

    # selectivity-index preferred condition: NaN-skipping argmax, no fill
    k_skip = np.where(np.isfinite(mean_tr), mean_tr, -np.inf).reshape(n_rois, -1).argmax(axis=1)
    pref_dir_idx, pref_sf_idx = np.divmod(k_skip, n_sf)

    n_diverge = int(np.sum(k_fill != k_skip))
    if n_diverge:
        warnings.warn(
            f"{family}: the two preferred-condition definitions disagree on {n_diverge} "
            f"of {n_rois} ROIs; SSI uses the fillna(-1) one, osi/dsi the other"
        )

    # Which definition `preferred_dir`/`preferred_sf` and surround suppression use.
    #
    # Neither argmax abstains: `fillna(-1)` and the `-inf` fill both return index 0 for an
    # ROI with no finite response at any condition, so both invent a preference where
    # there is no evidence of one. Correcting the choice of definition is therefore not
    # enough — those ROIs have to be marked as having no preferred condition, which is
    # what `-1` means here and what makes `preferred_dir` come out NaN.
    k_pref = k_fill if config.pref_cond_fillna else k_skip
    pref_dir_pref, pref_sf_pref = np.divmod(k_pref, n_sf)
    pref_cond_index = np.stack([pref_dir_pref, pref_sf_pref], axis=1).astype(int)
    pref_cond_index[~plane.is_valid] = -1
    if not config.pref_cond_fillna:
        no_response = ~np.isfinite(mean_tr).any(axis=(1, 2))
        pref_cond_index[no_response] = -1

    tuning = mean_tr[roi_ix, :, pref_sf_idx]             # (n_rois, n_dir) at preferred SF
    pref = tuning[roi_ix, pref_dir_idx]
    null_r = tuning[roi_ix, (pref_dir_idx + 6) % 12]
    orth_r = 0.5 * (tuning[roi_ix, (pref_dir_idx + 3) % 12]
                    + tuning[roi_ix, (pref_dir_idx - 3) % 12])

    osi = _ratio(pref - orth_r, pref + orth_r)
    dsi = _ratio(pref - null_r, pref + null_r)

    theta = np.deg2rad(dir_list.astype(float))
    # gosi normalises with a NaN-PROPAGATING sum, matching `_compute_osi`.
    #
    # Neither this nor `osi` above rectifies the tuning curve, and that is only safe
    # because `trace_type` for both grating families is `events` -- deconvolved, so
    # non-negative, so `L_norm` cannot be a small or negative number built by
    # cancellation. On a *signed* trace (baseline-subtracted dF/F) the same lines are
    # wrong: a negative response at the orthogonal direction drives `osi` outside
    # [0, 1], and a near-zero signed `L_norm` makes `gosi` explode. Implementations
    # working from dF/F clip to zero first (`np.clip(R, 0, None)`) for exactly this
    # reason. If `config.trace_type` is ever pointed at `dff` here, rectify first.
    L_norm = tuning.sum(axis=1)
    L_ori = tuning @ np.exp(2j * theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        gosi = np.abs(np.where(L_norm != 0, L_ori / np.where(L_norm != 0, L_norm, 1.0),
                               L_ori))
    # pref_dir_mean treats NaN as ZERO (skipna sum) -- the opposite convention to gosi
    vec = np.nan_to_num(tuning, nan=0.0) @ np.exp(1j * theta)
    pref_dir_mean = np.degrees(np.angle(vec)) % 360.0

    lifetime = _lifetime_sparseness_chunked(
        ta.transpose(1, 2, 3, 0).reshape(-1, n_trials, n_rois))

    null_single = tr.spontaneous_null(
        traces, plane.timestamps, spont[0], spont[1], window, None,
        n_boot=config.dg_n_boot, n_means=1, rng=rng,
        memory_budget_mb=config.memory_budget_mb,
    )
    pref_trials = ta[roi_ix, pref_dir_idx, pref_sf_idx, :]        # (n_rois, n_trials)
    frac = tr.frac_trials_above_null(pref_trials, null_single, p_thresh=config.sig_p_thresh)
    is_responsive = (plane.is_valid & (frac >= config.dg_frac_thresh)).astype(float)

    # per-trial running speed, shape (n_dir, n_sf, n_trials); no ROI axis
    trs = np.full((n_dir, n_sf, n_trials), np.nan)
    if running is not None:
        speed, rts = running
        pad = config.running_pad_seconds
        gstarts = grat["start_time"].to_numpy(dtype=np.float64)
        stops = grat["stop_time"].to_numpy(dtype=np.float64)
        cs, counts = tr.prefix_sums(np.asarray(speed, dtype=np.float64)[:, None])
        a = np.searchsorted(rts, gstarts - pad, side="left")
        b = np.searchsorted(rts, stops + pad, side="right")
        per_sweep = tr.window_means(cs, counts, a, b)              # (n_sweeps, 1)
        trs = tr.trial_array(per_sweep, code, n_trials=n_trials,
                             n_conditions=n_dir * n_sf).reshape(n_dir, n_sf, n_trials)

    # von Mises fits, one per (ROI, spatial frequency).
    # fit_sf_index: per-ROI SF index to fit (0..n_sf-1), skipping the other.
    # Windowed self-selects its own preferred SF — the only one SSI reads from it.
    # Full field must receive dgw.pref_cond_index[:, 1] from the caller, since SSI
    # reads dgf.tuning_params[roi, si] where si = dgw's preferred SF.
    # None → windowed auto-selects; for full field, all SFs are fitted (original behaviour).
    tuning_params = np.full((n_rois, n_sf, 6), np.nan)
    if config.fit_tuning_curves:
        _fit_sf = fit_sf_index
        if _fit_sf is None and dg_type == "windowed":
            _fit_sf = pref_cond_index[:, 1]   # self-select: saves ~half of windowed fits
        if config.fit_all_sf:
            _fit_sf = None        # completeness run: overrides the argument AND the
                                  # windowed self-selection, which has no other escape
        for roi in range(n_rois):
            if not plane.is_valid[roi]:
                continue
            if _fit_sf is not None:
                sf_i = int(_fit_sf[roi])
                if sf_i < 0:
                    continue
                p = vonmises_two_peak_fit(dir_list, mean_tr[roi, :, sf_i])
                if p is not None:
                    tuning_params[roi, sf_i] = p
            else:
                for sf_i in range(n_sf):
                    p = vonmises_two_peak_fit(dir_list, mean_tr[roi, :, sf_i])
                    if p is not None:
                        tuning_params[roi, sf_i] = p

    out = roi_frame(plane, mouse=mouse)
    out["dsi"] = dsi
    out["frac_responsive_trials"] = frac
    out["gosi"] = gosi
    out["is_responsive"] = is_responsive
    out["lifetime_sparseness"] = lifetime
    out["osi"] = osi
    out["preferred_dir"] = np.where(pref_cond_index[:, 0] >= 0,
                                    dir_list[pref_cond_index[:, 0]], np.nan)
    out["preferred_sf"] = np.where(pref_cond_index[:, 1] >= 0,
                                   sf_list[pref_cond_index[:, 1]], np.nan)
    out["pref_dir_mean"] = pref_dir_mean

    return DGResult(metrics=out, trial_responses=ta, trial_running_speeds=trs,
                    pref_cond_index=pref_cond_index, tuning_params=tuning_params,
                    dir_list=dir_list, sf_list=sf_list, blank_responses=blank,
                    center=center)


# ------------------------------------------------------------ surround suppression


SSI_COLUMNS = ["ssi", "ssi_avg", "ssi_avg_at_pref_sf", "ssi_running",
               "ssi_running_avg_at_pref_sf", "ssi_stationary",
               "ssi_stationary_avg_at_pref_sf", "ssi_tuning_fit"]


def surround_suppression_metrics(
    dgw: "DGResult",
    dgf: "DGResult",
    plane,
    *,
    config: MetricConfig = DEFAULT_CONFIG,
    mouse: Optional[str] = None,
    containment: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Eight surround-suppression indices, all of the form (W - F) / (W + F).

    W is the windowed (small-patch) response and F the full-field response. The
    **reference condition is always the windowed stimulus's preferred (direction, SF)**;
    the full-field response is sampled at that same condition, never at its own preferred
    one. ROIs whose preferred condition is -1 stay NaN.

    Running and stationary trials split at exactly 1 cm/s with **strict** inequalities on
    both sides, so a trial at exactly 1.0 belongs to neither. `ssi_running` and
    `ssi_stationary` additionally require at least three qualifying trials in *both*
    stimuli; the `*_avg_at_pref_sf` variants have no such minimum.

    `containment` is the frame from `window_containment`, spliced in so the schema stays
    owned here. It is passed in rather than computed here on purpose: it is a function of
    the receptive-field map and the aperture position, and has nothing to do with the SSI
    arithmetic. Computing it inside would make this family depend on locally sparse noise,
    so a session missing that stimulus would take surround suppression down with it.
    Omit it and the columns are NaN, which is what an absent LSN family should produce.
    """
    n_rois = plane.n_rois
    out = {m: np.full(n_rois, np.nan) for m in SSI_COLUMNS}

    thr = config.running_threshold_cm_s
    W, F = dgw.trial_responses, dgf.trial_responses
    W_run = np.where((dgw.trial_running_speeds > thr)[None], W, np.nan)
    W_stat = np.where((dgw.trial_running_speeds < thr)[None], W, np.nan)
    F_run = np.where((dgf.trial_running_speeds > thr)[None], F, np.nan)
    F_stat = np.where((dgf.trial_running_speeds < thr)[None], F, np.nan)

    def m(x):
        """nan-mean that returns NaN for an all-NaN slice instead of warning."""
        finite = np.isfinite(x)
        return np.nan if not finite.any() else float(np.mean(x[finite]))

    for roi in range(n_rois):
        di, si = dgw.pref_cond_index[roi]
        if di < 0 or si < 0:
            continue

        out["ssi"][roi] = _metric_index(m(W[roi, di, si]), m(F[roi, di, si]))
        out["ssi_avg"][roi] = _metric_index(m(W[roi]), m(F[roi]))
        out["ssi_avg_at_pref_sf"][roi] = _metric_index(m(W[roi, :, si]), m(F[roi, :, si]))
        out["ssi_running_avg_at_pref_sf"][roi] = _metric_index(
            m(W_run[roi, :, si]), m(F_run[roi, :, si]))
        out["ssi_stationary_avg_at_pref_sf"][roi] = _metric_index(
            m(W_stat[roi, :, si]), m(F_stat[roi, :, si]))

        for key, wa, fa in (("ssi_stationary", W_stat, F_stat),
                            ("ssi_running", W_run, F_run)):
            ws, fs = wa[roi, di, si], fa[roi, di, si]
            ws, fs = ws[np.isfinite(ws)], fs[np.isfinite(fs)]
            if len(ws) >= config.ssi_min_trials and len(fs) >= config.ssi_min_trials:
                out[key][roi] = _metric_index(ws.mean(), fs.mean())

        wp, fp = dgw.tuning_params[roi, si], dgf.tuning_params[roi, si]
        if np.isfinite(wp).all() and np.isfinite(fp).all():
            d0 = vonmises_pref_dir(wp)
            # evaluated WITH the fitted baseline b, matching compute_ssi_from_h5 --
            # inconsistent with the pref-dir selection above, which subtracts it
            out["ssi_tuning_fit"][roi] = _metric_index(
                float(vonmises_two_peak(d0, *wp)), float(vonmises_two_peak(d0, *fp)))

    frame = roi_frame(plane, mouse=mouse)
    for k, v in out.items():
        frame[k] = v
    # grating-aperture centre carried through so consumers can filter by RF containment
    az, el = dgw.center
    frame["dgw_center_azimuth"] = np.full(n_rois, az)
    frame["dgw_center_elevation"] = np.full(n_rois, el)
    for c in CONTAINMENT_COLUMNS:
        if containment is None:
            frame[c] = np.full(n_rois, np.nan)
        else:
            if len(containment) != n_rois:
                raise ValueError(
                    f"containment has {len(containment)} rows, plane has {n_rois} ROIs")
            frame[c] = np.asarray(containment[c], dtype=np.float64)
    return frame



# ------------------------------------------------------------------- locomotion


LOCOMOTION_COLUMNS = ["run_frac", "spont_run_frac",
                      "spont_rate", "spont_rate_run", "spont_rate_stat",
                      "run_mod_dgf", "run_mod_dgw", "run_mod_spont"]


def _run_modulation(resp, speeds, thr, min_trials):
    """(R_run - R_stat) / (R_run + R_stat) per ROI, pooled over every trial.

    `resp` is (n_rois, n_dir, n_sf, n_trials), `speeds` (n_dir, n_sf, n_trials).
    """
    if resp is None or speeds is None:
        return None
    run = speeds > thr
    stat = speeds < thr                      # strict both sides: exactly thr is neither,
    if run.sum() < min_trials or stat.sum() < min_trials:   # matching the ssi convention
        return None
    r = _nanmean(np.where(run[None], resp, np.nan).reshape(resp.shape[0], -1), axis=1)
    s = _nanmean(np.where(stat[None], resp, np.nan).reshape(resp.shape[0], -1), axis=1)
    return _metric_index(r, s)


def locomotion_metrics(
    plane,
    dgw: Optional["DGResult"],
    dgf: Optional["DGResult"],
    spont: Sequence[float],
    running: Optional[Sequence[np.ndarray]] = None,
    *,
    config: MetricConfig = DEFAULT_CONFIG,
    mouse: Optional[str] = None,
) -> pd.DataFrame:
    """How much locomotion changes each neuron's activity, across three conditions.

    `(R_run - R_stat) / (R_run + R_stat)`, the conventional index and the same form as
    every `ssi_*` column. **Not** the white paper's `C*(Rmax - Rmin)/Rmax`, which divides
    by the max rather than the sum; that would put a second convention in the same asset
    for no gain, and the two are monotonically related anyway.

    Three conditions, all on **deconvolved events only**:

    * **`run_mod_dgf` / `run_mod_dgw`** — full-field and windowed gratings. Having both
      is a cross-check on `ssi_running` / `ssi_stationary`, which split the same trials at
      the same threshold but only at each ROI's preferred condition.
    * **`run_mod_spont`** — the spontaneous block, where nothing is on the screen. This is
      the control: locomotion modulates cortex whether or not there is a stimulus, so a
      grating index should be read against this rather than against zero.

    `spont_rate` is the mean activity over the whole spontaneous block, with
    `spont_rate_run` / `spont_rate_stat` the same quantity split by state. They are here
    for two reasons. They are the magnitudes behind `run_mod_spont`, so a ratio built from
    two near-zero numbers can be gated rather than trusted. And `spont_rate` is the only
    per-ROI **baseline activity level** in the asset — how much a neuron does with nothing
    on the screen — which is useful well beyond locomotion: as a normaliser for evoked
    responses, and for spotting unusually silent or hyperactive cells.

    Note `spont_rate` is **not** recoverable from the other two. `spont_run_frac` is a
    fraction of *time* on the running trace's own ~59 Hz samples (the white paper's
    definition), while the split above classifies *imaging frames*; the two are close but
    are not the weights that would recombine the state means. So the overall level is
    computed and shipped rather than left to be derived.

    **Pooled over all trials, not computed at the preferred condition.** That is forced by
    the data, not preference. `ssi_running` needs >=3 running trials at one condition out
    of 8, and only **6.3 % of ROIs in 9 of 25 sessions** clear that bar — running here is
    close to all-or-nothing per session, with many sessions 100 % stationary and two 100 %
    running. Pooling over all 192 grating trials makes the same threshold easy wherever
    the animal ran at all.

    `run_frac` (whole session) and `spont_run_frac` (the spontaneous block alone) are
    reported beside them because they say whether any of it is interpretable — the paper
    gates its locomotion analyses at a running fraction of 0.2, and the two fractions can
    differ substantially.

    **Events only, deliberately — a ratio index is not safe on a signed trace.** With
    non-negative events `R_run + R_stat` is a sum of magnitudes and vanishes only for a
    silent cell. On signed dF/F the same expression breaks in two ways: near-cancelling
    responses of opposite sign give an unbounded index (+0.050 vs -0.049 -> 99), and when
    both responses are negative the sign inverts, so a suppressed cell that is *less*
    suppressed while running scores negative.

    For `spont_rate` the case against dF/F is stronger still, and different: dF/F is
    defined against a rolling baseline, so its mean over a long stimulus-free block is
    ~0 **by construction**. It would not be unstable, it would be uninformative.

    The white paper's `C*(Rmax - Rmin)/Rmax` does not rescue the ratio; it is worse. Its
    denominator can itself be negative -- `max(-0.01, -0.05) = -0.01` -- giving -4.0 for
    that same both-negative case. Rectifying, or dividing by `|R_run| + |R_stat|`, would
    fix the sign and the bound, but neither fixes the deeper problem below, and a dF/F
    column would then need a different formula from every other index in this asset.

    **What no denominator fixes**, and which applies to events too: when both responses
    are near zero the ratio is large and meaningless. `R_run = 1e-6, R_stat = 2e-6` gives
    -0.33 under every variant. That is a signal-to-noise problem. Gate on magnitude before
    trusting a value from a quiet cell -- for the gratings the raw per-trial responses and
    running speeds are both in `tuning_curves_*.npz`, so `R_run` and `R_stat` can be
    recomputed and thresholded however you like; for the spontaneous block, gate on
    `spont_rate_run` / `spont_rate_stat`.

    (Note the contrast with `reliability`, which *is* reported on both trace types. That
    is a correlation -- invariant to sign and scale, no denominator -- so the argument
    against a dF/F variant here does not apply there.)
    """
    n_rois = plane.n_rois
    out = {c: np.full(n_rois, np.nan) for c in LOCOMOTION_COLUMNS}
    thr = config.running_threshold_cm_s
    n_min = config.ssi_min_trials

    for name, res in (("dgf", dgf), ("dgw", dgw)):
        if res is None:
            continue
        got = _run_modulation(res.trial_responses, res.trial_running_speeds, thr, n_min)
        if got is not None:
            out[f"run_mod_{name}"] = got

    if running is not None:
        speed, rts = np.asarray(running[0], dtype=np.float64), np.asarray(running[1])
        moving = speed > thr
        out["run_frac"] = np.full(n_rois, float(moving.mean()) if moving.size else np.nan)

        # Spontaneous has no trials, so each imaging frame is the unit: classify the frame
        # by the running speed over its own interval, then average the trace within each
        # class. No padding, unlike the trial windows -- frames are contiguous, so a pad
        # would count the same running samples into both neighbours.
        t0, t1 = float(spont[0]), float(spont[1])
        in_spont = (rts >= t0) & (rts <= t1)
        out["spont_run_frac"] = np.full(
            n_rois, float((speed[in_spont] > thr).mean()) if in_spont.any() else np.nan)

        ts = np.asarray(plane.timestamps, dtype=np.float64)
        frames = np.flatnonzero((ts >= t0) & (ts <= t1))
        if frames.size:
            cs, counts = tr.prefix_sums(speed[:, None])
            a = np.searchsorted(rts, ts[frames], side="left")
            b = np.searchsorted(rts, ts[frames] + plane.dt, side="right")
            per_frame = tr.window_means(cs, counts, a, b)[:, 0]
            run_f = per_frame > thr
            stat_f = per_frame < thr
            # Same trace the gratings use, so every number in this table is in one
            # currency: mean event magnitude per imaging sample.
            traces = plane.traces.get(config.trace_type["drifting_gratings_full"])
            if traces is not None:
                block = np.asarray(traces)[frames]           # (n_spont_frames, n_rois)
                # Gated separately, on purpose. A session where the animal never ran still
                # has a baseline rate and a stationary rate; requiring both states would
                # throw those away, and 13 of 25 sessions here are one-sided.
                out["spont_rate"] = _nanmean(block, axis=0)
                if run_f.sum() >= n_min:
                    out["spont_rate_run"] = _nanmean(block[run_f], axis=0)
                if stat_f.sum() >= n_min:
                    out["spont_rate_stat"] = _nanmean(block[stat_f], axis=0)
                if run_f.sum() >= n_min and stat_f.sum() >= n_min:
                    out["run_mod_spont"] = _metric_index(out["spont_rate_run"],
                                                         out["spont_rate_stat"])

    frame = roi_frame(plane, mouse=mouse)
    for k, v in out.items():
        frame[k] = v
    return frame


# --------------------------------------------------------------- natural images


def natural_images_metrics(
    plane,
    trials: pd.DataFrame,
    spont: Sequence[float],
    *,
    ns_type: str = "natural_images",
    config: MetricConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
    mouse: Optional[str] = None,
) -> pd.DataFrame:
    """Natural-image metrics: one row per ROI.

    Structurally the same as `natural_movie_metrics` — group trials by condition, take
    the condition means, find each neuron's preferred one — with two differences that
    matter:

    * **Conditions are `image_index`, not `image_order`.** `image_order` is the raw
      presentation slot; `image_index` is the image's identity in the 118-image catalog.
      `natural_images_12` draws twelve images from that *same* namespace, so its
      `pref_img` values are a sparse subset of 0..117 (2, 4, 5, ..., 68) rather than
      0..11. Re-ranking them to 0..11 would look tidier and be wrong.
    * **`frac_responsive_trials` is a statistical test here**, unlike natural movie's
      `mean(response > 0)`: the fraction of preferred-image trials whose response beats a
      bootstrapped spontaneous null at p < 0.05. So this column carries bootstrap noise
      and should be read against a seed control, not against zero.

    The response window is `config.ni_response_seconds`. The original took it from an NWB
    `duration_sec` attribute that the current files no longer carry, so it is a recovered
    parameter rather than a known one — see the window probe in the notebook.
    """
    if not len(trials):
        return absent_frame(plane, ns_type, mouse)
    rng = np.random.default_rng() if rng is None else rng
    traces = plane.traces[config.trace_type[ns_type]]
    window = (0.0, float(config.ni_response_seconds))

    starts = trials["start_time"].to_numpy(dtype=np.float64)
    img = trials["image_index"].to_numpy()
    if np.isnan(img).any():
        raise ValueError(f"{ns_type}: trials contain NaN image_index")
    img = img.astype(int)

    image_ids = np.unique(img)
    code = _condition_codes(img, image_ids)
    n_trials = int(np.bincount(code).max())

    if config.ni_response_frames is not None:
        sweeps = tr.sweep_responses_frames(traces, plane.timestamps, starts,
                                           int(config.ni_response_frames))
    else:
        sweeps = tr.sweep_responses(traces, plane.timestamps, starts, window, None)
    ta = tr.trial_array(sweeps, code, n_trials=n_trials, n_conditions=len(image_ids))

    mean_resp = _nanmean(ta, axis=1)                        # (n_images, n_rois)
    n_rois = plane.n_rois
    roi_ix = np.arange(n_rois)

    all_nan = np.all(~np.isfinite(mean_resp), axis=0)
    pref_idx = np.where(np.isfinite(mean_resp), mean_resp, -np.inf).argmax(axis=0)
    pref_response = np.where(all_nan, np.nan, mean_resp[pref_idx, roi_ix])
    pref_img = np.where(all_nan, -1, image_ids[pref_idx])

    null_single = tr.spontaneous_null(
        traces, plane.timestamps, spont[0], spont[1], window, None,
        n_boot=config.other_n_boot, n_means=1, rng=rng,
        memory_budget_mb=config.memory_budget_mb,
    )
    pref_trials = ta[pref_idx, :, roi_ix]                   # (n_rois, n_trials)
    frac = tr.frac_trials_above_null(pref_trials, null_single, p_thresh=config.sig_p_thresh)

    # The multi-trial null averages n_trials draws per bootstrap sample. For
    # natural_images_12 that is 10,000 x 40 = 400,000 window means -- the heaviest single
    # call in the pipeline, and the reason spontaneous_null blocks by memory budget.
    null_multi = tr.spontaneous_null(
        traces, plane.timestamps, spont[0], spont[1], window, None,
        n_boot=config.other_n_boot, n_means=n_trials, rng=rng,
        memory_budget_mb=config.memory_budget_mb,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        z_score = (pref_response - null_multi.mean(axis=1)) / null_multi.std(axis=1)

    out = roi_frame(plane, mouse=mouse)
    out["frac_responsive_trials"] = frac
    out["lifetime_sparseness"] = _lifetime_sparseness_chunked(ta)
    out["reliability"] = tr.trial_reliability(ta)
    out["reliability_dff"] = _reliability_on(
        plane, "dff", starts, code, n_trials=n_trials, n_conditions=len(image_ids),
        window=window, frames=config.ni_response_frames)
    out["pref_img"] = pref_img
    out["pref_response"] = pref_response
    out["z_score"] = z_score
    return out


# --------------------------------------------------------------- receptive fields


def _rf_pixel_to_degrees(mean_idx, centers: np.ndarray, scale_bug: bool) -> np.ndarray:
    """Map a fractional pixel index to degrees of visual angle.

    Two mappings, because the original's is wrong and the published table carries the
    wrong one. `point_to_alt_azi` divides the centre-to-centre *range*
    (`centers[-1] - centers[0]`, which spans `n - 1` pixel pitches) by `len(centers)`,
    so its effective pitch is `(n-1)/n` of the real one. With 8 altitude rows that
    compresses the map by 12.5 %, and with 14 azimuth columns by 7.1 %: a centroid on the
    last pixel comes out at 28.48 deg instead of 32.55, and 56.13 instead of 60.45.

    Reproducing the published numbers means reproducing that, so it is the default.
    `scale_bug=False` gives the correct mapping, which is simply interpolation into the
    real pixel centres.
    """
    mean_idx = np.asarray(mean_idx, dtype=np.float64)
    if scale_bug:
        pitch = (centers[-1] - centers[0]) / len(centers)      # should be len - 1
        return (mean_idx + 0.5) * pitch + centers[0]
    pitch = (centers[-1] - centers[0]) / (len(centers) - 1)
    return mean_idx * pitch + centers[0]


def receptive_field_metrics(
    plane,
    trials: pd.DataFrame,
    spont: Sequence[float],
    lsn: Mapping[str, Any],
    *,
    config: MetricConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
    mouse: Optional[str] = None,
) -> pd.DataFrame:
    """Receptive fields from the locally-sparse-noise stimulus.

    The odd one out in three ways, all of which the original does deliberately:

    * **dF/F, not deconvolved events**, and the only family with a *subtracted* baseline
      (the 1 s before onset). Every other family uses events with no baseline at all.
    * **No trial array.** Instead a design matrix records which pixels were bright and
      which dark on each sweep, and the map is the fraction of a pixel's presentations
      that produced a significant response.
    * **No GLM.** The published README describes "a GLM framework"; there is no
      regression anywhere in `locally_sparse_noise.py`. It builds the design matrix and
      then uses it purely as a counting indicator. Do not go looking for the model.

    Significance is per-ROI: a sweep counts if its response exceeds the 95th percentile
    of that ROI's bootstrapped spontaneous responses. Pixel fractions below
    `rf_frac_thresh` are zeroed, so "has a receptive field" reduces to "at least one
    pixel survived", and the centre is the **unweighted** centroid of the surviving pixel
    indices — the fractions are not used as weights.

    `lsn` is the dict from `v1dd_nwb.load_lsn_template`. Its `pixel_on` / `pixel_off` are
    read from the template rather than hard-coded: this asset encodes the stimulus as
    -1 / 0 / 1 where the original assumed 0 / 127 / 255, and hard-coding those would make
    both design matrices all-False and report zero receptive fields for every ROI.
    """
    images = np.asarray(lsn["images"])
    n_rows, n_cols = images.shape[1], images.shape[2]

    if not len(trials):
        empty_map = np.zeros((plane.n_rois, 2, n_rows, n_cols), dtype=np.float32)
        return absent_frame(plane, "rf_metrics", mouse), empty_map
    rng = np.random.default_rng() if rng is None else rng
    traces = plane.traces[config.trace_type["locally_sparse_noise"]]
    window = (0.0, config.lsn_response_frames * plane.dt)
    baseline = (-1.0, 0.0)
    pixel_on, pixel_off = lsn.get("pixel_on"), lsn.get("pixel_off")
    if pixel_on is None or pixel_off is None:
        raise ValueError(
            f"could not determine ON/OFF pixel codes from the template "
            f"(values seen: {lsn.get('pixel_values')})"
        )

    starts = trials["start_time"].to_numpy(dtype=np.float64)
    frames = trials["frame"].to_numpy()
    if np.isnan(frames).any():
        raise ValueError("locally_sparse_noise trials contain NaN frame indices")
    frames = frames.astype(int)
    if frames.max() >= len(images):
        raise ValueError(
            f"frame index {frames.max()} exceeds the {len(images)}-frame template"
        )

    n_pixels = n_rows * n_cols

    # (2 * n_pixels, n_sweeps): rows 0..n_pixels-1 are ON, the rest OFF. Gray is neither.
    stim_pixels = images[frames].reshape(len(frames), n_pixels)
    design = np.concatenate([stim_pixels == pixel_on, stim_pixels == pixel_off], axis=1).T

    sweeps = tr.sweep_responses(traces, plane.timestamps, starts, window, baseline)

    null = tr.spontaneous_null(
        traces, plane.timestamps, spont[0], spont[1], window, baseline,
        n_boot=config.other_n_boot, n_means=1, rng=rng,
        memory_budget_mb=config.memory_budget_mb,
    )
    threshold = np.quantile(null, 0.95, axis=1)                 # (n_rois,)
    significant = sweeps > threshold[None, :]                   # (n_sweeps, n_rois)

    n_pixel_trials = design.sum(axis=1)                         # (2 * n_pixels,)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (design.astype(np.int64) @ significant).T / np.where(
            n_pixel_trials > 0, n_pixel_trials, np.nan)
    frac = np.nan_to_num(frac, nan=0.0)
    frac[~plane.is_valid] = 0.0   # blank = excluded (not "no RF"), documented in rf_map
    # continuous pre-threshold map: (n_rois, 2, n_rows, n_cols) float32.
    # Graded values before zeroing sub-threshold pixels; recoverable to post-threshold
    # in one line, but the reverse is not. Saved alongside the per-ROI metrics.
    rf_map = frac.reshape(plane.n_rois, 2, n_rows, n_cols).astype(np.float32).copy()
    frac[frac < config.rf_frac_thresh] = 0.0
    rf = frac.reshape(plane.n_rois, 2, n_rows, n_cols)          # dim 1: 0 = ON, 1 = OFF

    mask = rf > 0
    counts = mask.sum(axis=(2, 3))                              # (n_rois, 2)
    rows_ix = np.arange(n_rows)[None, None, :, None]
    cols_ix = np.arange(n_cols)[None, None, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_row = (mask * rows_ix).sum(axis=(2, 3)) / counts
        mean_col = (mask * cols_ix).sum(axis=(2, 3)) / counts

    altitudes = np.asarray(lsn["altitudes"], dtype=np.float64)
    azimuths = np.asarray(lsn["azimuths"], dtype=np.float64)
    alt = _rf_pixel_to_degrees(mean_row, altitudes, config.rf_center_scale_bug)
    azi = _rf_pixel_to_degrees(mean_col, azimuths, config.rf_center_scale_bug)

    has_on = counts[:, 0] > 0
    has_off = counts[:, 1] > 0

    out = roi_frame(plane, mouse=mouse)
    out["has_rf_on"] = has_on
    out["has_rf_off"] = has_off
    out["has_rf_on_or_off"] = has_on | has_off
    out["azimuth_rf_on"] = np.where(has_on, azi[:, 0], np.nan)
    out["altitude_rf_on"] = np.where(has_on, alt[:, 0], np.nan)
    out["azimuth_rf_off"] = np.where(has_off, azi[:, 1], np.nan)
    out["altitude_rf_off"] = np.where(has_off, alt[:, 1], np.nan)
    return out, rf_map


# ------------------------------------------------- receptive field vs. grating window


CONTAINMENT_COLUMNS = ["dgw_rf_distance_on", "dgw_rf_distance_off",
                       "dgw_rf_overlap_on", "dgw_rf_overlap_off"]


def _window_coverage(azimuths, altitudes, center, radius: float, sub: int = 8):
    """Fraction of each stimulus pixel's **area** inside the aperture disc.

    Depends only on the window position, so one array serves every ROI in a session.

    A pixel-centre-inside-the-disc test is not good enough here and the numbers say why:
    the pixels are 9.3 degrees, the aperture is 30, so the disc spans about 3.2 pixels
    while covering roughly 8 pixels' worth of area. Whether a centre test counts 5 pixels
    or 9 then depends on how the disc happens to land on the grid — a swing of about
    40 %. Sub-sampling each pixel on a `sub` x `sub` grid removes that: at sub=8 the
    recovered area is within ~1 % of pi*r^2, which the unit tests assert.
    """
    az = np.asarray(azimuths, dtype=np.float64)
    alt = np.asarray(altitudes, dtype=np.float64)
    caz, cel = float(center[0]), float(center[1])
    if not (np.isfinite(caz) and np.isfinite(cel)):
        return None

    # pitch is read off the grid, never assumed: this asset is 9.3 degrees, but a
    # different locally-sparse-noise template would silently produce wrong areas.
    def pitch(v, name):
        d = np.diff(v)
        if len(d) and not np.allclose(d, d[0]):
            raise ValueError(f"{name} are not evenly spaced: {np.unique(np.round(d, 6))}")
        return float(abs(d[0])) if len(d) else 0.0

    p_az, p_alt = pitch(az, "azimuths"), pitch(alt, "altitudes")
    offs = (np.arange(sub) + 0.5) / sub - 0.5                    # sub-cell centres
    d_az = (az[:, None] + offs[None, :] * p_az) - caz            # (n_cols, sub)
    d_alt = (alt[:, None] + offs[None, :] * p_alt) - cel         # (n_rows, sub)
    inside = ((d_alt ** 2)[:, :, None, None] + (d_az ** 2)[None, None, :, :]
              <= radius * radius)                                # (rows, sub, cols, sub)
    return inside.mean(axis=(1, 3))                              # (n_rows, n_cols)


def window_containment(
    rf_frame: pd.DataFrame,
    rf_map: np.ndarray,
    lsn: Mapping[str, Any],
    center: Sequence[float],
    *,
    config: MetricConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """How much of each ROI's receptive field the windowed grating actually covered.

    `ssi` compares a windowed grating response against a full-field one, which only means
    "surround suppression" if the window covered the cell's receptive field. A cell whose
    RF sat outside the aperture was barely stimulated, and its weak windowed response
    reads as suppression when it was a targeting miss.

    Two measures, deliberately, because they disagree about which cells to keep:

    * **`dgw_rf_distance_*`** — degrees from the RF centre to the window centre. The
      conservative reading, and the one the white paper used. It is also blunt: the
      centre is an unweighted centroid on a 9.3-degree grid, so one marginal pixel moves
      it ~4.6 degrees.
    * **`dgw_rf_overlap_*`** — the fraction of the RF's mass falling inside the aperture,
      in [0, 1]. More permissive and better behaved: it keeps cells whose field overlaps
      the window even though the centroid does not, which on this asset is 1,572 cells
      against 970 at a 0.05 cut.

    The overlap is weighted by the **post-threshold** map. The continuous pre-threshold
    map is dominated by noise floor — its mean overlap is 0.086 against the 0.073 a
    uniform random map would give, i.e. it mostly measures the window's share of the
    screen rather than anything about the cell.

    **Neither is a filter.** On this asset overlap correlates with `ssi` at r = +0.07
    (n = 6,827) with a non-monotonic profile, so the targeting concern is directionally
    supported but weak. These columns are reported so a consumer can judge; gating on
    them would discard most of the data on thin evidence.

    Returns a frame of `CONTAINMENT_COLUMNS`, all NaN where the window position is
    unknown (two sessions in this asset record none) or the ROI has no field.
    """
    n_rois = len(rf_frame)
    out = {c: np.full(n_rois, np.nan) for c in CONTAINMENT_COLUMNS}
    cov = _window_coverage(lsn["azimuths"], lsn["altitudes"], center,
                           config.dgw_window_radius_deg)
    if cov is None:                       # no recorded aperture -> everything stays NaN
        return pd.DataFrame(out, index=rf_frame.index)

    caz, cel = float(center[0]), float(center[1])
    rf_map = np.asarray(rf_map)
    for i, sub in enumerate(("on", "off")):
        out[f"dgw_rf_distance_{sub}"] = np.hypot(
            rf_frame[f"azimuth_rf_{sub}"].to_numpy(dtype=np.float64) - caz,
            rf_frame[f"altitude_rf_{sub}"].to_numpy(dtype=np.float64) - cel)
        w = np.asarray(rf_map[:, i, :, :], dtype=np.float64).copy()
        w[w < config.rf_frac_thresh] = 0.0
        den = w.sum(axis=(1, 2))
        num = (w * cov[None, :, :]).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"dgw_rf_overlap_{sub}"] = np.where(den > 0, num / np.where(den > 0, den, 1.0),
                                                    np.nan)
    return pd.DataFrame(out, index=rf_frame.index)
