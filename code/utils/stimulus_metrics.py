"""Stimulus-response metrics, ported from `allen_v1dd` onto NWB-Zarr.

One family per function, each returning a tidy per-ROI DataFrame with the same column
names the published `data_frames/*_M409828.csv` tables use, so a regenerated table drops
in where the old one was.

**These reproduce the original, bugs included.** Where the original does something
defensible-but-wrong, the behaviour is kept and named in a `MetricConfig` flag, so the
published numbers stay reachable and the corrected version is one argument away. The
alternative — quietly fixing things — would make every disagreement with the published
table ambiguous between "we fixed a bug" and "we introduced one", which destroys the only
validation signal available.

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
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import trial_responses as tr

__all__ = [
    "MetricConfig",
    "DEFAULT_CONFIG",
    "COLUMN_ORDER",
    "natural_movie_metrics",
    "roi_frame",
    "to_published_schema",
    "compare_to_published",
    "load_published",
]


@dataclass(frozen=True)
class MetricConfig:
    """Knobs, defaulted to reproduce the published tables.

    The `*_bug` flags exist so that "match the published numbers" and "compute the right
    thing" are both reachable, and which one you asked for is written down.
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
    #: Natural-images response window, seconds (the original's `duration_sec`).
    ni_response_seconds: Any = 0.30
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
    ssi_min_trials: int = 3

    # --- expensive extras, absent from every published table
    permutation_test_shuffles: int = 0
    chisq_shuffles: int = 0
    fit_tuning_curves: bool = True

    # --- bug compatibility (True == reproduce the published numbers)
    rf_center_scale_bug: bool = True
    pref_cond_fillna: bool = True

    memory_budget_mb: float = 64.0


DEFAULT_CONFIG = MetricConfig()


#: Published column names and order, per output file. `surround_supression_index` keeps
#: the misspelling from the published filename.
COLUMN_ORDER: Dict[str, Sequence[str]] = {
    "drifting_gratings_full": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi",
        "dsi", "frac_responsive_trials", "gosi", "is_responsive",
        "lifetime_sparseness", "osi", "preferred_dir", "preferred_sf", "pref_dir_mean"],
    "natural_images": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi",
        "frac_responsive_trials", "lifetime_sparseness", "pref_img", "pref_response",
        "z_score"],
    "surround_supression_index": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi",
        "ssi", "ssi_avg", "ssi_avg_at_pref_sf", "ssi_running",
        "ssi_running_avg_at_pref_sf", "ssi_stationary",
        "ssi_stationary_avg_at_pref_sf", "ssi_tuning_fit"],
    "rf_metrics": [
        "roi_unique_id", "mouse", "column", "volume", "plane", "roi",
        "has_rf_on", "has_rf_off", "has_rf_on_or_off",
        "azimuth_rf_on", "altitude_rf_on", "azimuth_rf_off", "altitude_rf_off"],
}
COLUMN_ORDER["drifting_gratings_windowed"] = COLUMN_ORDER["drifting_gratings_full"]
COLUMN_ORDER["natural_images_12"] = COLUMN_ORDER["natural_images"]
COLUMN_ORDER["natural_movie"] = COLUMN_ORDER["natural_images"]


# --------------------------------------------------------------------- identity


def roi_frame(plane, mouse: str = "M409828") -> pd.DataFrame:
    """The six identity columns every published table starts with.

    `roi_unique_id` reproduces the published format `M{mouse}_{volume}_{plane}_{roi}`,
    which **omits the column** and therefore collides across the five columns — 164,345
    published rows share only 56,449 distinct ids. It is emitted for drop-in
    compatibility; `roi_key` is the non-colliding version. **Join on
    `(column, volume, plane, roi)`, never on either string.**
    """
    n = plane.n_rois
    mouse_num = mouse.lstrip("M")
    return pd.DataFrame({
        "roi_unique_id": [f"M{mouse_num}_{plane.volume}_{plane.plane}_{r}" for r in plane.roi],
        "roi_key": [
            f"M{mouse_num}_{plane.column}{plane.volume}_{plane.plane}_{r}" for r in plane.roi
        ],
        "mouse": [mouse] * n,
        "column": np.full(n, plane.column, dtype=int),
        "volume": [plane.volume] * n,
        "plane": np.full(n, plane.plane, dtype=int),
        "roi": plane.roi.astype(int),
    })


def to_published_schema(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Reorder and dtype a metrics frame to match the published CSV exactly."""
    cols = list(COLUMN_ORDER[family])
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
    mouse: str = "M409828",
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
    out["pref_img"] = pref_img
    out["pref_response"] = pref_response
    out["z_score"] = z_score
    return out


# --------------------------------------------------------------------- validation


def load_published(published_dir: str, family: str, mouse: str = "M409828") -> pd.DataFrame:
    """Read a published metrics CSV with the dtypes it actually needs.

    `volume` must be read as a string: volumes run 1..9 and a..f, so `int` crashes on the
    3p sessions. Pandas also warns about mixed types in that column without an explicit
    dtype.
    """
    import os

    path = os.path.join(published_dir, f"{family}_{mouse}.csv")
    df = pd.read_csv(path, dtype={"volume": str})
    return df.astype({"column": int, "plane": int, "roi": int})


def compare_to_published(
    new: pd.DataFrame,
    published: pd.DataFrame,
    metrics: Sequence[str],
    keys: Sequence[str] = ("column", "volume", "plane", "roi"),
    exact: Sequence[str] = (),
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Per-metric agreement between a regenerated table and the published one.

    Joins on `(column, volume, plane, roi)`, never on `roi_unique_id`, which omits the
    column and collides. Reports set differences explicitly, because the NWB asset is
    ROI-filtered while the published tables carry every segmented ROI, so an inner join
    is expected to drop rows on the published side.

    `exact` names categorical columns (preferred direction, preferred image) compared by
    equality rather than correlation.
    """
    keys = list(keys)
    left = new.copy()
    right = published.copy()
    for frame in (left, right):
        frame["volume"] = frame["volume"].astype(str)
        for k in ("column", "plane", "roi"):
            frame[k] = frame[k].astype(int)

    merged = left.merge(right, on=keys, how="inner", suffixes=("_new", "_pub"))
    only_new = len(left) - len(merged)
    only_pub = len(right) - len(merged)

    report: Dict[str, Any] = {
        "n_new": int(len(left)),
        "n_published": int(len(right)),
        "n_joined": int(len(merged)),
        "n_only_new": int(only_new),
        "n_only_published": int(only_pub),
        "metrics": {},
    }

    for m in metrics:
        a = pd.to_numeric(merged.get(f"{m}_new"), errors="coerce")
        b = pd.to_numeric(merged.get(f"{m}_pub"), errors="coerce")
        if a is None or b is None:
            report["metrics"][m] = {"error": "column missing on one side"}
            continue
        both = a.notna() & b.notna()
        entry: Dict[str, Any] = {
            "n_both_finite": int(both.sum()),
            "n_new_only_finite": int((a.notna() & ~b.notna()).sum()),
            "n_pub_only_finite": int((~a.notna() & b.notna()).sum()),
        }
        if both.sum():
            av, bv = a[both].to_numpy(), b[both].to_numpy()
            diff = np.abs(av - bv)
            entry.update({
                "max_abs_diff": float(diff.max()),
                "median_abs_diff": float(np.median(diff)),
                "p95_abs_diff": float(np.percentile(diff, 95)),
                "frac_within_tol": float(np.mean(diff <= tol)),
                "tol": tol,
            })
            if m in exact:
                entry["frac_exact"] = float(np.mean(av == bv))
            if both.sum() > 2 and np.std(av) > 0 and np.std(bv) > 0:
                entry["pearson_r"] = float(np.corrcoef(av, bv)[0, 1])
        report["metrics"][m] = entry

    return report
