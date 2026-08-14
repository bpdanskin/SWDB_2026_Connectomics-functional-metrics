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
    "natural_images_metrics",
    "drifting_gratings_metrics",
    "surround_suppression_metrics",
    "DGResult",
    "SSI_COLUMNS",
    "vonmises_two_peak",
    "vonmises_two_peak_fit",
    "vonmises_pref_dir",
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
    #: Note the margin is only 8e-5 s. If dt ever changes, re-run the probe -- or set
    #: `ni_response_frames=2`, which expresses the same intent and cannot drift.
    ni_response_seconds: Any = 0.33
    #: If set, natural images uses a FIXED number of imaging samples from each onset
    #: instead of a time window, and `ni_response_seconds` is ignored. The two differ in
    #: how many samples land in a trial: a time window varies with onset phase, a frame
    #: count does not. Because per-trial rescaling changes the relative pattern across
    #: images, scale-invariant metrics like lifetime_sparseness can distinguish them.
    ni_response_frames: Optional[int] = None
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
                # Relative, not bitwise. `preferred_sf` carries a float32 round-trip
                # (0.04 arrives as 0.039999999...), so two values agreeing to 16 digits
                # can still fail `==` and report 0% agreement on a column that is in
                # fact identical. A genuine category change is a 100% difference, so
                # nothing real hides under this tolerance.
                entry["frac_exact"] = float(np.mean(np.isclose(av, bv, rtol=1e-9, atol=0)))
                entry["frac_exact_bitwise"] = float(np.mean(av == bv))
            if both.sum() > 2 and np.std(av) > 0 and np.std(bv) > 0:
                entry["pearson_r"] = float(np.corrcoef(av, bv)[0, 1])
        report["metrics"][m] = entry

    return report


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
    config: MetricConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
    mouse: str = "M409828",
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

    # published preferred condition: argmax over fillna(-1), C order
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

    pref_cond_index = np.stack([pref_dir_fill, pref_sf_fill], axis=1).astype(int)
    pref_cond_index[~plane.is_valid] = -1

    tuning = mean_tr[roi_ix, :, pref_sf_idx]             # (n_rois, n_dir) at preferred SF
    pref = tuning[roi_ix, pref_dir_idx]
    null_r = tuning[roi_ix, (pref_dir_idx + 6) % 12]
    orth_r = 0.5 * (tuning[roi_ix, (pref_dir_idx + 3) % 12]
                    + tuning[roi_ix, (pref_dir_idx - 3) % 12])

    osi = _ratio(pref - orth_r, pref + orth_r)
    dsi = _ratio(pref - null_r, pref + null_r)

    theta = np.deg2rad(dir_list.astype(float))
    # gosi normalises with a NaN-PROPAGATING sum, matching `_compute_osi`
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

    # von Mises fits, one per (ROI, spatial frequency)
    tuning_params = np.full((n_rois, n_sf, 6), np.nan)
    if config.fit_tuning_curves:
        for roi in range(n_rois):
            if not plane.is_valid[roi]:
                continue
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
                    dir_list=dir_list, sf_list=sf_list, blank_responses=blank)


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
    mouse: str = "M409828",
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
    mouse: str = "M409828",
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
    out["pref_img"] = pref_img
    out["pref_response"] = pref_response
    out["z_score"] = z_score
    return out
