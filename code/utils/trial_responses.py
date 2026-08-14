"""Stimulus-locked responses, trial arrays, and bootstrap nulls — the arithmetic layer.

Every stimulus metric in `stimulus_metrics.py` reduces to one question: *what was this
neuron's mean activity in a time window after a stimulus onset?* Asked a few million
times. The `allen_v1dd` original asked it with a Python loop —

    for i in stim_table.index:
        response = traces.sel(time=slice(start + w0, start + w1)).mean("time")

— which is ~0.3-1 ms per call, so a natural-movie plane (29,724 sweeps) costs 10-30 s and
a 10,000-draw bootstrap costs another 15 s, per family, per plane. The whole port would
run 40-50 minutes.

The fix is a **prefix sum over time**. Once you have `cs[k] = traces[:k].sum(axis=0)`,
the mean over any window is `(cs[b] - cs[a]) / (b - a)`: two gathers and a divide, with
no dependence on how many samples the window contains. The thing that looks like it
forces a loop — response windows landing on different numbers of imaging frames,
because stimulus onsets are not frame-aligned — turns out to be free, since the samples
are never materialised. `b - a` is just an integer vector.

That single change is ~50x on the dominant term and takes the port to ~5 minutes.

Two subtleties are load-bearing, and both are about *matching the original exactly*
rather than about speed:

1. **Trial windows are label-closed; bootstrap windows are frame-indexed.** The original
   selected trial responses with `xarray.sel(time=slice(...))`, which includes both
   endpoints and so spans a variable, data-dependent number of samples. But its bootstrap
   used a different primitive, `traces[:, f + r0 : f + r1]`, a half-open slice of *fixed*
   width `round(w / dt)`. At dt = 0.164 s a 2 s drifting-grating window gives 13 samples
   for a trial and 12 for a bootstrap draw. Reproducing the published numbers means
   reproducing both primitives, so `window_bounds` implements the first and
   `spontaneous_null` the second.

2. **Trial arrays are NaN-padded.** Conditions are not all presented the same number of
   times, so the trial axis is sized to the maximum and short conditions keep NaN in the
   tail. Every reduction here is nan-aware, and `lifetime_sparseness` drops NaNs *before*
   counting, so its normaliser varies per neuron.

Nothing in this module imports pandas, hdmf_zarr, or xarray — it is arrays in, arrays
out, so it can be tested against a synthetic trace with no data mounted.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "prefix_sums",
    "window_bounds",
    "window_means",
    "sweep_responses",
    "sweep_responses_frames",
    "spontaneous_null",
    "trial_array",
    "frac_trials_above_null",
    "lifetime_sparseness",
    "si_permutation_test",
]


# ------------------------------------------------------------------ the primitive


def prefix_sums(traces: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Cumulative sums over time, for constant-cost window means.

    Parameters
    ----------
    traces:
        (n_frames, n_rois). NWB-Zarr's native orientation — do not transpose first.

    Returns
    -------
    cs:
        (n_frames + 1, n_rois) float64, with `cs[k] == traces[:k].sum(axis=0)`, so the
        mean over `traces[a:b]` is `(cs[b] - cs[a]) / (b - a)`. float64 regardless of
        input dtype: a float32 cumsum over 20k frames loses precision that shows up in
        the sixth decimal of a response.
    counts:
        (n_frames + 1, n_rois) cumulative count of finite samples, or None when the
        traces contain no NaN. Present only when needed, because carrying it doubles
        the memory and most planes don't need it.
    """
    traces = np.asarray(traces)
    if traces.ndim != 2:
        raise ValueError(f"expected (n_frames, n_rois), got shape {traces.shape}")

    n_frames, n_rois = traces.shape
    finite = np.isfinite(traces)
    counts = None
    if not finite.all():
        counts = np.zeros((n_frames + 1, n_rois), dtype=np.int64)
        np.cumsum(finite, axis=0, out=counts[1:])
        traces = np.where(finite, traces, 0.0)

    cs = np.zeros((n_frames + 1, n_rois), dtype=np.float64)
    np.cumsum(traces, axis=0, dtype=np.float64, out=cs[1:])
    return cs, counts


def window_bounds(
    timestamps: np.ndarray, starts: np.ndarray, t0: float, t1: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Half-open index bounds for the label-CLOSED interval [start + t0, start + t1].

    This is what `xarray.DataArray.sel(time=slice(lo, hi))` does on a monotonic index:
    `searchsorted(lo, "left")` and `searchsorted(hi, "right")`, i.e. **both endpoints
    included**. A naive `(t >= lo) & (t < hi)` mask drops one sample per trial and shifts
    every response slightly — which is exactly the kind of difference that survives into
    a metric and looks like an algorithm bug.

    The returned windows have variable width. That is intended, and costs nothing.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    a = np.searchsorted(timestamps, starts + t0, side="left")
    b = np.searchsorted(timestamps, starts + t1, side="right")
    return a, b


def window_means(
    cs: np.ndarray, counts: Optional[np.ndarray], a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Mean over `traces[a[i]:b[i]]` for each window i. Returns (n_windows, n_rois).

    An empty window (b <= a) yields NaN via 0/0, matching `xarray.mean` of an empty
    selection. That happens at session edges and must not raise.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    total = cs[b] - cs[a]
    n = (b - a)[:, None].astype(np.float64) if counts is None else (counts[b] - counts[a])
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / n


# ------------------------------------------------------------------ sweep responses


def sweep_responses(
    traces: np.ndarray,
    timestamps: np.ndarray,
    starts: np.ndarray,
    response_window: Tuple[float, float],
    baseline_window: Optional[Tuple[float, float]] = None,
    block: int = 8192,
) -> np.ndarray:
    """Mean response per stimulus sweep. Returns (n_sweeps, n_rois).

    Replaces the original's per-sweep `get_responses(start, baseline, response)` loop.

    `baseline_window=None` means *no baseline subtraction at all* — that is the shipped
    configuration for every events-based family, and it is not the same as subtracting
    zero, because it also skips a second pair of gathers.

    `block` bounds the (block, n_rois) intermediate; the default keeps it under ~30 MB
    at 450 ROIs.
    """
    cs, counts = prefix_sums(traces)
    starts = np.asarray(starts, dtype=np.float64)
    out = np.empty((starts.size, traces.shape[1]), dtype=np.float64)

    for s in range(0, starts.size, block):
        sl = slice(s, min(s + block, starts.size))
        a, b = window_bounds(timestamps, starts[sl], *response_window)
        r = window_means(cs, counts, a, b)
        if baseline_window is not None:
            a0, b0 = window_bounds(timestamps, starts[sl], *baseline_window)
            r = r - window_means(cs, counts, a0, b0)
        out[sl] = r

    return out


# ------------------------------------------------------------------ bootstrap null


def sweep_responses_frames(
    traces: np.ndarray,
    timestamps: np.ndarray,
    starts: np.ndarray,
    n_frames: int,
    offset_frames: int = 0,
) -> np.ndarray:
    """Mean of exactly `n_frames` samples from each onset. Returns (n_sweeps, n_rois).

    The *fixed-width* alternative to `sweep_responses`. Where that one takes every sample
    inside a time window — so the count varies from trial to trial with where the onset
    falls between samples — this takes the same number of samples every time, starting at
    the first sample at or after the onset.

    The distinction matters because the two disagree in a way no choice of window length
    can reconcile. A varying sample count rescales each trial differently, which changes
    the *relative* pattern of responses across conditions; a fixed count does not. Metrics
    that are scale-invariant per trial (`lifetime_sparseness`) can therefore tell the two
    apart even though the mean response looks similar.

    `allen_v1dd` used the varying-width form for trial responses and the fixed-width form
    for its bootstrap null, so both are needed to reproduce it.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    cs, counts = prefix_sums(traces)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)

    a = np.searchsorted(timestamps, starts, side="left") + offset_frames
    a = np.clip(a, 0, len(timestamps))
    b = np.clip(a + n_frames, 0, len(timestamps))
    return window_means(cs, counts, a, b)


def _nearest_index(timestamps: np.ndarray, value: float) -> int:
    """`pandas.Index.get_loc(value, method="nearest")`, which pandas 2.0 removed.

    Ties break toward the earlier sample, matching pandas.
    """
    n = len(timestamps)
    i = int(np.clip(np.searchsorted(timestamps, value), 1, n - 1))
    return i - int(abs(value - timestamps[i - 1]) <= abs(timestamps[i] - value))


def spontaneous_null(
    traces: np.ndarray,
    timestamps: np.ndarray,
    spont_start: float,
    spont_stop: float,
    response_window: Tuple[float, float],
    baseline_window: Optional[Tuple[float, float]] = None,
    n_boot: int = 2500,
    n_means: int = 1,
    rng: Optional[np.random.Generator] = None,
    memory_budget_mb: float = 64.0,
) -> np.ndarray:
    """Bootstrap distribution of spontaneous responses. Returns (n_rois, n_boot).

    Bug-compatible with `StimulusAnalysis.get_spont_null_dist`: windows are **frame
    indexed** (half-open, fixed width `round(w / dt)`), onsets are drawn uniformly with
    replacement from a single spontaneous block, and `n_means` independent draws are
    averaged per bootstrap sample.

    `n_means=1` gives the single-trial null used for `frac_responsive_trials`;
    `n_means=n_trials` gives the multi-trial null used for `z_score`. The latter is the
    expensive one — natural-images-12 draws 10,000 x 40 = 400,000 windows — which is what
    `memory_budget_mb` is for. The loop below is over *memory blocks*, not bootstraps.
    """
    rng = np.random.default_rng() if rng is None else rng
    timestamps = np.asarray(timestamps, dtype=np.float64)
    dt = float(np.median(np.diff(timestamps)))

    r0, r1 = (int(round(response_window[0] / dt)), int(round(response_window[1] / dt)))
    if baseline_window is None:
        b0 = b1 = None
        start_pad = 0
    else:
        b0, b1 = (int(round(baseline_window[0] / dt)), int(round(baseline_window[1] / dt)))
        start_pad = -b0

    lo = _nearest_index(timestamps, spont_start) + start_pad
    hi = _nearest_index(timestamps, spont_stop) - r1  # exclusive, as np.random.randint
    if hi <= lo:
        raise ValueError(
            f"spontaneous block [{spont_start:.1f}, {spont_stop:.1f}] s is too short for "
            f"response window {response_window} at dt={dt:.4f} s"
        )

    cs, counts = prefix_sums(traces)
    n_rois = traces.shape[1]
    idx = rng.integers(lo, hi, size=(n_boot, n_means))

    per_row_bytes = max(n_means * n_rois * 8, 1)
    block = max(1, int(memory_budget_mb * 1e6 // per_row_bytes))

    out = np.empty((n_boot, n_rois), dtype=np.float64)
    for s in range(0, n_boot, block):
        j = idx[s : s + block]
        f = j.ravel()
        r = window_means(cs, counts, f + r0, f + r1)
        if b0 is not None:
            r = r - window_means(cs, counts, f + b0, f + b1)
        out[s : s + block] = r.reshape(j.shape[0], n_means, n_rois).mean(axis=1)

    return out.T


# ------------------------------------------------------------------ trial arrays


def trial_array(
    sweep_resp: np.ndarray,
    condition: np.ndarray,
    n_trials: Optional[int] = None,
    n_conditions: Optional[int] = None,
) -> np.ndarray:
    """Scatter per-sweep responses into (n_conditions, n_trials, n_rois), NaN-padded.

    `condition` is an integer code per sweep, in table order — which is chronological.
    A sweep's trial index is its chronological rank *within its condition*, reproducing
    `for trial, stim_i in enumerate(get_stim_idx(dir, sf))`. That alignment matters
    because the per-trial running speeds are indexed the same way, and the SSI code
    masks one array with the other.

    Conditions presented fewer than `n_trials` times keep NaN in the tail. Sweeps beyond
    `n_trials` for a condition are dropped, which cannot happen when `n_trials` is the
    observed maximum but can if a caller passes a smaller value deliberately.
    """
    sweep_resp = np.asarray(sweep_resp)
    condition = np.asarray(condition)
    if condition.size != sweep_resp.shape[0]:
        raise ValueError(
            f"condition has {condition.size} entries but sweep_resp has "
            f"{sweep_resp.shape[0]} sweeps"
        )
    if condition.size and condition.min() < 0:
        raise ValueError("condition codes must be non-negative")

    # stable sort keeps time order inside each condition group
    order = np.argsort(condition, kind="stable")
    codes = condition[order]

    if n_conditions is None:
        n_conditions = int(codes.max()) + 1 if codes.size else 0
    first = np.searchsorted(codes, np.arange(n_conditions), side="left")
    trial = np.arange(codes.size) - first[codes]

    if n_trials is None:
        n_trials = int(trial.max()) + 1 if trial.size else 0

    out = np.full((n_conditions, n_trials, sweep_resp.shape[1]), np.nan, dtype=np.float64)
    keep = trial < n_trials
    out[codes[keep], trial[keep]] = sweep_resp[order[keep]]
    return out


# ------------------------------------------------------------------ reductions


def frac_trials_above_null(
    trial_resp: np.ndarray,
    null_single: np.ndarray,
    p_thresh: float = 0.05,
    block_rois: int = 64,
) -> np.ndarray:
    """Fraction of a neuron's trials that beat its spontaneous null. Returns (n_rois,).

    Parameters
    ----------
    trial_resp:
        (n_rois, n_trials) responses at each neuron's preferred condition, NaN-padded.
    null_single:
        (n_rois, n_boot) single-trial spontaneous null.

    A trial counts as responsive when `mean(null > response) < p_thresh` — the original's
    one-sided test. Note this is *not* the same as `response > quantile(null, 0.95)`,
    which interpolates between order statistics and disagrees on a few percent of trials.

    NaN-padded trials are excluded rather than counted: without the explicit NaN guard,
    `null > nan` is False for every draw, giving p = 0 and scoring a non-existent trial
    as responsive.
    """
    trial_resp = np.asarray(trial_resp, dtype=np.float64)
    null_single = np.asarray(null_single, dtype=np.float64)
    n_rois = trial_resp.shape[0]
    out = np.full(n_rois, np.nan)

    for s in range(0, n_rois, block_rois):
        sl = slice(s, min(s + block_rois, n_rois))
        r = trial_resp[sl]                                    # (k, n_trials)
        p = (null_single[sl][:, None, :] > r[:, :, None]).mean(axis=2)
        sig = np.where(np.isnan(r), np.nan, (p < p_thresh).astype(np.float64))
        with np.errstate(invalid="ignore"):
            out[sl] = np.nanmean(sig, axis=1) if sig.shape[1] else np.nan

    return out


def lifetime_sparseness(x: np.ndarray) -> np.ndarray:
    """Olsen & Wilson (2008) lifetime sparseness. `x` is (n_rois, n_responses).

    Computed over **every individual trial response**, flattened across conditions and
    trials — not over the condition means. NaNs are dropped first, so the normaliser
    `1 - 1/n` uses each neuron's own count of real responses.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    n = finite.sum(axis=1).astype(np.float64)
    xs = np.where(finite, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = xs.sum(axis=1) / n
        mean_sq = (xs * xs).sum(axis=1) / n
        return (1.0 - mean**2 / mean_sq) / (1.0 - 1.0 / n)


def _vector_strength(tuning: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """|sum R_theta e^{i k theta}| / sum R_theta, matching `_selectivity_index`.

    A plain (NaN-propagating) sum in the denominator, deliberately: the original used
    `np.sum`, and switching to `np.nansum` here would silently change which neurons get
    a finite selectivity index.
    """
    norm = np.sum(tuning, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(tuning @ phase / np.where(norm == 0, np.nan, norm))


def si_permutation_test(
    trial_responses: np.ndarray,
    n_shuffles: int = 1000,
    rng: Optional[np.random.Generator] = None,
    block: int = 64,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Permutation test for orientation/direction selectivity.

    Parameters
    ----------
    trial_responses:
        (n_rois, n_directions, n_trials) at each neuron's preferred spatial frequency.

    Returns `{"osi": (si, p), "dsi": (si, p)}`.

    Replaces `n_shuffles` calls to `np.apply_along_axis(np.random.shuffle, ...)` —
    which is `n_shuffles * n_rois * n_trials` Python-level calls — with
    `rng.permuted(..., axis=2)`, which independently permutes every 1-D slice in C. Both
    metrics reuse one set of shuffled tunings.

    Note this selectivity index is a circular vector strength on *index-derived* angles,
    a different quantity from the ratio-form `osi`/`dsi` in the published tables. None of
    the published columns need it; it is here because it is now cheap enough to be worth
    having.
    """
    rng = np.random.default_rng() if rng is None else rng
    trial_responses = np.asarray(trial_responses, dtype=np.float64)
    n_rois, n_dir, n_trial = trial_responses.shape

    ang = np.arange(n_dir) / n_dir * 2 * np.pi
    phase = {"dsi": np.exp(1j * ang), "osi": np.exp(2j * ang)}

    with np.errstate(invalid="ignore", divide="ignore"):
        tune_true = _nanmean_quiet(trial_responses, axis=-1)
        true = {m: _vector_strength(tune_true, p) for m, p in phase.items()}
        exceed = {m: np.zeros(n_rois) for m in phase}

        for s in range(0, n_shuffles, block):
            k = min(block, n_shuffles - s)
            xs = np.broadcast_to(trial_responses, (k, n_rois, n_dir, n_trial)).copy()
            xs = rng.permuted(xs, axis=2)          # permute directions per (roi, trial)
            ts = _nanmean_quiet(xs, axis=-1)
            for m, p in phase.items():
                exceed[m] += (true[m] < _vector_strength(ts, p)).sum(axis=0)

    return {m: (true[m], exceed[m] / n_shuffles) for m in phase}


def _nanmean_quiet(x: np.ndarray, axis: int) -> np.ndarray:
    """`np.nanmean` without the all-NaN-slice RuntimeWarning, which fires constantly on
    NaN-padded trial arrays and drowns real warnings."""
    finite = np.isfinite(x)
    n = finite.sum(axis=axis)
    total = np.where(finite, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)
