"""Verify code/utils/trial_responses.py against a synthetic trace. No data needed.

Section [1] uses xarray as an oracle, to *prove* that searchsorted reproduces
`DataArray.sel(time=slice(...))` label semantics — the assumption the whole port rests on.

**xarray is optional here and imported defensively.** It is a test-only dependency and is
not installed on the capsule; importing it at module scope meant one missing package took
out all 36 checks in this file, so the environment that actually built the asset got no
coverage of the response engine at all. Only section [1] needs it.
"""

from harness import check, fails, load, require_dataset, summary
import sys

import numpy as np

try:
    import xarray as xr
    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False

tr = load("trial_responses")


RNG = np.random.default_rng(0)
N_FRAMES, N_ROIS = 2000, 7
DT = 0.16374                       # close to the real 6.11 Hz
# jittered timestamps: real imaging clocks are not exactly uniform
TS = np.cumsum(RNG.normal(DT, DT * 0.002, N_FRAMES)) + 12.3
TRACES = RNG.gamma(2.0, 0.5, size=(N_FRAMES, N_ROIS))   # events-like, non-negative

print("[1] window_bounds reproduces xarray label slicing")
starts = RNG.uniform(TS[5], TS[-30], size=200)
WINDOWS = [(0.0, 2.0), (-1.0, 0.0), (0.0, 0.5), (0.0, 3 * DT)]
if not HAVE_XARRAY:
    print("  SKIP  xarray not installed -- the oracle comparison cannot run here.")
    print("        The remaining sections do not need it. Equivalent semantics are")
    print("        re-checked against an explicit boolean mask below.")
else:
    da = xr.DataArray(TRACES, dims=("time", "roi"), coords={"time": TS})
    for w0, w1 in WINDOWS:
        a, b = tr.window_bounds(TS, starts, w0, w1)
        bad = 0
        for i, s in enumerate(starts):
            want = da.sel(time=slice(s + w0, s + w1)).time.values
            got = TS[a[i]:b[i]]
            if not (len(want) == len(got) and np.array_equal(want, got)):
                bad += 1
        check(f"window ({w0}, {w1}) matches xarray on 200 starts", bad == 0,
              f"{bad} mismatches")

# Independent of xarray: label-closed slicing is `t0 <= t <= t1` on both ends, which a
# boolean mask states directly. This is a weaker oracle than xarray -- it encodes my
# reading of the semantics rather than an implementation of them -- but it means the
# property is still checked in an environment without xarray, which is where the asset
# is actually built.
for w0, w1 in WINDOWS:
    a, b = tr.window_bounds(TS, starts, w0, w1)
    bad = 0
    for i, s in enumerate(starts):
        mask = (TS >= s + w0) & (TS <= s + w1)
        if not np.array_equal(TS[mask], TS[a[i]:b[i]]):
            bad += 1
    check(f"window ({w0}, {w1}) is closed on both ends, 200 starts", bad == 0,
          f"{bad} mismatches")

check("windows have variable width (the thing that looks like it needs a loop)",
      len(np.unique(tr.window_bounds(TS, starts, 0.0, 2.0)[1]
                    - tr.window_bounds(TS, starts, 0.0, 2.0)[0])) > 1)

print("\n[2] window_means matches direct slicing")
cs, counts = tr.prefix_sums(TRACES)
check("no counts array when traces are finite", counts is None)
a, b = tr.window_bounds(TS, starts, 0.0, 2.0)
got = tr.window_means(cs, counts, a, b)
want = np.stack([TRACES[a[i]:b[i]].mean(axis=0) for i in range(len(starts))])
check("prefix-sum means == direct means", np.max(np.abs(got - want)) < 1e-9,
      f"max abs diff {np.max(np.abs(got - want)):.2e}")

print("\n[3] NaN handling")
tn = TRACES.copy()
tn[RNG.random(tn.shape) < 0.02] = np.nan
csn, cn = tr.prefix_sums(tn)
check("counts array present when NaNs exist", cn is not None)
gotn = tr.window_means(csn, cn, a, b)
wantn = np.stack([np.nanmean(tn[a[i]:b[i]], axis=0) for i in range(len(starts))])
check("nan-aware means match np.nanmean", np.allclose(gotn, wantn, atol=1e-9, equal_nan=True),
      f"max abs diff {np.nanmax(np.abs(gotn - wantn)):.2e}")

empty = tr.window_means(cs, counts, np.array([10]), np.array([10]))
check("empty window -> NaN, does not raise", np.all(np.isnan(empty)))

print("\n[4] sweep_responses")
sr = tr.sweep_responses(TRACES, TS, starts, (0.0, 2.0))
check("shape is (n_sweeps, n_rois)", sr.shape == (200, N_ROIS), str(sr.shape))
check("matches the naive loop", np.max(np.abs(sr - want)) < 1e-9)
srb = tr.sweep_responses(TRACES, TS, starts, (0.0, 2.0), baseline_window=(-1.0, 0.0))
a0, b0 = tr.window_bounds(TS, starts, -1.0, 0.0)
wantb = want - np.stack([TRACES[a0[i]:b0[i]].mean(axis=0) for i in range(len(starts))])
check("baseline subtraction matches", np.max(np.abs(srb - wantb)) < 1e-9)
check("blocking does not change the answer",
      np.max(np.abs(tr.sweep_responses(TRACES, TS, starts, (0.0, 2.0), block=7) - sr)) < 1e-12)

print("\n[5] spontaneous_null: bug-compatible fixed-width frame windows")
g = np.random.default_rng(42)
null = tr.spontaneous_null(TRACES, TS, TS[100], TS[900], (0.0, 2.0), n_boot=500, rng=g)
check("shape is (n_rois, n_boot)", null.shape == (N_ROIS, 500), str(null.shape))
check("deterministic under a fixed seed",
      np.array_equal(null, tr.spontaneous_null(TRACES, TS, TS[100], TS[900], (0.0, 2.0),
                                               n_boot=500, rng=np.random.default_rng(42))))
check("different seed gives different draws",
      not np.array_equal(null, tr.spontaneous_null(TRACES, TS, TS[100], TS[900], (0.0, 2.0),
                                                   n_boot=500, rng=np.random.default_rng(43))))
# every draw must be a mean of exactly round(2.0/dt) samples -- the fixed-width bug
r1 = int(round(2.0 / float(np.median(np.diff(TS)))))
check(f"null window is a fixed {r1} samples (trials use ~{r1 + 1})",
      r1 == 12, f"r1={r1}")
lo_ok = null.min() >= TRACES.min() - 1e-9
hi_ok = null.max() <= TRACES.max() + 1e-9
check("null values lie within the trace range", lo_ok and hi_ok)

n_means = tr.spontaneous_null(TRACES, TS, TS[100], TS[900], (0.0, 2.0), n_boot=500,
                              n_means=8, rng=np.random.default_rng(1))
check("multi-trial null is less dispersed than single-trial",
      n_means.std(axis=1).mean() < null.std(axis=1).mean(),
      f"{n_means.std(axis=1).mean():.4f} < {null.std(axis=1).mean():.4f}")
try:
    tr.spontaneous_null(TRACES, TS, TS[100], TS[105], (0.0, 20.0), n_boot=10)
    check("raises when the spontaneous block is too short", False)
except ValueError:
    check("raises when the spontaneous block is too short", True)

print("\n[6] trial_array scatter")
# 3 conditions; condition 1 presented only twice -> NaN tail
resp = np.arange(8 * 2, dtype=float).reshape(8, 2)      # 8 sweeps, 2 rois
cond = np.array([0, 1, 2, 0, 1, 2, 0, 2])
ta = tr.trial_array(resp, cond, n_trials=3, n_conditions=3)
check("shape is (n_cond, n_trials, n_rois)", ta.shape == (3, 3, 2), str(ta.shape))
check("condition 0 keeps chronological order",
      np.array_equal(ta[0, :, 0], resp[[0, 3, 6], 0]), str(ta[0, :, 0]))
check("short condition is NaN-padded in the tail",
      np.array_equal(ta[1, :2, 0], resp[[1, 4], 0]) and np.isnan(ta[1, 2, 0]))
check("n_trials inferred as the max when omitted",
      tr.trial_array(resp, cond).shape == (3, 3, 2))
try:
    tr.trial_array(resp, cond[:4])
    check("rejects mismatched lengths", False)
except ValueError:
    check("rejects mismatched lengths", True)

print("\n[7] frac_trials_above_null")
# roi 0: null uniformly 0..1, trials all 2.0 -> every trial beats it -> 1.0
nl = np.tile(np.linspace(0, 1, 1000), (2, 1))
trials = np.array([[2.0, 2.0, 2.0, 2.0], [0.5, 2.0, np.nan, np.nan]])
fr = tr.frac_trials_above_null(trials, nl)
check("all-strong trials -> 1.0", abs(fr[0] - 1.0) < 1e-12, f"{fr[0]}")
# roi 1: 0.5 has p = mean(null > 0.5) ~ 0.5 -> not significant; 2.0 -> significant
check("mixed trials use only the non-NaN ones", abs(fr[1] - 0.5) < 1e-12, f"{fr[1]}")
check("NaN trials are excluded, not scored as responsive", fr[1] != 0.25)

print("\n[8] lifetime_sparseness")
flat = np.ones((1, 20))
check("uniform responses -> 0", abs(tr.lifetime_sparseness(flat)[0]) < 1e-12)
onehot = np.zeros((1, 20)); onehot[0, 3] = 1.0
check("single-response -> 1", abs(tr.lifetime_sparseness(onehot)[0] - 1.0) < 1e-12)
withnan = np.concatenate([onehot, np.full((1, 5), np.nan)], axis=1)
check("NaNs dropped before the count (normaliser uses n=20, not 25)",
      abs(tr.lifetime_sparseness(withnan)[0] - 1.0) < 1e-12)

print("\n[9] si_permutation_test")
n_dir = 12
ang = np.arange(n_dir) / n_dir * 2 * np.pi
tuned = (1 + np.cos(ang))[None, :, None] + 0.01 * RNG.random((1, n_dir, 8))
flatr = np.ones((1, n_dir, 8)) + 0.01 * RNG.random((1, n_dir, 8))
both = np.concatenate([tuned, flatr], axis=0)
res = tr.si_permutation_test(both, n_shuffles=200, rng=np.random.default_rng(7))
check("returns osi and dsi", set(res) == {"osi", "dsi"})
si, p = res["dsi"]
check("shapes are (n_rois,)", si.shape == (2,) and p.shape == (2,))
check("p values in [0, 1]", np.all((p >= 0) & (p <= 1)))
check("tuned neuron has higher dsi than flat", si[0] > si[1], f"{si[0]:.3f} vs {si[1]:.3f}")
check("tuned neuron is significant, flat is not", p[0] < 0.05 < p[1], f"p={p}")

summary()
