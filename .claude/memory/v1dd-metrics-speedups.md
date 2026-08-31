---
name: v1dd-metrics-speedups
description: Two unimplemented speedups for the V1DD stimulus-metrics run — deferred 2026-08-15 after profiling showed von Mises fits are 96% of the cost.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-16T04:06:51.167Z
---

Profiled on the capsule, one plane of `409828_V1DD_Filtered`. The full 25-session run is
**~7.2 h**, and drifting gratings is **96 %** of it. Deliberately not optimised — the run
was started as-is. Revisit on an efficiency pass.

## Where the time actually goes

| step | per plane | share |
|---|---|---|
| **drifting gratings (x2)** | **164.4 s** | **95.7 %** |
| natural images 12 | 2.6 s | 1.5 % |
| `load_plane` | 2.0 s | 1.2 % |
| natural movie | 1.0 s | 0.6 % |
| natural images | 0.9 s | 0.5 % |
| receptive fields | 0.7 s | 0.4 % |
| surround suppression | 0.1 s | 0.1 % |
| session setup (per session, not per plane) | 75.7 s | — |

Nothing outside drifting gratings is worth touching. Inside it, the cost is
`vonmises_two_peak_fit`: roughly **1,220 fits per plane** (~306 ROIs x 2 SFs x 2 grating
types) at ~134 ms each on real data. On clean synthetic curves the same fit takes 20-25 ms
with no failures, so the extra is real curves hitting the 2,000-eval attempt and retrying
at 10,000 (`max_fn_calls=(2000, 10000)`).

## Speedup 1: fit only the spatial frequency that gets read (~2x, output identical)

`tuning_params` has exactly one consumer — `surround_suppression_metrics`:

```python
wp, fp = dgw.tuning_params[roi, si], dgf.tuning_params[roi, si]   # si = DGW's preferred SF
```

One SF per ROI is read from each grating type. The fit loop in `drifting_gratings_metrics`
does `for roi: for sf_i in range(n_sf)`, so with `n_sf = 2` **half of every fit is
discarded**.

* **Windowed** can self-select: `pref_cond_index` is computed *before* the fit block, so
  it already knows which SF it will need.
* **Full field** cannot — `si` is the *windowed* stimulus's preferred SF. It needs a new
  `fit_sf_index` argument, and the caller passes `dgw.pref_cond_index[:, 1]` after
  computing windowed first.

Output cannot change: the skipped fits are exactly the ones nothing reads. Verify by
re-running the 2-session smoke test and diffing bit-for-bit — see
[[v1dd-metrics-refactor-decisions]] for that workflow.

## Speedup 2: a data-derived initial guess (maybe 2-3x, but changes numbers)

`vonmises_two_peak_fit` uses a fixed `p0=(0.1, 1, 180, 0.01, 1, 0.001)`. Event amplitudes
are ~1e-3 to 1e-2, so `scale_1 = 0.1` starts one to two orders of magnitude high, which is
plausibly why real curves exhaust 2,000 evaluations. Deriving `p0` from the data — peak
amplitude, peak location, baseline as the minimum — should cut iterations substantially.

**This can change `ssi_tuning_fit`**: a different starting point can settle in a different
local minimum, and the fit is bounded 6-parameter least squares on 12 points. It is not an
output-neutral change, so it needs re-validating against the reference tables rather than
just a bit-for-bit diff. Do speedup 1 first and separately.

## Not worth doing

Reducing `other_n_boot` from 10,000 — the four families that use it total under 2 % of
runtime. `fit_tuning_curves=False` would take the run to ~30 min but makes `ssi_tuning_fit`
all-NaN, losing the only surround-suppression column derived from a fitted curve.
