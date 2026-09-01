---
name: v1dd-metrics-speedups
description: Speedups for the V1DD stimulus-metrics run — speedup 1 (fit-only-used-SF) implemented 2026-09-01; speedup 2 (data-derived p0) still deferred.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-09-01T00:00:00.000Z
---

Profiled on the capsule, one plane of `409828_V1DD_Filtered`. The full 25-session run is
**~7.2 h**, and drifting gratings is **96 %** of it. Deliberately not optimised — the run
was started as-is. Revisit on an efficiency pass.

**Speedup 1 is now implemented (2026-09-01).** See implementation details below; expected
~2x wall-clock reduction for the full run (from ~7.2 h to ~3.6 h).

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

## Speedup 1: fit only the spatial frequency that gets read (~2x, output identical) ✓ DONE

**Implemented 2026-09-01.** `drifting_gratings_metrics` now has `fit_sf_index` parameter:
- Windowed: self-selects `pref_cond_index[:, 1]` automatically (when `fit_sf_index=None`)
- Full field: notebook passes `dg["windowed"].pref_cond_index[:, 1]` (windowed computed first)
- The loop is unrolled in the notebook cell 12: windowed first, then full field with `fit_sf_index`
- 41 DG tests still pass; output is bit-for-bit identical to the old loop for both types

`tuning_params` has exactly one consumer — `surround_suppression_metrics`:

```python
wp, fp = dgw.tuning_params[roi, si], dgf.tuning_params[roi, si]   # si = DGW's preferred SF
```

One SF per ROI is read from each grating type. With `n_sf = 2` **half of every fit was
discarded**. That half is now never computed.

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
