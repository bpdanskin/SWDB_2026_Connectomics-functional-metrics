---
name: dg-blank-sweeps-are-ragged
description: "V1DD drifting gratings: 192 is the TOTAL sweep count including blanks, blank counts vary 5-8 per session, and the tuning-curve export NaN-pads them rather than raising."
metadata:
  type: project
---

Measured across all 25 V1DD sessions on 2026-09-03 (stimulus tables only, ~6 min):

**192 is the total number of drifting-gratings sweeps per type, blank (grey) sweeps
included** — not the non-blank total, which is what HANDOFF.md claimed until this was
measured. So a session showing more blanks shows *fewer* grating trials, and the
12 x 2 x 8 = 192 condition slots only ever receive 184-187 of them.

| | `drifting_gratings_full` | `drifting_gratings_windowed` |
|---|---|---|
| total sweeps | 192 in all 25 | 192 in all 25 |
| blank sweeps | 7 (8 sessions), 8 (17) | 5 (1), 6 (1), 7 (9), 8 (14) |
| grating sweeps | 184-185 | 184-187 |

Only **13 of 25** sessions run the same blank count for both grating types, so it is a
per-stimulus property, not a per-session one. Blanks are detected as rows NaN in
`DG_PARAM_COLUMNS` (temporal_frequency, spatial_frequency, direction), so these are
genuine grey sweeps, not a masking artefact.

**Why it matters:** the tuning-curve writer cell asserted that every plane agreed on its
blank count, and that assertion killed the 2026-09-03 reproducible run after ~5 h — after
every metric had been computed, losing `tuning_curves`, `condition_means` and the
provenance JSON, none of which can be rebuilt from the CSVs. Now `dg{w,f}_blank` is
NaN-padded to the widest plane and `dg{w,f}_n_blank` `(n_planes,)` carries the true width,
keyed by `plane_key`. `np.nanmean(blank, axis=1)` is correct without reading it.

**The raise was not the mistake — the belief behind it was.** It is how this was
discovered. What is still unfixed is the *cost*: both array-writer cells run before
provenance and the manifest, so any raise in either still forfeits the whole run's
provenance. `condition_means`'s three assertions have still never seen real data. See
[[v1dd-metrics-asset-sanity-checks]] and [[response-window-deferred-tuning]].
