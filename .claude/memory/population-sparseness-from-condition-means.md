---
name: population-sparseness-from-condition-means
description: "How to compute population sparseness from condition_means_*.npz — the formula, the two readings of the white paper's definition, and the choices that change the answer. Deliberately not shipped as a column."
metadata:
  node_type: memory
  type: reference
  modified: 2026-09-02T00:00:00.000Z
---

The V1DD white paper reports population sparseness (Figure 19, bottom) beside lifetime
sparseness. **We deliberately do not ship it as a column** — it is not a per-ROI quantity,
and every table in the asset is one row per ROI. Instead the asset ships the matrix it is
computed from, so it can be derived at whatever population definition the question wants.

## Where the input lives

`condition_means_M409828.npz`, written by `V1DD Stimulus Metrics.ipynb`:

| array | shape |
|---|---|
| `ni_mean` | (39,407 x 118) trial-mean response per ROI per image |
| `ni12_mean` | (39,407 x 12) |
| `ni_images`, `ni12_images` | the image ids those columns correspond to |
| `roi_key` | joins to the wide table |
| `trace_type` | which trace the responses came from (`events`) |

**Natural movie is not in there** — (39,407 x 3,600) float32 is ~541 MiB even after
averaging over repeats, against ~20 MiB for the two image sets together.

`ni12_images` is a **sparse subset of 0..117**, not 0..11: images-12 draws from the same
118-image namespace. Re-ranking them to 0..11 would look tidier and be wrong.

## The formula

Vinje & Gallant (2000), the same expression `lifetime_sparseness` already uses, with the
index swapped from conditions to neurons:

```python
def sparseness(r, axis):
    """r >= 0. Returns 1 for a maximally sparse vector, 0 for a flat one."""
    n = np.sum(np.isfinite(r), axis=axis)
    s1 = np.nansum(r, axis=axis) ** 2
    s2 = np.nansum(r ** 2, axis=axis) * n
    return (1 - s1 / s2) / (1 - 1 / n)

# population sparseness, per image, within one population
pop = sparseness(ni12_mean[roi_in_population], axis=0)   # -> (n_images,)

# lifetime sparseness, per neuron -- reproduces the shipped column
life = sparseness(ni12_mean, axis=1)                     # -> (n_rois,)
```

## Three choices that change the answer

1. **Which reading of the paper.** Its words — "N is the number of neurons and Ri is
   average response vector of neuron i to *all stimulus conditions*" — describe averaging
   over conditions first, giving **one number per population** (how unevenly mean activity
   is spread across cells). The standard Vinje & Gallant reading is **per condition**:
   sparseness of the population response to each image, giving one number per image. They
   are different quantities. Prefer the per-condition version and say which you used.
2. **What counts as a population.** Six planes are imaged *simultaneously* within a volume,
   so a plane (~263 ROIs) and a session/volume (~1,576) are both defensible. The paper's
   "for each volume" implies the latter. **Sparseness depends strongly on N**, so the two
   are not comparable to each other.
3. **Which ROIs.** Invalid ROIs (`pika_roi_confidence <= 0.5`) are zeroed, not dropped, in
   several places upstream. A zero row inflates sparseness. Filter before computing —
   this matters most in column 4 / volume 1, where 1,038 of 1,550 ROIs are low confidence.

## Why it will not match Figure 19

The paper computes sparseness on **dF/F**; our condition means are on **events**. That
already applies to the shipped `lifetime_sparseness` column, which is therefore not
comparable to Figure 19's top panel either — a point worth making before anyone treats a
disagreement as a defect. See [[v1dd-metrics-open-questions]] for the wider trace-type
divergence from the paper.

## What else the matrix is for

Population sparseness was the reason it was shipped, but it is the smaller use. The matrix
is `(n_neurons, n_conditions)` trial-averaged responses, which is exactly what
`code/utils/functional_similarity.py`'s `signal_correlation` expects and previously had
nothing in the asset to read. Also decoding, dimensionality, and any population geometry
question. Every published natural-image column (`pref_img`, `pref_response`,
`lifetime_sparseness`, `z_score`) is a reduction of it, and a reduction cannot be un-taken.
