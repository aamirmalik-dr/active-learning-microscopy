# Results

Every number below was measured in a fresh Python 3.11 virtual environment by
running the fixed-seed configs in `configs/`; the raw values live in
`results/*.json` and regenerate with `python scripts/run_all.py` (about 10
minutes on a laptop CPU). RMSE is in units of the background standard
deviation. Replicate seeds regenerate both the scene and the measurement
noise, so spreads are across-scene, not just across-noise. All strategies are
scored through the same reconstructor (a GP with hyperparameters fitted to
that run's own measurements), so the tables compare measurement placement,
not reconstruction tricks.

The benchmark scene is a 64 x 64 grid (4096 raster positions), background
correlation length 10 px, measurement noise 0.3.

## 1. Reconstruction error versus budget (stationary field)

`configs/reconstruction.yaml`, 5 seeds, mean RMSE at each budget:

| Budget | random | LHS | raster (coarse-to-fine) | active variance | active gradient |
|---|---|---|---|---|---|
| 25 | 0.544 | 0.404 | 0.461 | 0.341 | **0.291** |
| 50 | 0.322 | 0.274 | 0.319 | **0.207** | 0.201 |
| 100 | 0.202 | 0.198 | 0.184 | **0.153** | 0.156 |
| 150 | 0.158 | 0.142 | 0.168 | **0.136** | 0.137 |
| 200 | 0.146 | 0.136 | 0.150 | **0.117** | 0.129 |
| 400 | 0.100 | 0.091 | 0.106 | **0.080** | 0.097 |

Measurements needed to reach RMSE 0.15 (per-seed interpolation, mean of 5
seeds, full raster = 4096):

| Strategy | Budget to RMSE 0.15 | Per-seed |
|---|---|---|
| active variance | **114** | 162, 112, 89, 112, 95 |
| LHS | 144 | 133, 143, 142, 168, 131 |
| active gradient | 157 | 368, 114, 114, 88, 102 |
| random | 185 | 238, 96, 215, 145, 228 |
| raster | 199 | 211, 207, 206, 173, 196 |

Reading: active variance sampling reaches the target with about 21 percent
fewer measurements than Latin hypercube (114 vs 144, winning on 4 of 5
seeds) and about 43 percent fewer than a coarse-to-fine raster, which is
2.8 percent of the full raster. The margin over a good space-filling design
is real but modest; the dramatic savings exist only relative to raster
scanning. Active gradient matches active variance on average but carries a
heavy-tailed risk (one seed needed 368), because chasing apparent gradients
early can misallocate the first measurements. At 150+ measurements LHS is
statistically close to active (0.142 +/- 0.013 vs 0.136 +/- 0.012); the
active advantage is concentrated in the sparse regime below about 4 percent
of the raster.

## 2. Non-stationary grain field

`configs/nonstationary.yaml`, 12 Voronoi domains with sharp boundaries, 5
seeds, mean RMSE:

| Budget | LHS | raster | active variance | active gradient |
|---|---|---|---|---|
| 100 | 0.480 | 0.502 | 0.466 | **0.451** |
| 200 | 0.428 | 0.453 | 0.407 | **0.380** |
| 400 | 0.358 | 0.351 | 0.361 | **0.342** |

Reading: everyone struggles, because a stationary RBF surrogate cannot be
simultaneously right inside domains and at their edges; errors are dominated
by boundary pixels regardless of design. Gradient weighting helps modestly
and consistently (0.380 +/- 0.023 vs 0.428 +/- 0.064 for LHS at 200), since
it steers measurements toward the boundaries where the field actually
changes. This is the regime where a data-dependent acquisition earns its
keep, and the improvement is incremental, not transformative.

## 3. Noise robustness at a fixed budget, including a failure regime

`configs/noise_sweep.yaml`, 150 measurements, 3 seeds, mean RMSE:

| Noise sigma | random | LHS | active variance |
|---|---|---|---|
| 0.05 | 0.032 | 0.036 | **0.028** |
| 0.1 | 0.053 | 0.060 | **0.052** |
| 0.2 | **0.091** | 0.098 | 0.104 |
| 0.4 | 0.163 | **0.156** | 0.184 |
| 0.8 | 0.289 | **0.255** | 0.276 |
| 1.6 | 0.488 | **0.386** | 0.565 |

Reading: this is a measured failure regime, reported as such. Active
sampling wins below noise 0.2 and loses increasingly badly above it; at
noise 1.6 it is 46 percent worse than LHS. Two mechanisms compound: noisy
data destabilises the online hyperparameter fit that active sampling
depends on, and once per-point information is low, averaging over space
(which a fixed design does implicitly) beats chasing the posterior. If the
per-measurement SNR is poor, use a space-filling design.

## 4. Defect search

`configs/defect_search.yaml`, 8 defects (amplitude 5, sigma 2 px) on the
smooth field, 5 seeds. Mean defects found (of 8):

| Budget | random | LHS | raster | active hunt |
|---|---|---|---|---|
| 50 | 1.2 | 0.8 | **1.6** | 1.2 |
| 100 | 1.6 | 1.2 | **2.8** | 2.2 |
| 150 | 3.2 | 2.4 | 4.6 | **5.4** |
| 200 | 4.0 | 3.0 | 6.0 | **6.8** |
| 300 | 5.8 | 4.4 | **8.0** | **8.0** |
| 500 | 6.8 | 7.0 | **8.0** | **8.0** |

Totals across all 40 defects at budget 500: raster and active hunt find 40
of 40 (median hit step 123.5 and 129.0); random finds 34, LHS 35. A defect
counts as found only when a measurement lands inside its half-amplitude
core, a purely geometric criterion.

Reading: the honest surprise is that a coarse-to-fine raster is an
excellent defect finder at this defect size. Its stride-4 pass is complete
by budget 256 and puts every position within 2.83 px of a sample; against
the 2.35 px core radius that covers 93.9 percent of possible centre
positions (`activescan.metrics.lattice_coverage_fraction(4, 2.355)`), and
a guarantee would require core radius 2.83 px, i.e. defect sigma 2.40.
Finding all 40 was therefore likely but not certain (about 2.4 expected
misses under uniform placement; these five seeds happened to draw none).
The hunt acquisition (expected exceedance with a found-and-move-on
exclusion) leads in the mid-budget window (5.4 vs 4.6 at 150, 6.8 vs 6.0
at 200) and matches the raster at 300. Random and LHS never reliably
finish, still missing about one defect in seven or eight at a
500-measurement budget.

## 5. Where the raster guarantee breaks: defect size

`configs/size_sweep.yaml`, the operating-point check for the raster-vs-hunt
parity above. Fraction of 8 defects found at 300 measurements, 3 seeds:

| Defect sigma (px) | LHS | raster | active hunt |
|---|---|---|---|
| 1.0 | **0.375** | 0.167 | 0.333 |
| 1.5 | 0.417 | 0.500 | **0.625** |
| 2.0 | 0.708 | **1.000** | **1.000** |
| 3.0 | 1.000 | 1.000 | 1.000 |

Reading: raster's parity at sigma 2.0 is a geometric coincidence between
the defect core and the stride ladder, and the stride-4 coverage fraction
of the core says exactly where it breaks: 94 percent at sigma 2.0, 61
percent at sigma 1.5, 27 percent at sigma 1.0 (raster observed 1.00, 0.50,
0.17, tracking the prediction within seed noise). Shrink defects to sigma
1.5 and the hunt finds 62.5 percent versus raster's 50 percent; at sigma
1.0 the cores (radius 1.18 px) slip between any affordable grid and defect
tails carry almost no signal, so every method fails and the hunt has no
advantage left (0.333 vs LHS 0.375, within one defect). Bayesian search
needs a signal to exploit; when the anomaly footprint approaches a single
pixel, nothing short of full-resolution coverage works.

## 6. Defect sparsity

`configs/sparsity_sweep.yaml`, fraction of defects found at 300
measurements, 3 seeds:

| Defects in scene | random | LHS | active hunt |
|---|---|---|---|
| 2 | 1.000 | 0.667 | **1.000** |
| 4 | 0.917 | 0.583 | **1.000** |
| 8 | 0.833 | 0.708 | **1.000** |
| 16 | 0.729 | 0.750 | **0.958** |

Reading: the hunt is at or near ceiling at every sparsity (its only misses
are 2 of 48 defects at n = 16, where exclusion zones start crowding the
field). LHS earns no defect-finding advantage over random: its per-axis
stratification controls 1D projections, not 2D point spacing, so its 2D
coverage is essentially that of a random design (measured max-gap 6.2 px
vs 6.4 px for random at this budget). The LHS-below-random cells in this
table are within the noise of 3 seeds, and the defect-search benchmark
above shows the two effectively tied (35 vs 34 of 40 at budget 500).

## 7. Surrogate misspecification (the honest check)

`configs/misspecification.yaml`, 150 measurements, 3 seeds. The active
design uses a surrogate whose lengthscale is pinned to a factor times the
true background lengthscale; both the active and the LHS designs are then
scored through the same pinned-lengthscale reconstructor, isolating the
design decision. RMSE:

| Surrogate lengthscale | LHS | active variance |
|---|---|---|
| 0.2 x true | 0.568 | **0.373** |
| 1 x true (pinned) | **0.145** | 0.154 |
| 5 x true | **0.285** | 0.373 (std 0.211) |
| fitted online | **0.129** | 0.137 |

Reading: three separate lessons. First, with the correct pinned
lengthscale, active variance sampling is statistically identical to LHS
(0.154 vs 0.145): with hyperparameters frozen, the GP posterior variance
does not depend on the measured values at all, so variance-driven sampling
is just a greedy space-filling design. Whatever advantage active learning
has must come from online hyperparameter adaptation. Second, an oversmooth
surrogate (5 x) is the dangerous direction: the active design trusts wrong
long-range extrapolations, underperforms LHS by 31 percent, and becomes
wildly unstable across seeds (std 0.211 vs 0.063). Third, the
overshort direction (0.2 x) punishes LHS harder than active sampling,
because a myopic reconstructor needs the small maximum-gap coverage that
greedy variance filling provides. And the fitted column is a reminder that
at this generous budget LHS with a well-fitted reconstructor already
matches active sampling; the case for active is the sparse regime (section
1), not this operating point.

## 8. Reconstructor-robustness (fairness) check

`configs/fairness.yaml`, 100 measurements (the operating point where the
headline active advantage is claimed), 3 seeds. The same four designs
scored through three reconstructors: the strategy's own fitted GP, a GP
with hyperparameters tuned once per scene on an independent 1000-point
random sample, and model-free cubic interpolation. RMSE:

| Strategy | GP fitted | GP reference-tuned | cubic interpolation |
|---|---|---|---|
| active variance | **0.155** | **0.153** | **0.315** |
| raster | 0.182 | 0.180 | 0.316 |
| random | 0.186 | 0.187 | 0.374 |
| LHS | 0.200 | 0.202 | 0.369 |

Reading: the active design is best under all three reconstructors,
including the GP-free one (where it ties raster, 0.315 vs 0.316), so the
budget-100 advantage is a property of where it measured, not an artifact
of scoring designs through the same model family the strategy optimises.
Cubic interpolation is uniformly much worse than GP reconstruction, and
favours regular grids, which is why raster closes the gap there.

## Wall-clock

Benchmark wall times on one laptop CPU core (from `results/metrics.json`):
reconstruction 100 s, nonstationary 89 s, noise sweep 22 s, defect search
58 s, sparsity sweep 37 s, size sweep 37 s, misspecification 11 s, fairness
104 s. A single 400-measurement active run on the 4096-position grid takes
about 7 s, of which most is the periodic hyperparameter refits; the
per-step posterior update is O(candidates x n) by rank-one updates.
