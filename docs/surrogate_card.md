# Surrogate card

This repository ships no trained model artifact, deliberately: the Gaussian-
process surrogate is refit online from each run's own measurements, so there
are no weights whose provenance needs tracking. This card documents the
surrogate the way a model card would, because the surrogate's assumptions are
exactly where active learning can fail.

## Specification

- Model: exact GP regression, from scratch (Cholesky solves, no GP library).
- Kernels: RBF (default) or Matern-3/2, isotropic, on probe coordinates
  normalised to [0, 1]^2.
- Prior mean: constant, the empirical mean of the observations at the last
  refit.
- Hyperparameters: lengthscale, signal variance, noise variance. Fitted by
  maximising the log marginal likelihood with L-BFGS-B over log-parameters,
  3 starts, bounds: lengthscale in [0.01, 1.0] (normalised units), signal
  variance within a factor 1000 of the sample variance, noise variance in
  [1e-8, 10].
- Refit schedule inside an active run: after the 16-point Latin-hypercube
  initialisation, then every 25 measurements (both configurable). Between
  refits the posterior updates by exact rank-one formulas that a unit test
  pins to the batch posterior at 1e-10.

## Intended use

Sequential measurement placement and map reconstruction on fields whose
variation is smooth on a single dominant length scale, with roughly constant
Gaussian measurement noise. That is the regime the simulator generates and
the regime the headline benchmarks measure.

## Known failure modes (measured, not hypothetical)

The numbers behind each of these live in `results/` and are discussed in
RESULTS.md.

- A pinned, badly wrong lengthscale degrades both the design and the
  reconstruction (`results/misspecification.json`). The benchmark isolates
  the design effect by scoring the Latin-hypercube baseline through the same
  pinned reconstructor.
- On the non-stationary grain field, one global lengthscale is a compromise:
  it cannot be short at boundaries and long inside domains at the same time
  (`results/nonstationary.json`).
- Posterior-variance acquisition with fixed hyperparameters does not use the
  measured values at all: the GP posterior variance depends only on where
  points are, not what they read. Its advantage over space-filling designs
  can therefore only come from online hyperparameter adaptation, and it
  concentrates samples at the field-of-view edges where the prior variance
  is least constrained.
- Defect hunting assumes positive-going anomalies against a robust
  (median + 3 x scaled MAD) threshold of the values observed so far. Bipolar
  or contrast-inverted defects need a different acquisition.

## Out of scope

Experimental noise that is dose-dependent, spatially correlated, or heavy-
tailed; drift during acquisition; anisotropic correlation; multi-channel
signals. None of these are simulated, and the surrogate has no terms for
them.
