# Changelog

## 0.1.1 (2026-09-26)

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`.
- Citation file (`CITATION.cff`) and this changelog added.
- CI badge added to the README.
- Related repositories section in the README linking the six sibling electron-microscopy repositories.
- Test guarding the package `__version__` against the installed distribution metadata.

## 0.1.0 (2026-07-18)

- Scene simulator: stationary Gaussian random fields or Voronoi grain fields, rare defects with exact recorded centres, noisy point measurements, exact ground truth throughout.
- Exact Gaussian-process surrogate from scratch (RBF and Matern-3/2, Cholesky solves, marginal-likelihood fitting) with exact rank-one sequential posterior updates tested against the batch posterior.
- Strategies: random, Latin hypercube, and coarse-to-fine raster baselines; active variance, gradient-weighted variance, and expected-exceedance defect hunting with found-and-move-on exclusion.
- Scoring by true reconstruction RMSE through one shared reconstructor and by defects found inside the half-amplitude core; benchmarks over budget, noise, sparsity, defect size, misspecification, non-stationarity, and reconstructor fairness, with a lattice-coverage geometry for the raster defect-search result.
- The `activescan` CLI, replay-your-own-map mode, committed twin sample scenes, results JSON, figures including the sampling-trajectory panel and acquisition GIF, surrogate card, API docs, executed tutorial, and a CI workflow.
- Maintenance after publication: claims and physics wording tightened, README expanded and restructured, hero re-exported at higher resolution, acquisition video added.
