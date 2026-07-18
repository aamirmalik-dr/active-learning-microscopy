# Python API

Everything below is importable from the top-level package. All examples are
runnable as-is from the repository root.

## Scenes

```python
from activescan import SceneParams, make_scene

scene = make_scene(SceneParams(
    grid=64,            # 64x64 allowed probe positions
    length_scale=10.0,  # background correlation length, px
    field_kind="smooth",  # or "grains" (sharp Voronoi boundaries)
    n_defects=8,        # rare defects on top of the background
    defect_amplitude=5.0,  # in units of the background std
    defect_sigma=2.0,   # px
    noise_sigma=0.3,    # measurement noise, units of background std
    seed=0,
))
scene.field            # (64, 64) exact ground truth
scene.defect_centers   # (8, 2) exact defect positions
scene.coords()         # (4096, 2) normalised probe coordinates
```

Scenes are deterministic in their parameters: the same `SceneParams` always
produces the same field, which is what makes the fixed-seed benchmarks and
the committed-sample regression test possible.

A measurement is the only way strategies see a scene:

```python
import numpy as np
rng = np.random.default_rng(0)
values = scene.measure(np.array([0, 100, 4095]), rng)  # field + Gaussian noise
```

## Strategies

```python
from activescan import run_strategy, STRATEGIES

print(STRATEGIES)
# ('random', 'lhs', 'raster', 'active_variance', 'active_gradient', 'active_hunt')

run = run_strategy(scene, "active_variance", budget=200, seed=0)
run.order    # flat grid indices in acquisition order
run.values   # the noisy measurements the strategy actually saw
```

Baselines (`random`, `lhs`, `raster`) commit to their design up front;
`raster` is coarse-to-fine (nested strides), so any budget prefix is a
progressively refined grid. Active strategies loop: fit a Gaussian-process
surrogate, maximise an acquisition function over unmeasured positions,
measure there, update. The acquisitions are posterior variance
(`active_variance`), variance weighted by the posterior-mean gradient
(`active_gradient`), and expected exceedance above a robust threshold for
defect hunting (`active_hunt`). `fixed_lengthscale=...` pins the surrogate
lengthscale, which is how the misspecification benchmark constructs a wrong
surrogate on purpose. `record_every=10` stores posterior snapshots for
animation.

## The GP surrogate

```python
from activescan import GP, GPHyperparams, SequentialGP

gp = GP("rbf")
hyper = gp.fit_hyperparams(x, y)      # marginal-likelihood fit (L-BFGS-B)
gp.fit(x, y)
mean, var = gp.predict(x_star)
```

`SequentialGP` maintains the same posterior over a fixed candidate set while
points arrive one at a time; each update is O(candidates x n) instead of a
full refit, and a unit test pins it to the batch posterior at 1e-10:

```python
seq = SequentialGP(scene.coords(), "rbf", GPHyperparams(0.15, 1.0, 0.09))
seq.add(index=2048, value=0.7)
seq.mu, seq.var        # posterior over all 4096 candidates
seq.refit()            # re-optimise hyperparameters on the data so far
```

## Reconstruction and scoring

```python
from activescan import gp_reconstruct, interp_reconstruct, rmse, defect_hit_steps

recon, hyper = gp_reconstruct(scene, run.order, run.values)  # shared reconstructor
err = rmse(recon, scene.field)          # true error, units of background std
steps = defect_hit_steps(scene, run.order)  # 1-based first-hit step per defect, -1 if missed
```

Every strategy is scored through the same reconstructor, so the benchmark
compares designs, not reconstruction tricks. `interp_reconstruct` (cubic,
GP-free) exists for the fairness check in `configs/fairness.yaml`.

## Benchmarks

```python
from activescan import run_config

result = run_config("configs/reconstruction.yaml")  # writes results/reconstruction.json
result["strategies"]["active_variance"]["mean"]     # RMSE per budget checkpoint
```

Modes: `reconstruction`, `noise_sweep`, `defect_search`, `sparsity_sweep`,
`misspecification`, `fairness`. Every config fixes its seeds; replicate seeds
regenerate both the scene and the noise.

## Persistence and external data

```python
from activescan import save_scene, load_scene, save_run, load_run, load_external

save_scene("scene.npz", scene)
scene2 = load_scene("scene.npz")            # ground truth and params intact
ext = load_external("my_map.npy", noise_sigma=0.0)  # replay your own map
```

See [data/README.md](../data/README.md) for the bring-your-own-data details.

## CLI

```
activescan simulate --grid 64 --defects 8 --out scene.npz --figure scene.png
activescan run scene.npz --strategy active_variance --budget 200 --figure run.png --gif run.gif
activescan replay my_map.npy --strategy active_variance --budget 300
activescan benchmark configs/reconstruction.yaml
activescan demo
```
