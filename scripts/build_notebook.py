"""Build and execute notebooks/tutorial.ipynb from scratch.

    python scripts/build_notebook.py

The notebook is generated programmatically (single source of truth) and then
executed top to bottom with nbclient, so the committed notebook always
carries real outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


CELLS = [
    md("""
# Where should the microscope look next?

A scanning microscope acquires one measurement per probe position, and dwell
time is the budget: a full raster of a 64 x 64 field of view costs 4096
measurements whether the sample is interesting everywhere or almost nowhere.
An autonomous microscope can instead decide, after every measurement, where
to measure next. This notebook walks that decision loop end to end on
simulated samples with exact ground truth:

1. simulate a sample and look at it,
2. see what a small measurement budget gives a raster scan,
3. build the Gaussian-process surrogate the active strategies use,
4. watch the active loop choose positions,
5. compare all strategies on reconstruction error versus budget,
6. hunt rare defects with an expected-exceedance acquisition,
7. read the two honest checks: surrogate misspecification and
   reconstructor-robustness.

Everything runs on CPU in a couple of minutes. Simulated fields, Gaussian
noise, exact ground truth; the README scope section says plainly what this
leaves out.
"""),
    code("""
%matplotlib inline
import json
import numpy as np
import matplotlib.pyplot as plt

from activescan import (
    SceneParams, make_scene, load_scene, run_strategy, gp_reconstruct,
    rmse, defect_hit_steps, GP, GPHyperparams, lhs_design,
)

scene = load_scene("../data/sample/scene_64.npz")          # with defects
smooth = load_scene("../data/sample/scene_smooth_64.npz")  # its defect-free twin
print(f"grid: {scene.grid}x{scene.grid} = {scene.n_positions} probe positions")
print(f"background correlation length: {scene.params.length_scale:g} px")
print(f"defects: {len(scene.defect_centers)}, "
      f"amplitude {scene.params.defect_amplitude:g}x background std")
print(f"measurement noise sigma: {scene.params.noise_sigma:g}")
"""),
    md("""
## 1. The sample

The committed sample scene is a smooth property field (composition, thickness,
strain, anything acquired point by point) with unit standard deviation and a
10 px correlation length, plus 8 narrow defects at 5x the background std.
Ground truth is exact, so every error number later is a true error.
"""),
    code("""
rng = np.random.default_rng(0)
noisy = scene.field + rng.normal(0, scene.params.noise_sigma, scene.field.shape)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, img, title in [
    (axes[0], scene.field, "ground truth (hidden from strategies)"),
    (axes[1], scene.background, "background component"),
    (axes[2], noisy, "what a full 4096-point raster returns"),
]:
    im = ax.imshow(img, cmap="cividis")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
axes[0].scatter(scene.defect_centers[:, 1], scene.defect_centers[:, 0],
                s=90, facecolors="none", edgecolors="w")
plt.show()
"""),
    md("""
## 2. What a small budget buys a raster scan

Suppose the budget is 64 measurements, 1.6 percent of the full raster, on
the defect-free twin. The fair raster comparison is coarse-to-fine (a
stride-8 grid), not a top-down raster that would only image the top rows.
Reconstruction here and everywhere
below is the same: a GP posterior mean with hyperparameters fitted to that
run's own measurements, so strategies differ only in where they measured.
"""),
    code("""
run64 = run_strategy(smooth, "raster", 64, seed=0)
recon64, hyper64 = gp_reconstruct(smooth, run64.order, run64.values)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
g = scene.grid
axes[0].imshow(smooth.field, cmap="cividis")
axes[0].scatter(run64.order % g, run64.order // g, c="r", s=10)
axes[0].set_title("64 raster positions", fontsize=10)
axes[1].imshow(recon64, cmap="cividis")
axes[1].set_title("GP reconstruction", fontsize=10)
err = np.abs(recon64 - smooth.field)
axes[2].imshow(err, cmap="magma")
axes[2].set_title(f"absolute error, rmse {rmse(recon64, smooth.field):.3f}", fontsize=10)
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.show()
print(f"rmse {rmse(recon64, smooth.field):.3f} in units of the background std")
"""),
    md("""
## 3. The surrogate

Active strategies need a model that predicts, at every unmeasured position,
both a value and an uncertainty. The package implements exact GP regression
from scratch: RBF kernel, constant prior mean, hyperparameters (lengthscale,
signal variance, noise variance) fitted by maximising the log marginal
likelihood. The posterior standard deviation is the raw material of every
acquisition function below.
"""),
    code("""
rng = np.random.default_rng(1)
idx40 = lhs_design(smooth, 40, rng)
vals40 = smooth.measure(idx40, rng)
gp = GP("rbf")
hyper = gp.fit_hyperparams(smooth.coords()[idx40], vals40)
gp.fit(smooth.coords()[idx40], vals40)
mu, var = gp.predict(smooth.coords())
print(f"fitted from 40 points: lengthscale {hyper.lengthscale:.3f} "
      f"(true background: {smooth.params.length_scale / smooth.grid:.3f}), "
      f"noise var {hyper.noise_var:.3f} (true: {smooth.params.noise_sigma**2:.3f})")
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].imshow(mu.reshape(g, g), cmap="cividis")
axes[0].scatter(idx40 % g, idx40 // g, c="r", s=8)
axes[0].set_title("posterior mean from 40 measurements", fontsize=10)
im = axes[1].imshow(np.sqrt(var).reshape(g, g), cmap="magma")
axes[1].set_title("posterior std: high where nothing was measured", fontsize=10)
fig.colorbar(im, ax=axes[1], fraction=0.046)
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.show()
"""),
    md("""
The fitted lengthscale and noise land close to the true generating values,
from 40 noisy points and no prior knowledge. That online adaptation is the
only information advantage active sampling has over a space-filling design:
with hyperparameters held fixed, the GP posterior variance depends only on
where the points are, not on what was measured, so variance-driven sampling
would reduce to a greedy space-filling design.

## 4. Watching the active loop decide

`active_variance` measures wherever the posterior std is highest, updating
the posterior after every measurement with exact rank-one updates (unit-
tested against the batch posterior). Snapshots show the posterior mean
sharpening and the uncertainty map being pushed down.
"""),
    code("""
active = run_strategy(smooth, "active_variance", 150, seed=0, record_every=25)
picks = [s for s in active.snapshots if s[0] in (50, 100, 150)]
fig, axes = plt.subplots(2, len(picks), figsize=(3.4 * len(picks), 6.4))
for j, (step, mu_s, sd_s) in enumerate(picks):
    axes[0, j].imshow(mu_s, cmap="cividis")
    pts = active.order[:step]
    axes[0, j].scatter(pts % g, pts // g, s=4, c="w", alpha=0.8, linewidths=0)
    axes[0, j].set_title(f"posterior mean, {step} measurements", fontsize=10)
    axes[1, j].imshow(sd_s, cmap="magma")
    axes[1, j].set_title("posterior std", fontsize=10)
for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
plt.show()
recon_a, _ = gp_reconstruct(smooth, active.order, active.values)
print(f"active_variance, 150 measurements: rmse {rmse(recon_a, smooth.field):.3f}")
"""),
    md("""
## 5. All strategies, same budget, same hidden field

150 measurements is 3.7 percent of the full raster. The benchmark curves
(committed under `results/`, regenerated by `configs/reconstruction.yaml`)
average 5 scene seeds; the table printed below is this notebook's own live
run on the sample scene.
"""),
    code("""
budget = 150
rows = []
for name in ("random", "lhs", "raster", "active_variance", "active_gradient"):
    r = run_strategy(smooth, name, budget, seed=0)
    rec, _ = gp_reconstruct(smooth, r.order, r.values)
    rows.append((name, rmse(rec, smooth.field)))
print(f"{'strategy':<18}{'rmse at 150':>12}")
for name, e in rows:
    print(f"{name:<18}{e:>12.3f}")
"""),
    code("""
with open("../results/reconstruction.json") as fh:
    recon_bench = json.load(fh)
fig, ax = plt.subplots(figsize=(6.8, 4.4))
colors = {"random": "#999999", "lhs": "#1f77b4", "raster": "#2ca02c",
          "active_variance": "#d62728", "active_gradient": "#9467bd"}
for name, block in recon_bench["strategies"].items():
    m = np.array(block["mean"]); s = np.array(block["std"])
    ax.plot(recon_bench["checkpoints"], m, label=name, color=colors[name], lw=1.8)
    ax.fill_between(recon_bench["checkpoints"], m - s, m + s, color=colors[name], alpha=0.15, lw=0)
ax.set_yscale("log"); ax.set_xlabel("measurements")
ax.set_ylabel("reconstruction RMSE (units of background std)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("committed benchmark: 5 seeds, full raster = 4096 points", fontsize=10)
plt.show()
tt = {s: b["budget_to_target"] for s, b in recon_bench["strategies"].items()}
print("measurements to reach RMSE 0.15 (per-seed mean):")
for s, vals in tt.items():
    ok = [v for v in vals if v is not None]
    print(f"  {s:<18}{np.mean(ok):7.0f}")
"""),
    md("""
## 6. Hunting rare defects

For defect search the acquisition switches to expected exceedance: how much
is this position expected to exceed a robust threshold (median + 3 scaled
MADs of everything measured so far)? Before any defect signal is seen the
acquisition behaves like space-filling exploration; after a measurement
clips a defect tail, the posterior mean lifts locally and the strategy homes
in. A defect counts as found only when a measurement lands inside its core
(the half-amplitude radius), a purely geometric criterion with no tunable
detector threshold.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, name in zip(axes, ("lhs", "active_hunt")):
    r = run_strategy(scene, name, 300, seed=0)
    steps = defect_hit_steps(scene, r.order)
    ax.imshow(scene.field, cmap="cividis")
    ax.scatter(r.order % g, r.order // g, c=np.arange(len(r.order)),
               cmap="autumn", s=7, linewidths=0)
    for k, (cy, cx) in enumerate(scene.defect_centers):
        found = steps[k] > 0
        ax.add_patch(plt.Circle((cx, cy), 3.5, fill=False, lw=1.6,
                                color="w" if found else "r", ls="-" if found else "--"))
    ax.set_title(f"{name}: {int((steps > 0).sum())}/8 defects in 300", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.show()
"""),
    code("""
with open("../results/defect_search.json") as fh:
    hunt_bench = json.load(fh)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, block in hunt_bench["strategies"].items():
    m = np.array(block["mean"]); s = np.array(block["std"])
    c = {"random": "#999999", "lhs": "#1f77b4", "raster": "#2ca02c",
         "active_hunt": "#e377c2"}[name]
    ax.plot(hunt_bench["checkpoints"], m, label=name, color=c, lw=1.8)
    ax.fill_between(hunt_bench["checkpoints"], m - s, m + s, color=c, alpha=0.15, lw=0)
ax.axhline(8, color="k", lw=0.8, ls=":")
ax.set_xlabel("measurements"); ax.set_ylabel("defects found (of 8)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("committed benchmark: 5 seeds", fontsize=10)
plt.show()
for name, block in hunt_bench["strategies"].items():
    print(f"{name:<14} found {block['n_found_at_budget']}/{block['n_defects_total']} "
          f"across seeds, median hit step {block['median_hit_step']}")
"""),
    md("""
## 7. The honest checks

**Misspecification.** Pin the surrogate lengthscale to 0.2x or 5x the true
value; both the active design and the reconstruction use the pinned value,
and the LHS design is scored through the same pinned reconstructor. This
isolates what a wrong surrogate does to the *decisions*.
"""),
    code("""
with open("../results/misspecification.json") as fh:
    miss = json.load(fh)
cols = miss["columns"]
x = np.arange(len(cols)); width = 0.35
fig, ax = plt.subplots(figsize=(6.8, 4))
for i, (name, c) in enumerate([("lhs", "#1f77b4"), ("active_variance", "#d62728")]):
    b = miss["strategies"][name]
    ax.bar(x + (i - 0.5) * width, b["mean"], width, yerr=b["std"], capsize=3,
           color=c, label=name)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("pinned_", "").replace("x", " x true ls") if "pinned" in c else c
                    for c in cols], fontsize=9)
ax.set_ylabel("RMSE at 150 measurements")
ax.legend(fontsize=9, frameon=False)
plt.show()
for name in ("lhs", "active_variance"):
    print(name, [round(v, 3) for v in miss["strategies"][name]["mean"]], "cols:", cols)
"""),
    md("""
**Reconstructor robustness.** The active-versus-LHS ranking could in
principle be an artifact of scoring every design through a GP, the same
model family the active strategy optimises. The fairness benchmark re-scores
the same designs through a GP tuned on an independent 1000-point reference
sample and through model-free cubic interpolation.
"""),
    code("""
with open("../results/fairness.json") as fh:
    fair = json.load(fh)
recons = fair["reconstructors"]
print(f"{'strategy':<18}" + "".join(f"{r:>20}" for r in recons))
for name, block in fair["strategies"].items():
    print(f"{name:<18}" + "".join(f"{m:>20.3f}" for m in block["mean"]))
"""),
    md("""
## Takeaways

- Active variance sampling reaches the target accuracy with meaningfully
  fewer measurements than Latin-hypercube sampling, and far fewer than
  coarse-to-fine raster, but the margin over a good space-filling design is
  a fraction, not an order of magnitude. The order-of-magnitude savings
  claim only holds against the raster habit.
- The advantage comes from online hyperparameter adaptation plus mild edge
  weighting, and it survives scoring through non-GP reconstructors.
- Defect hunting is where active decisions change the outcome qualitatively:
  expected exceedance converts one lucky graze into a found defect, which
  space-filling designs cannot do.
- A badly misspecified surrogate costs the active design real accuracy;
  with a pinned 5x lengthscale it falls behind the LHS baseline scored
  through the same reconstructor. Surrogate checking is part of the method,
  not an optional extra.

The benchmark tables with seeds and spreads are in RESULTS.md; every number
regenerates from `configs/*.yaml` via `scripts/run_all.py`.
"""),
]


def main() -> int:
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    out = ROOT / "notebooks" / "tutorial.ipynb"
    out.parent.mkdir(exist_ok=True)
    client = NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(out.parent)}})
    client.execute()
    nbformat.write(nb, out)
    n_err = sum(
        1
        for cell in nb.cells
        if cell.cell_type == "code"
        for o in cell.get("outputs", [])
        if o.get("output_type") == "error"
    )
    print(f"wrote {out} ({len(nb.cells)} cells, {n_err} error cells)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
