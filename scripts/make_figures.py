"""Regenerate every committed figure from committed results and fixed seeds.

    python scripts/make_figures.py

Benchmark curves are drawn straight from results/*.json; the hero panel, the
acquisition GIF, and the defect-hunt panel re-execute short fixed-seed runs
(about a minute total).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from activescan import (
    SceneParams,
    gp_reconstruct,
    load_scene,
    make_scene,
    run_strategy,
)
from activescan.metrics import defect_hit_steps, lattice_coverage_fraction
from activescan.plots import (
    STRATEGY_LABELS,
    animate_run,
    hero_figure,
    plot_curves,
    plot_defect_curves,
    plot_fairness,
    plot_misspecification,
    plot_scene,
    plot_sweep,
)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RES = ROOT / "results"

HERO_BUDGET = 150


def _load(name: str) -> dict:
    with open(RES / name, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    FIG.mkdir(exist_ok=True)

    # scene gallery: the two field kinds
    smooth = make_scene(SceneParams(grid=64, length_scale=10, n_defects=8, seed=42))
    grains = make_scene(
        SceneParams(grid=64, length_scale=10, field_kind="grains", n_grains=12, seed=0)
    )
    plot_scene(smooth, FIG / "scene_smooth.png")
    plot_scene(grains, FIG / "scene_grains.png")
    print("wrote scene figures")

    # hero: three strategies spend the same budget on the same hidden field
    scene = make_scene(SceneParams(grid=64, length_scale=10.0, noise_sigma=0.3, seed=0))
    runs = {}
    for name in ("lhs", "raster", "active_variance"):
        run = run_strategy(scene, name, HERO_BUDGET, seed=0)
        recon, _ = gp_reconstruct(scene, run.order, run.values)
        runs[name] = (run, recon)
    hero_figure(scene, runs, _load("reconstruction.json"), FIG / "hero.png")
    print("wrote hero.png")

    # acquisition animation
    run = run_strategy(scene, "active_variance", HERO_BUDGET, seed=0, record_every=5)
    animate_run(scene, run, FIG / "acquisition.gif", fps=5)
    print("wrote acquisition.gif")

    # defect hunt panel on the committed sample scene
    sample = load_scene(ROOT / "data" / "sample" / "scene_64.npz")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    g = sample.grid
    for ax, name in zip(axes, ("random", "lhs", "active_hunt")):
        hunt = run_strategy(sample, name, 300, seed=0)
        steps = defect_hit_steps(sample, hunt.order)
        ax.imshow(sample.field, cmap="cividis")
        sc = ax.scatter(
            hunt.order % g,
            hunt.order // g,
            c=np.arange(len(hunt.order)),
            cmap="autumn",
            s=7,
            linewidths=0,
        )
        for k, (cy, cx) in enumerate(sample.defect_centers):
            found = steps[k] > 0
            ax.add_patch(
                plt.Circle(
                    (cx, cy),
                    3.5,
                    fill=False,
                    color="w" if found else "r",
                    lw=1.6,
                    ls="-" if found else "--",
                )
            )
        n_found = int(np.sum(steps > 0))
        ax.set_title(
            f"{STRATEGY_LABELS[name]}: {n_found}/{len(steps)} defects in 300",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axes, fraction=0.02, label="acquisition order")
    fig.suptitle(
        "defect hunting on the committed sample: found defects circled white, missed dashed red",
        fontsize=11,
    )
    fig.savefig(FIG / "defect_hunt_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote defect_hunt_panel.png")

    # benchmark curves from committed JSON
    plot_curves(
        _load("reconstruction.json"),
        FIG / "reconstruction_curves.png",
        "stationary smooth field: reconstruction error vs budget (5 seeds)",
    )
    plot_curves(
        _load("nonstationary.json"),
        FIG / "nonstationary_curves.png",
        "non-stationary grain field: reconstruction error vs budget (5 seeds)",
    )
    plot_defect_curves(
        _load("defect_search.json"),
        FIG / "defect_search.png",
        "defects found vs budget, 8 defects on a 4096-position grid (5 seeds)",
    )
    plot_sweep(
        _load("noise_sweep.json"),
        FIG / "noise_sweep.png",
        "noise_sigmas",
        "measurement noise sigma (units of background std)",
        "reconstruction RMSE at 150 measurements",
        "noise robustness at a fixed budget (3 seeds)",
        logy=True,
    )
    plot_sweep(
        _load("sparsity_sweep.json"),
        FIG / "sparsity_sweep.png",
        "defect_counts",
        "number of defects in the scene",
        "fraction of defects found at 300 measurements",
        "defect sparsity sweep (3 seeds)",
    )
    size_result = _load("size_sweep.json")
    core = float(np.sqrt(2.0 * np.log(2.0)))
    sigmas = [float(s) for s in size_result["defect_sigmas"]]
    coverage = [lattice_coverage_fraction(4.0, core * s) for s in sigmas]
    plot_sweep(
        size_result,
        FIG / "size_sweep.png",
        "defect_sigmas",
        "defect sigma (px); core radius = 1.18 sigma",
        "fraction of defects found at 300 measurements",
        "defect size sweep: where raster's lattice coverage breaks (3 seeds)",
        overlay=(sigmas, coverage, "stride-4 coverage of the core (geometry)"),
    )
    plot_misspecification(_load("misspecification.json"), FIG / "misspecification.png")
    plot_fairness(_load("fairness.json"), FIG / "fairness.png")
    print("wrote benchmark figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
