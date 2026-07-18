"""Figures: scene views, run panels, benchmark curves, and the acquisition GIF.

All benchmark plot functions consume the JSON dictionaries written by
:mod:`activescan.benchmark`, so committed figures regenerate from committed
results without re-running anything.
"""

from __future__ import annotations

import io as _io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .sim import ScanScene
from .strategies import RunResult

STRATEGY_COLORS = {
    "random": "#999999",
    "lhs": "#1f77b4",
    "raster": "#2ca02c",
    "active_variance": "#d62728",
    "active_gradient": "#9467bd",
    "active_hunt": "#e377c2",
}

STRATEGY_LABELS = {
    "random": "random",
    "lhs": "Latin hypercube",
    "raster": "coarse-to-fine raster",
    "active_variance": "active (variance)",
    "active_gradient": "active (gradient-weighted)",
    "active_hunt": "active (expected exceedance)",
}

_FIELD_CMAP = "cividis"


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scene(scene: ScanScene, path: str | Path) -> None:
    """Ground truth, background, and one noisy full raster for reference."""
    rng = np.random.default_rng(0)
    noisy = scene.field + rng.normal(0, scene.params.noise_sigma, scene.field.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in [
        (axes[0], scene.field, "ground-truth field"),
        (axes[1], scene.background, "background component"),
        (axes[2], noisy, f"full raster at noise {scene.params.noise_sigma:g}"),
    ]:
        im = ax.imshow(img, cmap=_FIELD_CMAP)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    if len(scene.defect_centers):
        axes[0].scatter(
            scene.defect_centers[:, 1],
            scene.defect_centers[:, 0],
            s=90,
            facecolors="none",
            edgecolors="w",
            linewidths=1.2,
            label="defects",
        )
        axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"{scene.params.field_kind} scene, grid {scene.grid}, "
        f"correlation length {scene.params.length_scale:g} px",
        fontsize=11,
    )
    _save(fig, path)


def plot_run(scene: ScanScene, run: RunResult, recon: np.ndarray, path: str | Path) -> None:
    """Truth with visited positions, reconstruction, and absolute error."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    g = scene.grid
    im0 = axes[0].imshow(scene.field, cmap=_FIELD_CMAP)
    sc = axes[0].scatter(
        run.order % g,
        run.order // g,
        c=np.arange(len(run.order)),
        cmap="autumn",
        s=8,
        linewidths=0,
    )
    axes[0].set_title(f"where it looked ({len(run.order)} measurements)", fontsize=10)
    fig.colorbar(sc, ax=axes[0], fraction=0.046, label="acquisition order")
    im1 = axes[1].imshow(recon, cmap=_FIELD_CMAP, vmin=im0.get_clim()[0], vmax=im0.get_clim()[1])
    axes[1].set_title("reconstruction", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    err = np.abs(recon - scene.field)
    im2 = axes[2].imshow(err, cmap="magma")
    axes[2].set_title(f"absolute error (rmse {np.sqrt(np.mean(err**2)):.3f})", fontsize=10)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(STRATEGY_LABELS.get(run.strategy, run.strategy), fontsize=11)
    _save(fig, path)


def hero_figure(
    scene: ScanScene,
    runs: dict[str, tuple[RunResult, np.ndarray]],
    curves: dict[str, Any],
    path: str | Path,
) -> None:
    """The sampling-trajectory panel.

    One column per strategy showing where it looked (points coloured by
    acquisition order on the true field) and the resulting error map, plus
    a wide final column with the error-versus-budget curves.

    Args:
        scene: The common scene.
        runs: Mapping strategy -> (run, reconstruction) at equal budgets.
        curves: The ``reconstruction`` benchmark JSON dictionary.
        path: Output path.
    """
    names = list(runs)
    n = len(names)
    fig = plt.figure(figsize=(3.1 * n + 4.6, 6.6))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1.0] * n + [1.55], hspace=0.16, wspace=0.12)
    g = scene.grid
    vmin, vmax = scene.field.min(), scene.field.max()
    for j, name in enumerate(names):
        run, recon = runs[name]
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(scene.field, cmap=_FIELD_CMAP, vmin=vmin, vmax=vmax)
        ax.scatter(
            run.order % g,
            run.order // g,
            c=np.arange(len(run.order)),
            cmap="autumn",
            s=7,
            linewidths=0,
        )
        ax.set_title(STRATEGY_LABELS.get(name, name), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("where it looked", fontsize=10)
        ax2 = fig.add_subplot(gs[1, j])
        err = np.abs(recon - scene.field)
        ax2.imshow(err, cmap="magma", vmin=0, vmax=max(0.5, float(err.max())))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel(f"rmse {np.sqrt(np.mean(err**2)):.3f}", fontsize=10)
        if j == 0:
            ax2.set_ylabel("absolute error", fontsize=10)
    axc = fig.add_subplot(gs[:, n])
    axc.yaxis.tick_right()
    axc.yaxis.set_label_position("right")
    budgets = curves["checkpoints"]
    for name, block in curves["strategies"].items():
        mean = np.array(block["mean"])
        std = np.array(block["std"])
        color = STRATEGY_COLORS.get(name, "k")
        axc.plot(budgets, mean, color=color, label=STRATEGY_LABELS.get(name, name), lw=1.8)
        axc.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.15, lw=0)
    axc.set_xlabel("measurements")
    axc.set_ylabel("reconstruction RMSE (units of background std)")
    axc.set_yscale("log")
    axc.legend(fontsize=8, frameon=False)
    axc.set_title(f"error vs budget, full raster = {scene.n_positions} points", fontsize=10)
    _style(axc)
    fig.suptitle(
        f"where each strategy spends {len(next(iter(runs.values()))[0].order)} measurements "
        f"on the same hidden field ({scene.n_positions}-point raster equivalent)",
        fontsize=12,
    )
    _save(fig, path)


def animate_run(scene: ScanScene, run: RunResult, path: str | Path, fps: int = 4) -> None:
    """Render an active run's snapshots into an animated GIF.

    Frames show the posterior mean with visited positions and the posterior
    uncertainty that drives the next choice.

    Args:
        scene: The scene the run was executed on.
        run: A run recorded with ``record_every > 0``.
        path: Output .gif path.
        fps: Playback frames per second.
    """
    if not run.snapshots:
        raise ValueError("run has no snapshots; execute with record_every > 0")
    from PIL import Image

    frames = []
    g = scene.grid
    vmin, vmax = scene.field.min(), scene.field.max()
    smax = max(float(s.max()) for _, _, s in run.snapshots)
    for step, mu, sd in run.snapshots:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
        axes[0].imshow(scene.field, cmap=_FIELD_CMAP, vmin=vmin, vmax=vmax)
        axes[0].set_title("hidden ground truth", fontsize=9)
        axes[1].imshow(mu, cmap=_FIELD_CMAP, vmin=vmin, vmax=vmax)
        pts = run.order[:step]
        axes[1].scatter(pts % g, pts // g, s=4, c="w", linewidths=0, alpha=0.8)
        axes[1].scatter([pts[-1] % g], [pts[-1] // g], s=45, facecolors="none", edgecolors="r")
        axes[1].set_title(f"posterior mean, {step} measurements", fontsize=9)
        axes[2].imshow(sd, cmap="magma", vmin=0, vmax=smax)
        axes[2].set_title("posterior std (drives next choice)", fontsize=9)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(STRATEGY_LABELS.get(run.strategy, run.strategy), fontsize=10)
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def plot_curves(result: dict[str, Any], path: str | Path, title: str) -> None:
    """Error-versus-budget curves with across-seed bands, log-y."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    budgets = result["checkpoints"]
    for name, block in result["strategies"].items():
        mean = np.array(block["mean"])
        std = np.array(block["std"])
        color = STRATEGY_COLORS.get(name, "k")
        ax.plot(budgets, mean, color=color, label=STRATEGY_LABELS.get(name, name), lw=1.8)
        ax.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("measurements")
    ax.set_ylabel("reconstruction RMSE (units of background std)")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    _style(ax)
    _save(fig, path)


def plot_defect_curves(result: dict[str, Any], path: str | Path, title: str) -> None:
    """Defects found versus budget with across-seed bands."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    budgets = result["checkpoints"]
    n_def = result["config"]["scene"]["n_defects"]
    for name, block in result["strategies"].items():
        mean = np.array(block["mean"])
        std = np.array(block["std"])
        color = STRATEGY_COLORS.get(name, "k")
        ax.plot(budgets, mean, color=color, label=STRATEGY_LABELS.get(name, name), lw=1.8)
        ax.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.15, lw=0)
    ax.axhline(n_def, color="k", lw=0.8, ls=":", label=f"all {n_def} defects")
    ax.set_xlabel("measurements")
    ax.set_ylabel("defects found")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    _style(ax)
    _save(fig, path)


def plot_sweep(
    result: dict[str, Any],
    path: str | Path,
    x_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    logy: bool = False,
    overlay: tuple[list[float], list[float], str] | None = None,
) -> None:
    """Generic per-strategy sweep plot (noise sweep, sparsity sweep).

    Args:
        result: Benchmark JSON dictionary.
        path: Output path.
        x_key: Key of the x-axis values in the result.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        logy: Log-scale the y axis.
        overlay: Optional (xs, ys, label) reference line drawn dashed black,
            e.g. a geometric prediction to compare the measured curves with.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = result[x_key]
    if overlay is not None:
        ox, oy, olabel = overlay
        ax.plot(ox, oy, color="k", ls="--", lw=1.3, label=olabel, zorder=1)
    for name, block in result["strategies"].items():
        mean = np.array(block["mean"])
        std = np.array(block["std"])
        color = STRATEGY_COLORS.get(name, "k")
        ax.errorbar(
            xs,
            mean,
            yerr=std,
            color=color,
            label=STRATEGY_LABELS.get(name, name),
            lw=1.8,
            marker="o",
            ms=4,
            capsize=3,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    _style(ax)
    _save(fig, path)


def plot_misspecification(result: dict[str, Any], path: str | Path) -> None:
    """Grouped bars: RMSE per lengthscale pinning, per strategy."""
    cols = result["columns"]
    names = list(result["strategies"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.8 / len(names)
    x = np.arange(len(cols))
    for i, name in enumerate(names):
        mean = np.array(result["strategies"][name]["mean"])
        std = np.array(result["strategies"][name]["std"])
        ax.bar(
            x + (i - (len(names) - 1) / 2) * width,
            mean,
            width=width,
            yerr=std,
            capsize=3,
            color=STRATEGY_COLORS.get(name, "k"),
            label=STRATEGY_LABELS.get(name, name),
        )
    labels = [
        f"{c[len('pinned_'):-1]} x true lengthscale" if c.startswith("pinned_") else "fitted"
        for c in cols
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("reconstruction RMSE (units of background std)")
    ax.set_title(
        f"surrogate misspecification at {result['budget']} measurements "
        "(design and reconstruction share the pinned lengthscale)",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    _save(fig, path)


def plot_fairness(result: dict[str, Any], path: str | Path) -> None:
    """Grouped bars: RMSE per reconstructor, per strategy."""
    recons = result["reconstructors"]
    names = list(result["strategies"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.8 / len(names)
    x = np.arange(len(recons))
    for i, name in enumerate(names):
        mean = np.array(result["strategies"][name]["mean"])
        std = np.array(result["strategies"][name]["std"])
        ax.bar(
            x + (i - (len(names) - 1) / 2) * width,
            mean,
            width=width,
            yerr=std,
            capsize=3,
            color=STRATEGY_COLORS.get(name, "k"),
            label=STRATEGY_LABELS.get(name, name),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(recons, fontsize=9)
    ax.set_ylabel("reconstruction RMSE (units of background std)")
    ax.set_title(
        f"design ranking under three reconstructors at {result['budget']} measurements",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    _save(fig, path)
