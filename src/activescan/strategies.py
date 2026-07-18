"""Measurement-placement strategies: space-filling baselines and active learners.

Every strategy sees the same interface: a scene it may query only through
noisy measurements, a measurement budget, and a seed. Baselines commit to
their design up front; active strategies maintain a :class:`~activescan.gp.SequentialGP`
surrogate and choose each next position by maximising an acquisition
function over the unmeasured grid positions. No strategy ever reads the
ground-truth field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from .gp import GPHyperparams, SequentialGP
from .sim import ScanScene

BASELINES = ("random", "lhs", "raster")
ACTIVE = ("active_variance", "active_gradient", "active_hunt")
STRATEGIES = BASELINES + ACTIVE

#: Strategies whose budget-b prefix is itself the budget-b design.
NESTED = {"random": True, "lhs": False, "raster": True} | {s: True for s in ACTIVE}


@dataclass
class RunResult:
    """One completed measurement run.

    Attributes:
        strategy: Strategy name.
        order: Flat grid indices in acquisition order.
        values: Measured values in the same order.
        snapshots: Optional list of (step, posterior_mean, posterior_std)
            grids recorded during an active run, for animation.
        hyper_history: Hyperparameters after each refit of an active run.
    """

    strategy: str
    order: np.ndarray
    values: np.ndarray
    snapshots: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    hyper_history: list[GPHyperparams] = field(default_factory=list)


def random_design(scene: ScanScene, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random design without replacement."""
    return rng.choice(scene.n_positions, size=budget, replace=False)


def lhs_design(scene: ScanScene, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube design snapped to unmeasured grid positions.

    One stratified sample per axis (a random permutation pairing), jittered
    within strata, then snapped to the nearest free grid cell.

    Args:
        scene: Scene providing the grid geometry.
        budget: Number of points.
        rng: RNG for the permutation, jitter, and collision resolution.

    Returns:
        Flat indices of the design (length ``budget``, unique).
    """
    g = scene.grid
    strata_y = (np.arange(budget) + rng.uniform(size=budget)) / budget
    strata_x = (rng.permutation(budget) + rng.uniform(size=budget)) / budget
    rows = np.clip((strata_y * g).astype(int), 0, g - 1)
    cols = np.clip((strata_x * g).astype(int), 0, g - 1)
    idx = rows * g + cols
    taken = set()
    out = []
    for i in idx:
        j = int(i)
        while j in taken:
            j = int(rng.integers(scene.n_positions))
        taken.add(j)
        out.append(j)
    return np.array(out)


def raster_design(scene: ScanScene, budget: int) -> np.ndarray:
    """Coarse-to-fine raster: nested row-major grids at halving strides.

    The budget-b prefix is always a coarse raster refined as far as the
    budget allows, which is how a raster scan is actually made progressive;
    a naive top-to-bottom raster at full resolution would only ever image a
    band at the top of the field of view.
    """
    g = scene.grid
    stride = 1 << (g - 1).bit_length() - 1  # largest power of two < g
    seen: set[int] = set()
    order: list[int] = []
    while stride >= 1:
        for r in range(0, g, stride):
            for c in range(0, g, stride):
                i = r * g + c
                if i not in seen:
                    seen.add(i)
                    order.append(i)
        stride //= 2
    if budget > len(order):
        raise ValueError(f"budget {budget} exceeds grid positions {len(order)}")
    return np.array(order[:budget])


def _grad_weight(mu_grid: np.ndarray) -> np.ndarray:
    """Normalised, lightly smoothed gradient magnitude of the posterior mean."""
    gy, gx = np.gradient(mu_grid)
    mag = gaussian_filter(np.hypot(gy, gx), sigma=1.0)
    top = mag.max()
    return mag / top if top > 1e-12 else np.zeros_like(mag)


def _hunt_threshold(values: np.ndarray, k_tau: float) -> float:
    """Robust anomaly threshold: median + k_tau x scaled MAD of observations."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + k_tau * 1.4826 * max(mad, 1e-9)


def _acquisition(name: str, seq: SequentialGP, grid: int, k_tau: float) -> np.ndarray:
    if name == "active_variance":
        return seq.var.copy()
    if name == "active_gradient":
        w = _grad_weight(seq.mu.reshape(grid, grid)).ravel()
        return seq.var * (0.1 + w)
    if name == "active_hunt":
        tau = _hunt_threshold(np.asarray(seq.obs_y), k_tau)
        sigma = np.sqrt(seq.var)
        z = (seq.mu - tau) / sigma
        return (seq.mu - tau) * norm.cdf(z) + sigma * norm.pdf(z)
    raise ValueError(f"unknown active strategy {name!r}")


def run_strategy(
    scene: ScanScene,
    strategy: str,
    budget: int,
    seed: int,
    n_init: int = 16,
    refit_every: int = 25,
    kernel: str = "rbf",
    fixed_lengthscale: float | None = None,
    k_tau: float = 3.0,
    hunt_exclusion_px: float = 3.0,
    record_every: int = 0,
) -> RunResult:
    """Execute one strategy on one scene and return the measurement run.

    Args:
        scene: The (hidden) sample; queried only through noisy measurements.
        strategy: One of :data:`STRATEGIES`.
        budget: Total number of measurements, including initialisation.
        seed: Seed for measurement noise and any strategy randomness.
        n_init: Latin-hypercube initialisation size for active strategies.
        refit_every: Refit surrogate hyperparameters every this many steps.
        kernel: Surrogate kernel for active strategies.
        fixed_lengthscale: If set, the surrogate lengthscale is pinned to
            this value (normalised units) instead of being fitted; this is
            the misspecified-surrogate mode.
        k_tau: Robust-threshold multiplier for the hunt acquisition.
        hunt_exclusion_px: Radius (px) of the found-and-move-on exclusion
            around measurements that exceeded the hunt threshold.
        record_every: If > 0, store posterior snapshots at this cadence.

    Returns:
        A :class:`RunResult` with indices and values in acquisition order.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if budget > scene.n_positions:
        raise ValueError("budget exceeds number of grid positions")
    rng = np.random.default_rng(seed)
    if strategy in BASELINES:
        if strategy == "random":
            order = random_design(scene, budget, rng)
        elif strategy == "lhs":
            order = lhs_design(scene, budget, rng)
        else:
            order = raster_design(scene, budget)
        values = scene.measure(order, rng)
        return RunResult(strategy=strategy, order=order, values=values)

    n_init = min(n_init, budget)
    init_idx = lhs_design(scene, n_init, rng)
    init_val = scene.measure(init_idx, rng)
    hyper0 = GPHyperparams(
        lengthscale=fixed_lengthscale if fixed_lengthscale is not None else 0.1,
        signal_var=max(float(np.var(init_val)), 0.1),
        noise_var=max(0.1 * float(np.var(init_val)), 1e-4),
    )
    seq = SequentialGP(scene.coords(), kernel, hyper0)
    for i, v in zip(init_idx, init_val):
        seq.add(int(i), float(v))
    result = RunResult(strategy=strategy, order=init_idx.copy(), values=init_val.copy())
    if n_init >= 4:
        result.hyper_history.append(seq.refit(fixed_lengthscale=fixed_lengthscale))
    measured = np.zeros(scene.n_positions, dtype=bool)
    measured[init_idx] = True
    g = scene.grid
    all_rows = np.arange(scene.n_positions) // g
    all_cols = np.arange(scene.n_positions) % g
    for step in range(n_init, budget):
        score = _acquisition(strategy, seq, g, k_tau)
        score[measured] = -np.inf
        if strategy == "active_hunt":
            # found-and-move-on: once a measured value exceeds the current
            # robust threshold, that anomaly is confirmed, so suppress the
            # acquisition in its immediate neighbourhood instead of spending
            # further budget re-measuring a known defect. Uses only measured
            # values, never ground truth.
            tau = _hunt_threshold(np.asarray(seq.obs_y), k_tau)
            obs_idx = np.asarray(seq.obs_idx)
            hot = obs_idx[np.asarray(seq.obs_y) > tau]
            for h in hot:
                d2 = (all_rows - h // g) ** 2 + (all_cols - h % g) ** 2
                score[d2 <= hunt_exclusion_px**2] = -np.inf
            if not np.isfinite(score).any():
                score = seq.var.copy()
                score[measured] = -np.inf
        idx = int(np.argmax(score))
        value = float(scene.measure(np.array([idx]), rng)[0])
        seq.add(idx, value)
        measured[idx] = True
        result.order = np.append(result.order, idx)
        result.values = np.append(result.values, value)
        steps_done = step + 1
        if steps_done < budget and steps_done % refit_every == 0:
            result.hyper_history.append(seq.refit(fixed_lengthscale=fixed_lengthscale))
        if record_every and (steps_done % record_every == 0 or steps_done == budget):
            result.snapshots.append(
                (steps_done, seq.mu.reshape(g, g).copy(), np.sqrt(seq.var).reshape(g, g))
            )
    return result
