"""Scoring against exact ground truth.

Reconstruction quality is root-mean-square error over the full grid, in
units of the background standard deviation (the background is normalised
to unit std, so an RMSE of 0.3 means the map is wrong by 30 percent of the
natural field variation on average). Defect search is scored geometrically:
a defect counts as found at the first step a measurement lands inside its
core, the radius where the bump still exceeds half its amplitude, so the
score does not depend on any detector threshold.
"""

from __future__ import annotations

import numpy as np

from .sim import ScanScene


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error between two maps of equal shape."""
    a = np.asarray(estimate, dtype=float)
    b = np.asarray(truth, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error between two maps of equal shape."""
    a = np.asarray(estimate, dtype=float)
    b = np.asarray(truth, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)))


def measurements_to_target(budgets: np.ndarray, errors: np.ndarray, target: float) -> float | None:
    """Smallest budget at which the error curve first reaches a target.

    Linear interpolation between the bracketing checkpoints; None if the
    curve never reaches the target within the measured range.

    Args:
        budgets: Increasing budgets at which the error was evaluated.
        errors: Error at each budget.
        target: Error level to reach.

    Returns:
        Interpolated budget, or None.
    """
    budgets = np.asarray(budgets, dtype=float)
    errors = np.asarray(errors, dtype=float)
    below = np.nonzero(errors <= target)[0]
    if len(below) == 0:
        return None
    j = int(below[0])
    if j == 0:
        return float(budgets[0])
    b0, b1 = budgets[j - 1], budgets[j]
    e0, e1 = errors[j - 1], errors[j]
    if e0 == e1:
        return float(b1)
    frac = (e0 - target) / (e0 - e1)
    return float(b0 + frac * (b1 - b0))


def lattice_coverage_fraction(stride: float, radius: float, n: int = 1200) -> float:
    """Fraction of the plane within a radius of a square lattice of points.

    This is the geometric hit probability for a defect whose core radius is
    ``radius`` against a completed raster pass of the given stride, assuming
    the defect centre falls uniformly. It reaches 1.0 exactly at
    ``radius >= stride / sqrt(2)`` (the half-diagonal of a lattice cell),
    which is the only radius at which a raster pass carries a guarantee.

    Args:
        stride: Lattice spacing in pixels.
        radius: Capture radius in pixels (e.g. the defect core radius).
        n: Grid resolution per cell axis for the numerical estimate.

    Returns:
        Covered area fraction in [0, 1].
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    if radius >= stride * np.sqrt(0.5):
        return 1.0
    axis = (np.arange(n) + 0.5) / n * stride
    xx, yy = np.meshgrid(axis, axis)
    corners = [(0.0, 0.0), (0.0, stride), (stride, 0.0), (stride, stride)]
    d = np.minimum.reduce([np.hypot(xx - cx, yy - cy) for cy, cx in corners])
    return float(np.mean(d <= radius))


def defect_hit_steps(scene: ScanScene, order: np.ndarray) -> np.ndarray:
    """First acquisition step (1-based) at which each defect was hit.

    A hit is a measurement whose position lies within the defect core
    radius of the defect centre. Unfound defects get -1.

    Args:
        scene: Scene with defect ground truth.
        order: Flat measurement indices in acquisition order.

    Returns:
        Integer array of length n_defects.
    """
    if len(scene.defect_centers) == 0:
        return np.zeros(0, dtype=int)
    g = scene.grid
    rows = order // g
    cols = order % g
    radius = scene.defect_core_radius()
    steps = np.full(len(scene.defect_centers), -1, dtype=int)
    for k, (cy, cx) in enumerate(scene.defect_centers):
        dist = np.hypot(rows - cy, cols - cx)
        hits = np.nonzero(dist <= radius)[0]
        if len(hits):
            steps[k] = int(hits[0]) + 1
    return steps


def defects_found_curve(scene: ScanScene, order: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
    """Number of distinct defects found by each budget checkpoint."""
    steps = defect_hit_steps(scene, order)
    found = steps[steps > 0]
    return np.array([int(np.sum(found <= b)) for b in checkpoints])
