"""Turn a set of measurements into a full property map.

The benchmark scores designs, not reconstructors, so every strategy is
scored through the same reconstructor. The default is the GP posterior
mean with hyperparameters fitted to that strategy's own data; a cubic
interpolation reconstructor and an oracle-hyperparameter GP are provided
for the reconstructor-robustness (fairness) check.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from .gp import GP, GPHyperparams
from .sim import ScanScene


def gp_reconstruct(
    scene: ScanScene,
    order: np.ndarray,
    values: np.ndarray,
    kernel: str = "rbf",
    hyper: GPHyperparams | None = None,
    fixed_lengthscale: float | None = None,
) -> tuple[np.ndarray, GPHyperparams]:
    """GP posterior-mean map from measurements.

    Args:
        scene: Scene providing the grid geometry (ground truth is not read).
        order: Flat indices of measured positions.
        values: Measured values.
        kernel: GP kernel name.
        hyper: If given, use these hyperparameters instead of fitting.
        fixed_lengthscale: If fitting, pin the lengthscale to this value.

    Returns:
        Tuple of the (grid, grid) reconstruction and the hyperparameters
        used.
    """
    coords = scene.coords()[order]
    gp = GP(kernel, hyper)
    if hyper is None:
        gp.fit_hyperparams(coords, values, fixed_lengthscale=fixed_lengthscale)
    gp.fit(coords, values)
    mean, _ = gp.predict(scene.coords())
    assert gp.hyper is not None
    return mean.reshape(scene.grid, scene.grid), gp.hyper


def interp_reconstruct(scene: ScanScene, order: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Cubic scattered-data interpolation with nearest-neighbour fill.

    A deliberately GP-free reconstructor used to check that design rankings
    do not depend on scoring designs through the same model family that the
    active strategies optimise.
    """
    pts = scene.coords()[order]
    target = scene.coords()
    est = griddata(pts, values, target, method="cubic")
    holes = np.isnan(est)
    if holes.any():
        est[holes] = griddata(pts, values, target[holes], method="nearest")
    return est.reshape(scene.grid, scene.grid)
