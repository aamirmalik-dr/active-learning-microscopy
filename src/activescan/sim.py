"""Simulated samples for autonomous-microscopy experiments.

A scene is a property field defined on a square grid of allowed probe
positions, with exact ground truth. The background varies on a length scale
much larger than one pixel (a slowly varying composition, thickness, or
strain signal); optional rare defects sit on top of it as narrow positive
bumps. A measurement visits one grid position and returns the field value
plus independent Gaussian noise of constant variance. That is the
high-count limit of shot noise at fixed dwell time, valid when the
property contrast is small next to the mean detected signal; true
signal-dependent (Poisson) noise is deliberately out of scope. A full
raster scan is one measurement at every grid position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

FIELD_KINDS = ("smooth", "grains")


@dataclass(frozen=True)
class SceneParams:
    """Parameters that fully determine a scene given a seed.

    Attributes:
        grid: Side length of the square grid of allowed probe positions.
        length_scale: Background correlation length in pixels.
        field_kind: "smooth" (stationary random field) or "grains"
            (Voronoi domains with sharp boundaries, non-stationary).
        n_grains: Number of Voronoi domains for the "grains" kind.
        n_defects: Number of point defects added on top of the background.
        defect_amplitude: Defect peak height in units of the background
            standard deviation (the background is normalised to unit std).
        defect_sigma: Defect radius parameter in pixels (Gaussian sigma).
        noise_sigma: Measurement noise std in units of background std.
        seed: Seed for the scene-generation RNG.
    """

    grid: int = 64
    length_scale: float = 10.0
    field_kind: str = "smooth"
    n_grains: int = 8
    n_defects: int = 0
    defect_amplitude: float = 5.0
    defect_sigma: float = 2.0
    noise_sigma: float = 0.3
    seed: int = 0


@dataclass
class ScanScene:
    """A simulated sample with exact ground truth.

    Attributes:
        field: (grid, grid) noiseless ground-truth field (background of
            unit std, plus any defect bumps).
        background: (grid, grid) background component alone.
        defect_centers: (n_defects, 2) float array of (row, col) centres.
        params: The generating parameters.
    """

    field: np.ndarray
    background: np.ndarray
    defect_centers: np.ndarray
    params: SceneParams

    @property
    def grid(self) -> int:
        return self.params.grid

    @property
    def n_positions(self) -> int:
        return self.params.grid**2

    def coords(self) -> np.ndarray:
        """Return (grid**2, 2) normalised (y, x) coordinates of all positions.

        Positions are pixel centres mapped into [0, 1] on each axis, in
        row-major (raster) order, matching ``field.ravel()``.
        """
        g = self.params.grid
        axis = (np.arange(g) + 0.5) / g
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        return np.stack([yy.ravel(), xx.ravel()], axis=1)

    def measure(self, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return noisy measurements at flat grid indices.

        Args:
            indices: Integer array of flat positions into ``field.ravel()``.
            rng: Generator supplying the measurement noise.

        Returns:
            Array of measured values, one per index.
        """
        idx = np.asarray(indices, dtype=int)
        clean = self.field.ravel()[idx]
        return clean + rng.normal(0.0, self.params.noise_sigma, size=idx.shape)

    def defect_core_radius(self) -> float:
        """Radius (px) where a defect bump exceeds half its amplitude."""
        return float(self.params.defect_sigma * np.sqrt(2.0 * np.log(2.0)))


def _smooth_background(g: int, length_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary Gaussian random field: filtered white noise, unit std."""
    white = rng.normal(size=(g, g))
    smooth = gaussian_filter(white, sigma=length_scale, mode="wrap")
    smooth -= smooth.mean()
    std = smooth.std()
    if std < 1e-12:
        raise ValueError("degenerate background field; reduce length_scale")
    return smooth / std


def _grain_background(
    g: int, length_scale: float, n_grains: int, rng: np.random.Generator
) -> np.ndarray:
    """Voronoi domains with per-grain offsets plus smooth interior variation.

    The result has sharp steps at domain boundaries, so it violates the
    stationarity assumption of a smooth global kernel on purpose.
    """
    seeds = rng.uniform(0, g, size=(n_grains, 2))
    yy, xx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    d2 = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    label = np.argmin(d2, axis=-1)
    offsets = rng.normal(0.0, 1.0, size=n_grains)
    fld = offsets[label] + 0.4 * _smooth_background(g, length_scale, rng)
    fld -= fld.mean()
    return fld / fld.std()


def _place_defects(g: int, n_defects: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Draw defect centres uniformly with a minimum mutual separation."""
    if n_defects == 0:
        return np.zeros((0, 2))
    margin = 2.0 * sigma
    min_sep = 6.0 * sigma
    centers: list[np.ndarray] = []
    for _ in range(10_000):
        cand = rng.uniform(margin, g - margin, size=2)
        if all(np.hypot(*(cand - c)) >= min_sep for c in centers):
            centers.append(cand)
            if len(centers) == n_defects:
                break
    else:
        raise ValueError(
            f"could not place {n_defects} defects with separation {min_sep:.1f} on a {g} grid"
        )
    return np.array(centers)


def make_scene(params: SceneParams) -> ScanScene:
    """Generate a scene deterministically from its parameters.

    Args:
        params: Scene parameters, including the seed.

    Returns:
        A ScanScene with unit-std background and any requested defects.
    """
    if params.field_kind not in FIELD_KINDS:
        raise ValueError(f"field_kind must be one of {FIELD_KINDS}, got {params.field_kind!r}")
    rng = np.random.default_rng(params.seed)
    g = params.grid
    if params.field_kind == "smooth":
        background = _smooth_background(g, params.length_scale, rng)
    else:
        background = _grain_background(g, params.length_scale, params.n_grains, rng)
    centers = _place_defects(g, params.n_defects, params.defect_sigma, rng)
    fld = background.copy()
    if len(centers):
        yy, xx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        for cy, cx in centers:
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            fld += params.defect_amplitude * np.exp(-r2 / (2.0 * params.defect_sigma**2))
    return ScanScene(field=fld, background=background, defect_centers=centers, params=params)
