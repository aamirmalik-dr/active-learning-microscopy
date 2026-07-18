"""Saving and loading scenes and runs, plus the bring-your-own-data path.

External data enters as a fully acquired 2D map (any square image of a
measured property). :func:`load_external` wraps it as a scene so every
strategy can be replayed against it: the map acts as the measurement
oracle, which answers the practical question "how many measurements of my
own sample would this strategy have needed". The wrapped map has no defect
ground truth, so only reconstruction metrics apply.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .sim import ScanScene, SceneParams
from .strategies import RunResult


def save_scene(path: str | Path, scene: ScanScene) -> None:
    """Write a scene with its full ground truth to an .npz file."""
    np.savez_compressed(
        path,
        field=scene.field.astype(np.float32),
        background=scene.background.astype(np.float32),
        defect_centers=scene.defect_centers.astype(np.float64),
        **{f"param_{k}": v for k, v in asdict(scene.params).items()},
    )


def load_scene(path: str | Path) -> ScanScene:
    """Load a scene saved by :func:`save_scene`."""
    with np.load(path, allow_pickle=False) as npz:
        raw = {k[len("param_") :]: npz[k][()] for k in npz.files if k.startswith("param_")}
        params = SceneParams(
            grid=int(raw["grid"]),
            length_scale=float(raw["length_scale"]),
            field_kind=str(raw["field_kind"]),
            n_grains=int(raw["n_grains"]),
            n_defects=int(raw["n_defects"]),
            defect_amplitude=float(raw["defect_amplitude"]),
            defect_sigma=float(raw["defect_sigma"]),
            noise_sigma=float(raw["noise_sigma"]),
            seed=int(raw["seed"]),
        )
        return ScanScene(
            field=npz["field"].astype(float),
            background=npz["background"].astype(float),
            defect_centers=npz["defect_centers"].astype(float),
            params=params,
        )


def load_external(
    source: str | Path | np.ndarray, noise_sigma: float = 0.1, normalize: bool = True
) -> ScanScene:
    """Wrap an externally acquired 2D map as a replayable scene.

    Args:
        source: Path to a .npy/.npz file holding one 2D array, or the
            array itself. Must be square (crop before loading otherwise).
        noise_sigma: Noise std applied when strategies re-measure the map,
            in units of the (normalised) map std. Use 0 to replay the map
            exactly as recorded.
        normalize: If True, shift/scale the map to zero mean and unit std
            so thresholds and noise levels mean the same as in simulation.

    Returns:
        A ScanScene whose "ground truth" is the supplied map and which has
        no defect annotations.
    """
    if isinstance(source, (str, Path)):
        loaded = np.load(source, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if len(loaded.files) != 1:
                raise ValueError(
                    f"expected exactly one array in {source}, found {list(loaded.files)}"
                )
            arr = loaded[loaded.files[0]]
        else:
            arr = loaded
    else:
        arr = np.asarray(source)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square 2D map, got shape {arr.shape}")
    if normalize:
        std = arr.std()
        if std < 1e-12:
            raise ValueError("map is constant; nothing to scan")
        arr = (arr - arr.mean()) / std
    params = SceneParams(grid=arr.shape[0], noise_sigma=noise_sigma, n_defects=0)
    return ScanScene(
        field=arr, background=arr.copy(), defect_centers=np.zeros((0, 2)), params=params
    )


def save_run(path: str | Path, run: RunResult) -> None:
    """Write a measurement run (indices and values) to an .npz file."""
    np.savez_compressed(
        path,
        strategy=np.array(run.strategy),
        order=run.order.astype(np.int64),
        values=run.values.astype(np.float64),
    )


def load_run(path: str | Path) -> RunResult:
    """Load a run saved by :func:`save_run` (snapshots are not persisted)."""
    with np.load(path, allow_pickle=False) as npz:
        return RunResult(
            strategy=str(npz["strategy"][()]),
            order=npz["order"].astype(int),
            values=npz["values"].astype(float),
        )
