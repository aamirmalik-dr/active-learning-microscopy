"""activescan: simulated autonomous-microscopy experiments.

Where should a microscope measure next? This package simulates samples
with exact ground truth, provides space-filling baselines and Gaussian-
process-driven active strategies, and benchmarks reconstruction error and
defect discovery against the measurement budget.
"""

from .benchmark import MODES, run_config
from .gp import GP, KERNELS, GPHyperparams, SequentialGP, kernel_matrix
from .io import load_external, load_run, load_scene, save_run, save_scene
from .metrics import (
    defect_hit_steps,
    defects_found_curve,
    lattice_coverage_fraction,
    mae,
    measurements_to_target,
    rmse,
)
from .reconstruct import gp_reconstruct, interp_reconstruct
from .sim import FIELD_KINDS, ScanScene, SceneParams, make_scene
from .strategies import (
    ACTIVE,
    BASELINES,
    STRATEGIES,
    RunResult,
    lhs_design,
    random_design,
    raster_design,
    run_strategy,
)

__all__ = [
    "ACTIVE",
    "BASELINES",
    "FIELD_KINDS",
    "GP",
    "GPHyperparams",
    "KERNELS",
    "MODES",
    "RunResult",
    "STRATEGIES",
    "ScanScene",
    "SceneParams",
    "SequentialGP",
    "defect_hit_steps",
    "defects_found_curve",
    "gp_reconstruct",
    "interp_reconstruct",
    "kernel_matrix",
    "lattice_coverage_fraction",
    "lhs_design",
    "load_external",
    "load_run",
    "load_scene",
    "mae",
    "make_scene",
    "measurements_to_target",
    "random_design",
    "raster_design",
    "rmse",
    "run_config",
    "run_strategy",
    "save_run",
    "save_scene",
]

__version__ = "0.1.0"
