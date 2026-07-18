"""Config-driven, fixed-seed benchmarks.

Each YAML config names a mode, a scene, strategies, seeds, and budgets;
running it writes one JSON file under ``results/``. Modes:

* ``reconstruction``: error-versus-budget curves per strategy (also used
  for the non-stationary grain scene).
* ``noise_sweep``: error at a fixed budget across measurement-noise levels.
* ``defect_search``: defects-found-versus-budget curves per strategy.
* ``sparsity_sweep``: fraction of defects found at a fixed budget as the
  defect count varies.
* ``size_sweep``: fraction of defects found at a fixed budget as the
  defect size varies (the operating-point check for raster coverage).
* ``misspecification``: active design with a pinned (wrong) surrogate
  lengthscale against Latin-hypercube, both scored through the same
  pinned-lengthscale reconstructor.
* ``fairness``: one operating point scored under three different
  reconstructors, to check the strategy ranking is not an artifact of
  scoring designs through the surrogate's own model family.

Replicate seeds regenerate both the scene and the measurement noise, so
curves come with across-scene spread, not just noise spread.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .gp import GP
from .metrics import defect_hit_steps, defects_found_curve, measurements_to_target, rmse
from .reconstruct import gp_reconstruct, interp_reconstruct
from .sim import SceneParams, make_scene
from .strategies import NESTED, RunResult, lhs_design, run_strategy

MODES = (
    "reconstruction",
    "noise_sweep",
    "defect_search",
    "sparsity_sweep",
    "size_sweep",
    "misspecification",
    "fairness",
)

_MEASURE_SEED_OFFSET = 10_000


def _scene_params(cfg: dict[str, Any], seed: int, **overrides: Any) -> SceneParams:
    scene_cfg = dict(cfg.get("scene", {}))
    scene_cfg.update(overrides)
    scene_cfg["seed"] = seed
    return SceneParams(**scene_cfg)


def _run(cfg: dict[str, Any], scene, strategy: str, seed: int, budget: int, **kw) -> RunResult:
    return run_strategy(
        scene,
        strategy,
        budget,
        seed=_MEASURE_SEED_OFFSET + seed,
        n_init=int(cfg.get("n_init", 16)),
        refit_every=int(cfg.get("refit_every", 25)),
        kernel=str(cfg.get("kernel", "rbf")),
        **kw,
    )


def _curve_rmse(
    scene,
    run: RunResult,
    checkpoints: list[int],
    kernel: str,
    fixed_lengthscale: float | None = None,
) -> list[float]:
    """Reconstruct from each budget prefix of a nested run and score it."""
    out = []
    for b in checkpoints:
        recon, _ = gp_reconstruct(
            scene, run.order[:b], run.values[:b], kernel, fixed_lengthscale=fixed_lengthscale
        )
        out.append(rmse(recon, scene.field))
    return out


def _lhs_curve_rmse(
    scene,
    seed: int,
    checkpoints: list[int],
    kernel: str,
    fixed_lengthscale: float | None = None,
) -> list[float]:
    """LHS designs are not nested, so draw a fresh design per budget."""
    out = []
    for b in checkpoints:
        rng = np.random.default_rng(_MEASURE_SEED_OFFSET + seed * 1000 + b)
        order = lhs_design(scene, b, rng)
        values = scene.measure(order, rng)
        recon, _ = gp_reconstruct(scene, order, values, kernel, fixed_lengthscale=fixed_lengthscale)
        out.append(rmse(recon, scene.field))
    return out


def _summarise(per_seed: list[list[float]]) -> dict[str, list[float]]:
    arr = np.asarray(per_seed, dtype=float)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "per_seed": arr.tolist(),
    }


def _mode_reconstruction(cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoints = [int(b) for b in cfg["checkpoints"]]
    kernel = str(cfg.get("kernel", "rbf"))
    target = cfg.get("target_rmse")
    out: dict[str, Any] = {"checkpoints": checkpoints, "strategies": {}}
    for strategy in cfg["strategies"]:
        per_seed = []
        for seed in cfg["seeds"]:
            scene = make_scene(_scene_params(cfg, seed))
            if NESTED[strategy]:
                run = _run(cfg, scene, strategy, seed, max(checkpoints))
                per_seed.append(_curve_rmse(scene, run, checkpoints, kernel))
            else:
                per_seed.append(_lhs_curve_rmse(scene, seed, checkpoints, kernel))
        summary = _summarise(per_seed)
        if target is not None:
            budgets_needed = [
                measurements_to_target(np.array(checkpoints), np.array(row), float(target))
                for row in per_seed
            ]
            summary["budget_to_target"] = [None if b is None else float(b) for b in budgets_needed]
        out["strategies"][strategy] = summary
    return out


def _mode_noise_sweep(cfg: dict[str, Any]) -> dict[str, Any]:
    budget = int(cfg["budget"])
    kernel = str(cfg.get("kernel", "rbf"))
    noise_sigmas = [float(s) for s in cfg["noise_sigmas"]]
    out: dict[str, Any] = {"noise_sigmas": noise_sigmas, "budget": budget, "strategies": {}}
    for strategy in cfg["strategies"]:
        per_seed = []
        for seed in cfg["seeds"]:
            row = []
            for sigma in noise_sigmas:
                scene = make_scene(_scene_params(cfg, seed, noise_sigma=sigma))
                if NESTED[strategy]:
                    run = _run(cfg, scene, strategy, seed, budget)
                    order, values = run.order, run.values
                else:
                    rng = np.random.default_rng(_MEASURE_SEED_OFFSET + seed)
                    order = lhs_design(scene, budget, rng)
                    values = scene.measure(order, rng)
                recon, _ = gp_reconstruct(scene, order, values, kernel)
                row.append(rmse(recon, scene.field))
            per_seed.append(row)
        out["strategies"][strategy] = _summarise(per_seed)
    return out


def _mode_defect_search(cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoints = [int(b) for b in cfg["checkpoints"]]
    budget = max(checkpoints)
    out: dict[str, Any] = {"checkpoints": checkpoints, "strategies": {}}
    n_defects = int(cfg.get("scene", {}).get("n_defects", 0))
    for strategy in cfg["strategies"]:
        per_seed = []
        hit_steps: list[int] = []
        for seed in cfg["seeds"]:
            scene = make_scene(_scene_params(cfg, seed))
            if NESTED[strategy]:
                run = _run(cfg, scene, strategy, seed, budget)
                order = run.order
            else:
                rng = np.random.default_rng(_MEASURE_SEED_OFFSET + seed)
                order = lhs_design(scene, budget, rng)
            per_seed.append(defects_found_curve(scene, order, np.array(checkpoints)).tolist())
            hit_steps.extend(int(s) for s in defect_hit_steps(scene, order))
        summary = _summarise(per_seed)
        found = [s for s in hit_steps if s > 0]
        summary["n_defects_total"] = n_defects * len(cfg["seeds"])
        summary["n_found_at_budget"] = len(found)
        summary["median_hit_step"] = float(np.median(found)) if found else None
        out["strategies"][strategy] = summary
    return out


def _mode_sparsity_sweep(cfg: dict[str, Any]) -> dict[str, Any]:
    budget = int(cfg["budget"])
    counts = [int(n) for n in cfg["defect_counts"]]
    out: dict[str, Any] = {"defect_counts": counts, "budget": budget, "strategies": {}}
    for strategy in cfg["strategies"]:
        per_seed = []
        for seed in cfg["seeds"]:
            row = []
            for n_d in counts:
                scene = make_scene(_scene_params(cfg, seed, n_defects=n_d))
                if NESTED[strategy]:
                    run = _run(cfg, scene, strategy, seed, budget)
                    order = run.order
                else:
                    rng = np.random.default_rng(_MEASURE_SEED_OFFSET + seed)
                    order = lhs_design(scene, budget, rng)
                found = defects_found_curve(scene, order, np.array([budget]))[0]
                row.append(found / n_d)
            per_seed.append(row)
        out["strategies"][strategy] = _summarise(per_seed)
    return out


def _mode_size_sweep(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fraction of defects found at a fixed budget as the defect size varies.

    This is the operating-point check for the raster-versus-hunt comparison:
    a coarse-to-fine raster at budget b guarantees coverage down to a core
    radius set by its finest completed stride, so its parity with the hunt
    acquisition can be a geometric coincidence of one defect size.
    """
    budget = int(cfg["budget"])
    sigmas = [float(s) for s in cfg["defect_sigmas"]]
    out: dict[str, Any] = {"defect_sigmas": sigmas, "budget": budget, "strategies": {}}
    for strategy in cfg["strategies"]:
        per_seed = []
        for seed in cfg["seeds"]:
            row = []
            for sig in sigmas:
                scene = make_scene(_scene_params(cfg, seed, defect_sigma=sig))
                if NESTED[strategy]:
                    run = _run(cfg, scene, strategy, seed, budget)
                    order = run.order
                else:
                    rng = np.random.default_rng(_MEASURE_SEED_OFFSET + seed)
                    order = lhs_design(scene, budget, rng)
                found = defects_found_curve(scene, order, np.array([budget]))[0]
                row.append(found / len(scene.defect_centers))
            per_seed.append(row)
        out["strategies"][strategy] = _summarise(per_seed)
    return out


def _mode_misspecification(cfg: dict[str, Any]) -> dict[str, Any]:
    budget = int(cfg["budget"])
    kernel = str(cfg.get("kernel", "rbf"))
    factors = [float(f) for f in cfg["lengthscale_factors"]]
    out: dict[str, Any] = {
        "lengthscale_factors": factors,
        "budget": budget,
        "strategies": {},
        "note": (
            "active designs use a surrogate whose lengthscale is pinned to "
            "factor x the true background lengthscale; both the active and "
            "the LHS design are scored through the same pinned-lengthscale "
            "reconstructor, so the comparison isolates the design decision. "
            "factor null means the lengthscale is fitted (well-specified reference)."
        ),
    }
    for strategy in cfg["strategies"]:
        per_seed = []
        for seed in cfg["seeds"]:
            scene = make_scene(_scene_params(cfg, seed))
            true_ls = scene.params.length_scale / scene.grid
            row = []
            for factor in factors:
                pinned = factor * true_ls
                if NESTED[strategy]:
                    run = _run(cfg, scene, strategy, seed, budget, fixed_lengthscale=pinned)
                    order, values = run.order, run.values
                else:
                    rng = np.random.default_rng(_MEASURE_SEED_OFFSET + seed)
                    order = lhs_design(scene, budget, rng)
                    values = scene.measure(order, rng)
                recon, _ = gp_reconstruct(scene, order, values, kernel, fixed_lengthscale=pinned)
                row.append(rmse(recon, scene.field))
            # fitted-lengthscale reference at the same budget
            if NESTED[strategy]:
                run = _run(cfg, scene, strategy, seed, budget)
                order, values = run.order, run.values
            recon, _ = gp_reconstruct(scene, order, values, kernel)
            row.append(rmse(recon, scene.field))
            per_seed.append(row)
        out["strategies"][strategy] = _summarise(per_seed)
    out["columns"] = [f"pinned_{f:g}x" for f in factors] + ["fitted"]
    return out


def _mode_fairness(cfg: dict[str, Any]) -> dict[str, Any]:
    budget = int(cfg["budget"])
    kernel = str(cfg.get("kernel", "rbf"))
    out: dict[str, Any] = {
        "budget": budget,
        "reconstructors": ["gp_fitted", "gp_reference_tuned", "cubic_interp"],
        "strategies": {},
        "note": (
            "gp_reference_tuned uses hyperparameters fitted once per scene on "
            "an independent 1000-point random sample, so no strategy's own "
            "data influences them; cubic_interp is model-free."
        ),
    }
    for strategy in cfg["strategies"]:
        per_seed = []
        for seed in cfg["seeds"]:
            scene = make_scene(_scene_params(cfg, seed))
            ref_rng = np.random.default_rng(_MEASURE_SEED_OFFSET + 777 + seed)
            n_ref = min(int(cfg.get("reference_points", 1000)), scene.n_positions // 2)
            ref_order = ref_rng.choice(scene.n_positions, size=n_ref, replace=False)
            ref_values = scene.measure(ref_order, ref_rng)
            ref_hyper = GP(kernel).fit_hyperparams(scene.coords()[ref_order], ref_values)
            if NESTED[strategy]:
                run = _run(cfg, scene, strategy, seed, budget)
                order, values = run.order, run.values
            else:
                rng = np.random.default_rng(_MEASURE_SEED_OFFSET + seed)
                order = lhs_design(scene, budget, rng)
                values = scene.measure(order, rng)
            recon_fit, _ = gp_reconstruct(scene, order, values, kernel)
            recon_ref, _ = gp_reconstruct(scene, order, values, kernel, hyper=ref_hyper)
            recon_cub = interp_reconstruct(scene, order, values)
            per_seed.append(
                [
                    rmse(recon_fit, scene.field),
                    rmse(recon_ref, scene.field),
                    rmse(recon_cub, scene.field),
                ]
            )
        out["strategies"][strategy] = _summarise(per_seed)
    return out


_DISPATCH = {
    "reconstruction": _mode_reconstruction,
    "noise_sweep": _mode_noise_sweep,
    "defect_search": _mode_defect_search,
    "sparsity_sweep": _mode_sparsity_sweep,
    "size_sweep": _mode_size_sweep,
    "misspecification": _mode_misspecification,
    "fairness": _mode_fairness,
}


def run_config(config_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    """Run one benchmark config and write its JSON result.

    Args:
        config_path: Path to a YAML config with at least ``mode`` and ``out``.
        out_path: Override for the output JSON path.

    Returns:
        The result dictionary that was written.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mode = cfg.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    t0 = time.perf_counter()
    result = _DISPATCH[mode](cfg)
    result["mode"] = mode
    result["config"] = cfg
    result["wall_time_s"] = round(time.perf_counter() - t0, 1)
    dest = Path(out_path) if out_path is not None else config_path.parent.parent / cfg["out"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result
