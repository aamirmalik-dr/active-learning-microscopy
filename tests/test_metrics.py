"""Tests for scoring functions."""

import numpy as np
import pytest

from activescan import (
    SceneParams,
    defect_hit_steps,
    defects_found_curve,
    mae,
    make_scene,
    measurements_to_target,
    rmse,
)


def test_rmse_mae_basics():
    a = np.zeros((4, 4))
    b = np.full((4, 4), 2.0)
    assert rmse(a, b) == pytest.approx(2.0)
    assert mae(a, b) == pytest.approx(2.0)
    assert rmse(a, a) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse(np.zeros((3, 3)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        mae(np.zeros((3, 3)), np.zeros((4, 4)))


def test_measurements_to_target_interpolates():
    budgets = np.array([100, 200, 300])
    errors = np.array([0.4, 0.2, 0.1])
    assert measurements_to_target(budgets, errors, 0.3) == pytest.approx(150.0)
    assert measurements_to_target(budgets, errors, 0.5) == pytest.approx(100.0)
    assert measurements_to_target(budgets, errors, 0.05) is None


def test_defect_hit_steps_geometry():
    scene = make_scene(SceneParams(grid=64, n_defects=3, defect_sigma=2.0, seed=1))
    g = scene.grid
    cy, cx = scene.defect_centers[0]
    hit_idx = int(round(cy)) * g + int(round(cx))
    far_idx = 0 if hit_idx != 0 else 1
    order = np.array([far_idx, hit_idx])
    steps = defect_hit_steps(scene, order)
    assert steps[0] == 2  # 1-based step of the direct hit
    assert (steps[1:] == -1).all()


def test_defect_hit_respects_core_radius():
    scene = make_scene(SceneParams(grid=64, n_defects=1, defect_sigma=2.0, seed=2))
    g = scene.grid
    cy, cx = scene.defect_centers[0]
    r = scene.defect_core_radius()
    outside = int(round(cy)) * g + int(round(cx + r + 2.0))
    steps = defect_hit_steps(scene, np.array([outside]))
    assert steps[0] == -1


def test_defects_found_curve_monotone():
    scene = make_scene(SceneParams(grid=64, n_defects=5, seed=3))
    g = scene.grid
    order = np.array([int(round(cy)) * g + int(round(cx)) for cy, cx in scene.defect_centers])
    curve = defects_found_curve(scene, order, np.array([1, 3, 5]))
    assert list(curve) == [1, 3, 5]
    assert (np.diff(curve) >= 0).all()


def test_no_defects_gives_empty_metrics():
    scene = make_scene(SceneParams(grid=32, n_defects=0, seed=0))
    assert defect_hit_steps(scene, np.array([0, 1])).size == 0
