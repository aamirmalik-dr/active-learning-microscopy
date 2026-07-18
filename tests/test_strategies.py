"""Tests for baseline designs and active strategies."""

import numpy as np
import pytest

from activescan import (
    STRATEGIES,
    SceneParams,
    lhs_design,
    make_scene,
    raster_design,
    run_strategy,
)


@pytest.fixture(scope="module")
def scene():
    return make_scene(SceneParams(grid=32, length_scale=6, noise_sigma=0.3, seed=0))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_runs_are_valid_and_deterministic(scene, strategy):
    a = run_strategy(scene, strategy, 40, seed=5, n_init=8, refit_every=16)
    b = run_strategy(scene, strategy, 40, seed=5, n_init=8, refit_every=16)
    assert len(a.order) == 40
    assert len(np.unique(a.order)) == 40
    assert a.order.min() >= 0 and a.order.max() < scene.n_positions
    assert np.array_equal(a.order, b.order)
    assert np.allclose(a.values, b.values)


def test_unknown_strategy_raises(scene):
    with pytest.raises(ValueError):
        run_strategy(scene, "oracle", 10, seed=0)


def test_budget_exceeding_grid_raises(scene):
    with pytest.raises(ValueError):
        run_strategy(scene, "random", scene.n_positions + 1, seed=0)


def test_lhs_is_stratified(scene):
    rng = np.random.default_rng(0)
    n = 32
    idx = lhs_design(scene, n, rng)
    rows = idx // scene.grid
    # with 32 points on a 32-row grid, LHS puts about one point per row
    assert len(np.unique(rows)) >= n - 4


def test_raster_prefix_is_nested(scene):
    full = raster_design(scene, 200)
    short = raster_design(scene, 60)
    assert np.array_equal(full[:60], short)


def test_raster_starts_coarse(scene):
    order = raster_design(scene, 4)
    g = scene.grid
    rows, cols = order // g, order % g
    # the first pass is a stride-16 grid on a 32 grid: corners of coarse cells
    assert set(rows).issubset({0, 16}) and set(cols).issubset({0, 16})


def test_raster_full_budget_covers_everything(scene):
    order = raster_design(scene, scene.n_positions)
    assert len(np.unique(order)) == scene.n_positions


def test_active_records_snapshots_and_refits(scene):
    run = run_strategy(
        scene, "active_variance", 48, seed=0, n_init=8, refit_every=16, record_every=8
    )
    assert len(run.snapshots) >= 3
    step, mu, sd = run.snapshots[-1]
    assert step == 48
    assert mu.shape == (scene.grid, scene.grid)
    assert (sd >= 0).all()
    assert len(run.hyper_history) >= 2  # initial refit plus at least one scheduled


def test_active_variance_spreads_out(scene):
    """Variance acquisition should not clump: nearest-neighbour spacing stays healthy."""
    run = run_strategy(scene, "active_variance", 64, seed=1, n_init=8)
    pts = np.stack([run.order // scene.grid, run.order % scene.grid], axis=1).astype(float)
    d = np.sqrt(((pts[None] - pts[:, None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    assert np.median(nn) > 1.5  # not collapsed onto adjacent pixels


def test_hunt_beats_random_on_defect_scene():
    """Regression check at fixed seeds: expected exceedance finds more defects."""
    scene = make_scene(SceneParams(grid=48, length_scale=8, n_defects=8, noise_sigma=0.3, seed=11))
    budget = 300
    hunt = run_strategy(scene, "active_hunt", budget, seed=0)
    rand = run_strategy(scene, "random", budget, seed=0)
    from activescan import defect_hit_steps

    n_hunt = int(np.sum(defect_hit_steps(scene, hunt.order) > 0))
    n_rand = int(np.sum(defect_hit_steps(scene, rand.order) > 0))
    assert n_hunt >= n_rand
    assert n_hunt >= 4


def test_fixed_lengthscale_propagates(scene):
    run = run_strategy(
        scene, "active_variance", 40, seed=0, n_init=8, refit_every=16, fixed_lengthscale=0.07
    )
    assert all(h.lengthscale == pytest.approx(0.07) for h in run.hyper_history)
