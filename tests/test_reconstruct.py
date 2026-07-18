"""Tests for the shared reconstructors."""

import numpy as np
import pytest

from activescan import (
    SceneParams,
    gp_reconstruct,
    interp_reconstruct,
    make_scene,
    rmse,
)


@pytest.fixture(scope="module")
def dense_case():
    scene = make_scene(SceneParams(grid=32, length_scale=6, noise_sigma=0.1, seed=0))
    rng = np.random.default_rng(0)
    order = rng.choice(scene.n_positions, size=400, replace=False)
    values = scene.measure(order, rng)
    return scene, order, values


def test_gp_reconstruct_beats_noise_floor(dense_case):
    scene, order, values = dense_case
    recon, hyper = gp_reconstruct(scene, order, values)
    # with 400 of 1024 points the GP posterior mean should average noise down
    assert rmse(recon, scene.field) < scene.params.noise_sigma
    assert hyper.lengthscale > 0


def test_gp_reconstruct_fixed_lengthscale(dense_case):
    scene, order, values = dense_case
    _, hyper = gp_reconstruct(scene, order, values, fixed_lengthscale=0.4)
    assert hyper.lengthscale == pytest.approx(0.4)


def test_gp_reconstruct_reuses_given_hyper(dense_case):
    scene, order, values = dense_case
    _, hyper = gp_reconstruct(scene, order, values)
    recon2, hyper2 = gp_reconstruct(scene, order, values, hyper=hyper)
    assert hyper2 == hyper
    assert recon2.shape == (scene.grid, scene.grid)


def test_interp_reconstruct_covers_grid(dense_case):
    scene, order, values = dense_case
    recon = interp_reconstruct(scene, order, values)
    assert recon.shape == (scene.grid, scene.grid)
    assert np.isfinite(recon).all()
    assert rmse(recon, scene.field) < 0.5
