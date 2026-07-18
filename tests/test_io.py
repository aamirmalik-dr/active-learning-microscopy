"""Tests for persistence and the bring-your-own-data path."""

import numpy as np
import pytest

from activescan import (
    RunResult,
    SceneParams,
    load_external,
    load_run,
    load_scene,
    make_scene,
    run_strategy,
    save_run,
    save_scene,
)


def test_scene_roundtrip(tmp_path):
    scene = make_scene(SceneParams(grid=32, n_defects=3, field_kind="grains", seed=9))
    path = tmp_path / "scene.npz"
    save_scene(path, scene)
    loaded = load_scene(path)
    assert np.allclose(loaded.field, scene.field, atol=1e-6)  # float32 storage
    assert np.array_equal(loaded.defect_centers, scene.defect_centers)
    assert loaded.params == scene.params


def test_scene_regenerates_from_stored_params(tmp_path):
    scene = make_scene(SceneParams(grid=32, n_defects=2, seed=4))
    path = tmp_path / "scene.npz"
    save_scene(path, scene)
    loaded = load_scene(path)
    regenerated = make_scene(loaded.params)
    assert np.allclose(regenerated.field, loaded.field, atol=1e-6)


def test_run_roundtrip(tmp_path):
    run = RunResult(strategy="random", order=np.array([3, 1, 2]), values=np.array([0.1, -0.2, 0.5]))
    path = tmp_path / "run.npz"
    save_run(path, run)
    loaded = load_run(path)
    assert loaded.strategy == "random"
    assert np.array_equal(loaded.order, run.order)
    assert np.allclose(loaded.values, run.values)


def test_load_external_normalises():
    arr = 5.0 + 3.0 * np.random.default_rng(0).normal(size=(40, 40))
    scene = load_external(arr, noise_sigma=0.0)
    assert scene.field.mean() == pytest.approx(0.0, abs=1e-10)
    assert scene.field.std() == pytest.approx(1.0)
    assert len(scene.defect_centers) == 0


def test_load_external_rejects_non_square():
    with pytest.raises(ValueError):
        load_external(np.zeros((10, 20)))
    with pytest.raises(ValueError):
        load_external(np.zeros(10))


def test_load_external_rejects_constant_map():
    with pytest.raises(ValueError):
        load_external(np.ones((8, 8)))


def test_load_external_from_files(tmp_path):
    arr = np.random.default_rng(1).normal(size=(16, 16))
    npy = tmp_path / "map.npy"
    np.save(npy, arr)
    scene = load_external(npy, noise_sigma=0.0)
    assert scene.grid == 16
    npz = tmp_path / "map.npz"
    np.savez(npz, mymap=arr)
    scene2 = load_external(npz, noise_sigma=0.0)
    assert np.allclose(scene.field, scene2.field)


def test_replay_with_zero_noise_returns_exact_values():
    arr = np.random.default_rng(2).normal(size=(24, 24))
    scene = load_external(arr, noise_sigma=0.0)
    run = run_strategy(scene, "random", 30, seed=0)
    assert np.allclose(run.values, scene.field.ravel()[run.order])
