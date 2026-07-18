"""Tests for scene generation and the measurement model."""

import numpy as np
import pytest

from activescan import ScanScene, SceneParams, make_scene


def test_scene_deterministic_from_seed():
    a = make_scene(SceneParams(grid=32, seed=7))
    b = make_scene(SceneParams(grid=32, seed=7))
    assert np.array_equal(a.field, b.field)
    c = make_scene(SceneParams(grid=32, seed=8))
    assert not np.array_equal(a.field, c.field)


def test_background_normalised():
    scene = make_scene(SceneParams(grid=64, seed=0))
    assert abs(scene.background.mean()) < 1e-9
    assert scene.background.std() == pytest.approx(1.0, abs=1e-9)


def test_grains_have_sharper_steps_than_smooth():
    smooth = make_scene(SceneParams(grid=64, field_kind="smooth", seed=1))
    grains = make_scene(SceneParams(grid=64, field_kind="grains", seed=1))

    def max_step(f):
        return max(np.abs(np.diff(f, axis=0)).max(), np.abs(np.diff(f, axis=1)).max())

    assert max_step(grains.background) > 3 * max_step(smooth.background)


def test_invalid_field_kind_raises():
    with pytest.raises(ValueError):
        make_scene(SceneParams(field_kind="perlin"))


def test_defects_count_amplitude_and_separation():
    params = SceneParams(grid=64, n_defects=6, defect_amplitude=5.0, defect_sigma=2.0, seed=3)
    scene = make_scene(params)
    assert scene.defect_centers.shape == (6, 2)
    bump = scene.field - scene.background
    for cy, cx in scene.defect_centers:
        assert bump[int(round(cy)), int(round(cx))] > 0.7 * params.defect_amplitude
    for i in range(6):
        for j in range(i + 1, 6):
            d = np.hypot(*(scene.defect_centers[i] - scene.defect_centers[j]))
            assert d >= 6.0 * params.defect_sigma


def test_measure_noise_statistics():
    scene = make_scene(SceneParams(grid=32, noise_sigma=0.4, seed=0))
    rng = np.random.default_rng(0)
    idx = np.zeros(20000, dtype=int)
    vals = scene.measure(idx, rng)
    resid = vals - scene.field.ravel()[0]
    assert abs(resid.mean()) < 0.02
    assert resid.std() == pytest.approx(0.4, abs=0.02)


def test_coords_layout():
    scene = make_scene(SceneParams(grid=16, seed=0))
    c = scene.coords()
    assert c.shape == (256, 2)
    assert c.min() > 0 and c.max() < 1
    assert np.allclose(c[0], [0.5 / 16, 0.5 / 16])
    assert np.allclose(c[1], [0.5 / 16, 1.5 / 16])  # row-major: x advances first


def test_defect_core_radius():
    scene = make_scene(SceneParams(n_defects=1, defect_sigma=2.0, seed=0))
    r = scene.defect_core_radius()
    assert r == pytest.approx(2.0 * np.sqrt(2 * np.log(2)))
    # at the core radius the bump is half the amplitude by definition
    assert np.exp(-(r**2) / (2 * 2.0**2)) == pytest.approx(0.5)


def test_scene_helpers():
    scene = make_scene(SceneParams(grid=24, seed=0))
    assert isinstance(scene, ScanScene)
    assert scene.grid == 24
    assert scene.n_positions == 576
