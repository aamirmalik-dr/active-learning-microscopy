"""Regression tests for the committed sample scene."""

from pathlib import Path

import numpy as np
import pytest

from activescan import load_scene, make_scene
from activescan.cli import main

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "sample" / "scene_64.npz"
SMOOTH_SAMPLE = REPO / "data" / "sample" / "scene_smooth_64.npz"


@pytest.fixture(scope="module")
def sample_scene():
    if not SAMPLE.exists():
        pytest.skip("committed sample scene not present")
    return load_scene(SAMPLE)


def test_sample_scene_shape_and_truth(sample_scene):
    assert sample_scene.grid == 64
    assert len(sample_scene.defect_centers) == 8
    assert sample_scene.params.noise_sigma == pytest.approx(0.3)
    assert np.isfinite(sample_scene.field).all()


def test_sample_scene_regenerates_from_params(sample_scene):
    """The committed file must stay in sync with the simulator."""
    regenerated = make_scene(sample_scene.params)
    assert np.allclose(regenerated.field, sample_scene.field, atol=1e-6)
    assert np.array_equal(regenerated.defect_centers, sample_scene.defect_centers)


def test_smooth_twin_shares_background(sample_scene):
    if not SMOOTH_SAMPLE.exists():
        pytest.skip("committed smooth sample not present")
    smooth = load_scene(SMOOTH_SAMPLE)
    assert len(smooth.defect_centers) == 0
    assert np.allclose(smooth.background, sample_scene.background, atol=1e-6)
    regenerated = make_scene(smooth.params)
    assert np.allclose(regenerated.field, smooth.field, atol=1e-6)


def test_demo_runs_on_sample(capsys, monkeypatch):
    if not SAMPLE.exists():
        pytest.skip("committed sample scene not present")
    monkeypatch.chdir(REPO)
    rc = main(["demo", "--budget", "60"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "active_variance" in out and "rmse" in out
