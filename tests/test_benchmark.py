"""Smoke tests for the benchmark harness on miniature configs."""

import json

import pytest
import yaml

from activescan import run_config


def _write_config(tmp_path, name, cfg):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(exist_ok=True)
    path = cfg_dir / name
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


_TINY_SCENE = {"grid": 24, "length_scale": 5, "noise_sigma": 0.3}


def test_reconstruction_mode(tmp_path):
    path = _write_config(
        tmp_path,
        "recon.yaml",
        {
            "mode": "reconstruction",
            "out": "results/recon.json",
            "scene": _TINY_SCENE,
            "strategies": ["random", "lhs", "active_variance"],
            "seeds": [0],
            "checkpoints": [20, 35],
            "n_init": 8,
            "refit_every": 16,
            "target_rmse": 0.5,
        },
    )
    result = run_config(path)
    assert (tmp_path / "results" / "recon.json").exists()
    for name in ("random", "lhs", "active_variance"):
        block = result["strategies"][name]
        assert len(block["mean"]) == 2
        assert len(block["budget_to_target"]) == 1
    with open(tmp_path / "results" / "recon.json", encoding="utf-8") as fh:
        assert json.load(fh)["mode"] == "reconstruction"


def test_defect_search_mode(tmp_path):
    path = _write_config(
        tmp_path,
        "defect.yaml",
        {
            "mode": "defect_search",
            "out": "results/defect.json",
            "scene": dict(_TINY_SCENE, n_defects=3),
            "strategies": ["random", "active_hunt"],
            "seeds": [0],
            "checkpoints": [20, 40],
            "n_init": 8,
            "refit_every": 16,
        },
    )
    result = run_config(path)
    block = result["strategies"]["active_hunt"]
    assert block["n_defects_total"] == 3
    assert 0 <= block["n_found_at_budget"] <= 3


def test_misspecification_mode(tmp_path):
    path = _write_config(
        tmp_path,
        "miss.yaml",
        {
            "mode": "misspecification",
            "out": "results/miss.json",
            "scene": _TINY_SCENE,
            "strategies": ["lhs", "active_variance"],
            "seeds": [0],
            "budget": 30,
            "lengthscale_factors": [0.3, 1.0],
            "n_init": 8,
            "refit_every": 16,
        },
    )
    result = run_config(path)
    assert result["columns"] == ["pinned_0.3x", "pinned_1x", "fitted"]
    for block in result["strategies"].values():
        assert len(block["mean"]) == 3


def test_fairness_mode(tmp_path):
    path = _write_config(
        tmp_path,
        "fair.yaml",
        {
            "mode": "fairness",
            "out": "results/fair.json",
            "scene": _TINY_SCENE,
            "strategies": ["lhs"],
            "seeds": [0],
            "budget": 30,
        },
    )
    result = run_config(path)
    assert len(result["strategies"]["lhs"]["mean"]) == 3


def test_size_sweep_mode(tmp_path):
    path = _write_config(
        tmp_path,
        "size.yaml",
        {
            "mode": "size_sweep",
            "out": "results/size.json",
            "scene": dict(_TINY_SCENE, n_defects=2),
            "strategies": ["lhs", "raster"],
            "seeds": [0],
            "budget": 40,
            "defect_sigmas": [1.0, 1.5],
            "n_init": 8,
            "refit_every": 16,
        },
    )
    result = run_config(path)
    for block in result["strategies"].values():
        assert len(block["mean"]) == 2
        assert all(0.0 <= v <= 1.0 for v in block["mean"])


def test_unknown_mode_raises(tmp_path):
    path = _write_config(tmp_path, "bad.yaml", {"mode": "teleport", "out": "results/x.json"})
    with pytest.raises(ValueError):
        run_config(path)
