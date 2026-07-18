"""End-to-end CLI tests."""

import numpy as np
import pytest

from activescan.cli import main


def test_simulate_then_run(tmp_path, capsys):
    scene_path = tmp_path / "scene.npz"
    fig_path = tmp_path / "scene.png"
    rc = main(
        [
            "simulate",
            "--grid",
            "24",
            "--defects",
            "2",
            "--seed",
            "1",
            "--out",
            str(scene_path),
            "--figure",
            str(fig_path),
        ]
    )
    assert rc == 0
    assert scene_path.exists() and fig_path.exists()

    run_fig = tmp_path / "run.png"
    run_out = tmp_path / "run.npz"
    rc = main(
        [
            "run",
            str(scene_path),
            "--strategy",
            "active_variance",
            "--budget",
            "40",
            "--n-init",
            "8",
            "--refit-every",
            "16",
            "--figure",
            str(run_fig),
            "--out",
            str(run_out),
        ]
    )
    assert rc == 0
    assert run_fig.exists() and run_out.exists()
    out = capsys.readouterr().out
    assert "rmse" in out and "defects found" in out


def test_replay_external_map(tmp_path, capsys):
    arr = np.random.default_rng(0).normal(size=(24, 24))
    map_path = tmp_path / "map.npy"
    np.save(map_path, arr)
    rc = main(
        [
            "replay",
            str(map_path),
            "--strategy",
            "lhs",
            "--budget",
            "40",
        ]
    )
    assert rc == 0
    assert "rmse" in capsys.readouterr().out


def test_gif_from_active_run(tmp_path):
    scene_path = tmp_path / "scene.npz"
    main(["simulate", "--grid", "24", "--seed", "0", "--out", str(scene_path)])
    gif = tmp_path / "run.gif"
    rc = main(
        [
            "run",
            str(scene_path),
            "--strategy",
            "active_variance",
            "--budget",
            "30",
            "--n-init",
            "8",
            "--record-every",
            "10",
            "--gif",
            str(gif),
        ]
    )
    assert rc == 0
    assert gif.exists() and gif.stat().st_size > 1000


def test_missing_command_errors():
    with pytest.raises(SystemExit):
        main([])
