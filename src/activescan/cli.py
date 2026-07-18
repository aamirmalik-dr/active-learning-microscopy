"""Command-line interface: activescan <command>.

Commands:
    simulate   generate a scene with exact ground truth
    run        execute one strategy on a scene and score it
    replay     execute a strategy against your own fully acquired 2D map
    benchmark  run a fixed-seed YAML benchmark config
    demo       all strategies on the committed sample scene
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .benchmark import run_config
from .io import load_external, load_scene, save_run, save_scene
from .metrics import defect_hit_steps, rmse
from .plots import animate_run, plot_run, plot_scene
from .reconstruct import gp_reconstruct
from .sim import FIELD_KINDS, SceneParams, make_scene
from .strategies import STRATEGIES, run_strategy

DEFAULT_SAMPLE = "data/sample/scene_64.npz"
DEFAULT_SMOOTH_SAMPLE = "data/sample/scene_smooth_64.npz"


def _cmd_simulate(args: argparse.Namespace) -> int:
    params = SceneParams(
        grid=args.grid,
        length_scale=args.length_scale,
        field_kind=args.kind,
        n_grains=args.grains,
        n_defects=args.defects,
        defect_amplitude=args.defect_amplitude,
        defect_sigma=args.defect_sigma,
        noise_sigma=args.noise,
        seed=args.seed,
    )
    scene = make_scene(params)
    save_scene(args.out, scene)
    print(
        f"wrote {args.out}: {params.field_kind} field, grid {params.grid}, "
        f"{params.n_defects} defects, noise {params.noise_sigma:g}"
    )
    if args.figure:
        plot_scene(scene, args.figure)
        print(f"wrote {args.figure}")
    return 0


def _execute(scene, args: argparse.Namespace) -> int:
    run = run_strategy(
        scene,
        args.strategy,
        args.budget,
        seed=args.seed,
        n_init=args.n_init,
        refit_every=args.refit_every,
        kernel=args.kernel,
        record_every=args.record_every if args.gif else 0,
    )
    recon, hyper = gp_reconstruct(scene, run.order, run.values, args.kernel)
    err = rmse(recon, scene.field)
    frac = args.budget / scene.n_positions
    print(f"strategy      : {args.strategy}")
    print(f"budget        : {args.budget} of {scene.n_positions} positions ({frac:.1%})")
    print(f"rmse          : {err:.4f} (units of background std)")
    print(
        f"fitted kernel : lengthscale {hyper.lengthscale:.4f}, "
        f"signal var {hyper.signal_var:.3f}, noise var {hyper.noise_var:.4f}"
    )
    if len(scene.defect_centers):
        steps = defect_hit_steps(scene, run.order)
        found = steps[steps > 0]
        print(
            f"defects found : {len(found)} of {len(steps)}"
            + (f", first at step {found.min()}" if len(found) else "")
        )
    if args.out:
        save_run(args.out, run)
        print(f"wrote {args.out}")
    if args.figure:
        plot_run(scene, run, recon, args.figure)
        print(f"wrote {args.figure}")
    if args.gif:
        if not run.snapshots:
            print("note: --gif needs an active strategy; baselines have no posterior to animate")
        else:
            animate_run(scene, run, args.gif)
            print(f"wrote {args.gif}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return _execute(load_scene(args.scene), args)


def _cmd_replay(args: argparse.Namespace) -> int:
    scene = load_external(args.map, noise_sigma=args.noise)
    print(
        f"replaying strategies against {args.map} "
        f"({scene.grid}x{scene.grid}, replay noise {args.noise:g})"
    )
    return _execute(scene, args)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    result = run_config(args.config)
    dest = Path(args.config).parent.parent / result["config"]["out"]
    print(f"mode {result['mode']}: wrote {dest} in {result['wall_time_s']} s")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    smooth = load_scene(args.smooth_scene)
    defects = load_scene(args.scene)
    budget = args.budget
    print(
        "committed samples: one background field twice, without and with its "
        f"{len(defects.defect_centers)} defects (grid {smooth.grid}, noise "
        f"{smooth.params.noise_sigma:g})"
    )
    print(
        f"budget {budget} of {smooth.n_positions} positions "
        f"({budget / smooth.n_positions:.1%} of a full raster)\n"
    )
    print("task 1: map the defect-free field (score: true RMSE, background-std units)")
    print(f"{'strategy':<18}{'rmse':>8}")
    for strategy in ("random", "lhs", "raster", "active_variance", "active_gradient"):
        run = run_strategy(smooth, strategy, budget, seed=0)
        recon, _ = gp_reconstruct(smooth, run.order, run.values)
        print(f"{strategy:<18}{rmse(recon, smooth.field):>8.4f}")
    hunt_budget = min(2 * budget, defects.n_positions)
    print(
        f"\ntask 2: find the defects in {hunt_budget} measurements "
        "(score: cores hit; geometric, no threshold)"
    )
    print(f"{'strategy':<18}{'defects found':>16}")
    for strategy in ("random", "lhs", "raster", "active_hunt"):
        run = run_strategy(defects, strategy, hunt_budget, seed=0)
        steps = defect_hit_steps(defects, run.order)
        print(f"{strategy:<18}{f'{int(np.sum(steps > 0))}/{len(steps)}':>16}")
    print("\nexact ground truth is known for both scenes, so these are true scores,")
    print("not proxies. Mapping is scored on the defect-free twin because map RMSE")
    print("on a defect scene mostly measures which narrow cores were sampled.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the activescan console command."""
    parser = argparse.ArgumentParser(prog="activescan", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="generate a scene with exact ground truth")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--length-scale", type=float, default=10.0)
    p.add_argument("--kind", choices=FIELD_KINDS, default="smooth")
    p.add_argument("--grains", type=int, default=8)
    p.add_argument("--defects", type=int, default=0)
    p.add_argument("--defect-amplitude", type=float, default=5.0)
    p.add_argument("--defect-sigma", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--figure")
    p.set_defaults(func=_cmd_simulate)

    for name, help_text in [
        ("run", "execute one strategy on a saved scene"),
        ("replay", "execute a strategy against your own 2D map (.npy/.npz)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        if name == "run":
            p.add_argument("scene")
        else:
            p.add_argument("map")
            p.add_argument(
                "--noise",
                type=float,
                default=0.0,
                help="replay noise std in units of map std (0 replays exactly)",
            )
        p.add_argument("--strategy", choices=STRATEGIES, default="active_variance")
        p.add_argument("--budget", type=int, default=200)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--n-init", type=int, default=16)
        p.add_argument("--refit-every", type=int, default=25)
        p.add_argument("--kernel", choices=("rbf", "matern32"), default="rbf")
        p.add_argument("--record-every", type=int, default=10)
        p.add_argument("--out")
        p.add_argument("--figure")
        p.add_argument("--gif")
        p.set_defaults(func=_cmd_run if name == "run" else _cmd_replay)

    p = sub.add_parser("benchmark", help="run a fixed-seed YAML benchmark config")
    p.add_argument("config")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("demo", help="all strategies on the committed sample scenes")
    p.add_argument("--scene", default=DEFAULT_SAMPLE)
    p.add_argument("--smooth-scene", default=DEFAULT_SMOOTH_SAMPLE)
    p.add_argument("--budget", type=int, default=150)
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
