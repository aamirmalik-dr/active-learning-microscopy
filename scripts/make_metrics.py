"""Distil the headline numbers from results/*.json into results/metrics.json.

    python scripts/make_metrics.py

Every number in README.md and RESULTS.md traces back to this file, which
traces back to the per-config JSONs, which regenerate from the fixed-seed
configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _load(name: str) -> dict:
    with open(RES / name, encoding="utf-8") as fh:
        return json.load(fh)


def _mean_std(block: dict, index: int) -> dict:
    return {
        "mean": round(block["mean"][index], 4),
        "std": round(block["std"][index], 4),
    }


def main() -> int:
    metrics: dict = {}

    recon = _load("reconstruction.json")
    cps = recon["checkpoints"]
    i100 = cps.index(100)
    i400 = cps.index(400)
    metrics["reconstruction"] = {
        "grid_positions": 4096,
        "rmse_at_100": {s: _mean_std(b, i100) for s, b in recon["strategies"].items()},
        "rmse_at_400": {s: _mean_std(b, i400) for s, b in recon["strategies"].items()},
        "budget_to_rmse_0p15": {
            s: (
                None
                if any(v is None for v in b["budget_to_target"])
                else round(float(np.mean(b["budget_to_target"])), 1)
            )
            for s, b in recon["strategies"].items()
        },
    }

    nonstat = _load("nonstationary.json")
    cps_n = nonstat["checkpoints"]
    i200 = cps_n.index(200)
    metrics["nonstationary"] = {
        "rmse_at_200": {s: _mean_std(b, i200) for s, b in nonstat["strategies"].items()},
    }

    defect = _load("defect_search.json")
    metrics["defect_search"] = {
        s: {
            "found_of_40_at_500": b["n_found_at_budget"],
            "median_hit_step": b["median_hit_step"],
        }
        for s, b in defect["strategies"].items()
    }

    sparsity = _load("sparsity_sweep.json")
    counts = sparsity["defect_counts"]
    metrics["sparsity_sweep"] = {
        s: dict(zip([f"n{c}" for c in counts], [round(m, 3) for m in b["mean"]]))
        for s, b in sparsity["strategies"].items()
    }

    size = _load("size_sweep.json")
    sig = size["defect_sigmas"]
    metrics["size_sweep"] = {
        s: dict(zip([f"sigma{x:g}" for x in sig], [round(m, 3) for m in b["mean"]]))
        for s, b in size["strategies"].items()
    }

    noise = _load("noise_sweep.json")
    sigmas = noise["noise_sigmas"]
    metrics["noise_sweep"] = {
        s: dict(zip([f"sigma{x:g}" for x in sigmas], [round(m, 4) for m in b["mean"]]))
        for s, b in noise["strategies"].items()
    }

    miss = _load("misspecification.json")
    metrics["misspecification"] = {
        "columns": miss["columns"],
        **{s: [round(m, 4) for m in b["mean"]] for s, b in miss["strategies"].items()},
    }

    fair = _load("fairness.json")
    metrics["fairness"] = {
        "reconstructors": fair["reconstructors"],
        **{s: [round(m, 4) for m in b["mean"]] for s, b in fair["strategies"].items()},
    }

    metrics["wall_time_s"] = {
        name.removesuffix(".json"): _load(name)["wall_time_s"]
        for name in (
            "reconstruction.json",
            "nonstationary.json",
            "noise_sweep.json",
            "defect_search.json",
            "sparsity_sweep.json",
            "size_sweep.json",
            "misspecification.json",
            "fairness.json",
        )
    }

    out = RES / "metrics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {out}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
