"""Re-run every committed benchmark config, then rebuild metrics and figures.

    python scripts/run_all.py

Total time is about 10 minutes on a laptop CPU; there is nothing to train,
so there is no skip flag. Individual configs can be re-run with
``activescan benchmark configs/<name>.yaml``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    "reconstruction.yaml",
    "nonstationary.yaml",
    "noise_sweep.yaml",
    "defect_search.yaml",
    "sparsity_sweep.yaml",
    "size_sweep.yaml",
    "misspecification.yaml",
    "fairness.yaml",
]


def main() -> int:
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        rc = subprocess.call(
            [sys.executable, "-m", "activescan.cli", "benchmark", f"configs/{cfg}"], cwd=ROOT
        )
        if rc != 0:
            return rc
        print(f"{cfg}: {time.perf_counter() - t0:.0f} s")
    for script in ("make_metrics.py", "make_figures.py"):
        rc = subprocess.call([sys.executable, f"scripts/{script}"], cwd=ROOT)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
