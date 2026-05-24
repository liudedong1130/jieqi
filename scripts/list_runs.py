#!/usr/bin/env python3
"""List training runs with summary information."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.run_manager import list_runs


def main() -> None:
    runs = list_runs("runs")
    if not runs:
        print("No runs found in ./runs")
        return

    print(f"{'Run ID':<35} {'Type':<12} {'Metrics':>8} {'CKPTs':>6}")
    print("-" * 65)
    for r in runs:
        cfg = r.get("config", {})
        algo = cfg.get("algo", cfg.get("model", "?"))
        metrics = r.get("metrics_rows", 0)
        ckpts = len(r.get("checkpoints", []))
        print(f"{r['run_id']:<35} {algo:<12} {metrics:>8} {ckpts:>6}")


if __name__ == "__main__":
    main()
