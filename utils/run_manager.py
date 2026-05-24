from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class RunManager:
    """Manages experiment directories, config, metrics, and checkpoints."""

    def __init__(
        self,
        algo: str = "train",
        seed: int = 0,
        base_dir: str = "runs",
        config: dict[str, Any] | None = None,
    ) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_id = f"{ts}_{algo}_seed{seed}"
        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.csv"
        self._arena_path = self.run_dir / "arena_results.jsonl"
        self._csv_file = open(self._metrics_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_header_written = False

        # Save config
        if config:
            self.save_config(config)

        # Git commit
        self._save_git_commit()

    def save_config(self, config: dict[str, Any]) -> None:
        path = self.run_dir / "config.yaml"
        with open(path, "w") as f:
            for k, v in sorted(config.items()):
                f.write(f"{k}: {v}\n")

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if not self._csv_header_written:
            self._csv_writer.writerow(list(metrics.keys()))
            self._csv_header_written = True
        self._csv_writer.writerow([metrics.get(k, "") for k in metrics.keys()])
        self._csv_file.flush()

    def log_arena_result(self, result: dict[str, Any]) -> None:
        with open(self._arena_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def checkpoint_path(self, name: str) -> str:
        return str(self.run_dir / "checkpoints" / name)

    def save_note(self, text: str) -> None:
        with open(self.run_dir / "notes.md", "a") as f:
            f.write(text + "\n")

    def close(self) -> None:
        self._csv_file.close()

    def _save_git_commit(self) -> None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=os.getcwd(),
            )
            if result.returncode == 0:
                (self.run_dir / "git_commit.txt").write_text(result.stdout.strip())
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Run listing
# ---------------------------------------------------------------------------


def list_runs(base_dir: str = "runs") -> list[dict[str, Any]]:
    """Scan *base_dir* and return summary of each run."""
    runs = []
    base = Path(base_dir)
    if not base.exists():
        return runs

    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        info: dict[str, Any] = {"run_id": run_dir.name, "path": str(run_dir)}

        # Parse config
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            cfg = {}
            for line in cfg_path.read_text().splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    cfg[k] = v
            info["config"] = cfg

        # Count checkpoints
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.pt"))
            info["checkpoints"] = [c.name for c in ckpts]

        # Parse last arena result
        arena_path = run_dir / "arena_results.jsonl"
        if arena_path.exists():
            lines = arena_path.read_text().strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                info["last_arena"] = last

        # Parse last metrics row
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists():
            lines = metrics_path.read_text().strip().splitlines()
            info["metrics_rows"] = max(0, len(lines) - 1)

        runs.append(info)

    return runs
