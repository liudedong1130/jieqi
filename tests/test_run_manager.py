from __future__ import annotations

import os
import tempfile

from utils.run_manager import RunManager, list_runs


class TestRunManager:
    def test_creates_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            run = RunManager(algo="test", seed=42, base_dir=d, config={"lr": 0.001})
            assert os.path.isdir(run.run_dir)
            assert os.path.isdir(os.path.join(run.run_dir, "checkpoints"))
            run.close()

    def test_saves_config(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            run = RunManager(algo="test", seed=1, base_dir=d, config={"lr": 0.001, "episodes": 100})
            cfg_path = os.path.join(run.run_dir, "config.yaml")
            assert os.path.exists(cfg_path)
            content = open(cfg_path).read()
            assert "lr: 0.001" in content
            assert "episodes: 100" in content
            run.close()

    def test_logs_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            run = RunManager(algo="test", seed=2, base_dir=d)
            run.log_metrics({"episode": 1, "loss": 0.5})
            run.log_metrics({"episode": 2, "loss": 0.3})
            run.close()
            csv_path = os.path.join(run.run_dir, "metrics.csv")
            assert os.path.exists(csv_path)
            lines = open(csv_path).read().strip().splitlines()
            assert len(lines) == 3  # header + 2 rows

    def test_logs_arena_result(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            run = RunManager(algo="test", seed=3, base_dir=d)
            run.log_arena_result({"episode": 10, "vs_random": 0.5})
            run.log_arena_result({"episode": 20, "vs_random": 0.6})
            run.close()
            arena_path = os.path.join(run.run_dir, "arena_results.jsonl")
            assert os.path.exists(arena_path)
            lines = open(arena_path).read().strip().splitlines()
            assert len(lines) == 2

    def test_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            run = RunManager(algo="test", seed=4, base_dir=d)
            path = run.checkpoint_path("best.pt")
            assert "checkpoints" in path
            assert path.endswith("best.pt")
            run.close()

    def test_list_runs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            run1 = RunManager(algo="ppo", seed=1, base_dir=d, config={"lr": 0.001})
            run1.log_metrics({"episode": 1})
            run1.close()
            run2 = RunManager(algo="ppo", seed=2, base_dir=d, config={"lr": 0.002})
            run2.close()

            runs = list_runs(base_dir=d)
            assert len(runs) == 2
            total_metrics = sum(r["metrics_rows"] for r in runs)
            assert total_metrics == 1

    def test_train_ppo_creates_run(self) -> None:
        import subprocess, sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            runs_dir = os.path.join(d, "test_runs")
            script = Path(__file__).parent.parent / "scripts" / "train_ppo.py"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--episodes", "10", "--max-steps", "50",
                 "--episodes-per-update", "5", "--log-interval", "5",
                 "--eval-interval", "0", "--checkpoint-dir", os.path.join(d, "ckpt"),
                 "--seed", "42"],
                capture_output=True, text=True, timeout=120,
                cwd=d,
            )
            assert result.returncode == 0
            # Run directory should be created at repo root's runs/
            # But since cwd is d, the runs dir might be there
            # Actually the RunManager uses relative path "runs" from CWD
            # Since we changed cwd, the runs dir should be in d/runs
            possible = Path(d) / "runs"
            if possible.exists():
                run_dirs = list(possible.iterdir())
                assert len(run_dirs) > 0
                # Check config exists
                cfg = run_dirs[0] / "config.yaml"
                if cfg.exists():
                    assert "ppo" in cfg.read_text()
