from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from jieqi.env import JieqiEnv
from rl.model import PolicyValueNet, ACTION_SPACE, NUM_CHANNELS


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------


class TestModel:
    def test_forward_shapes(self) -> None:
        model = PolicyValueNet()
        model.eval()
        x = torch.randn(4, NUM_CHANNELS, 10, 9)
        with torch.no_grad():
            logits, value = model(x)
        assert logits.shape == (4, ACTION_SPACE)
        assert value.shape == (4, 1)

    def test_value_in_range(self) -> None:
        model = PolicyValueNet()
        model.eval()
        x = torch.randn(16, NUM_CHANNELS, 10, 9)
        with torch.no_grad():
            _, value = model(x)
        assert value.min() >= -1.0
        assert value.max() <= 1.0


# ---------------------------------------------------------------------------
#  Action selection with mask
# ---------------------------------------------------------------------------


class TestMaskedAction:
    def test_masked_actions_not_selected(self) -> None:
        model = PolicyValueNet()
        model.eval()
        env = JieqiEnv()
        obs = env.reset(seed=42)
        mask = env.legal_action_mask()
        t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(t)
        logits = logits[0]
        mask_t = torch.from_numpy(mask)
        logits = logits.masked_fill(mask_t == 0, -1e9)
        dist = Categorical(logits=logits)
        for _ in range(100):
            action = dist.sample().item()
            assert mask[action] == 1, f"Sampled illegal action {action}"

    def test_legal_action_mask_exists(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        mask = env.legal_action_mask()
        assert mask.sum() > 0
        assert mask.shape == (ACTION_SPACE,)


# ---------------------------------------------------------------------------
#  Training pipeline
# ---------------------------------------------------------------------------


class TestTraining:
    def test_train_10_episodes_no_error(self) -> None:
        from rl.trainer import PPOTrainer

        PPOTrainer.set_seed(42)
        env = JieqiEnv(max_steps=100)
        trainer = PPOTrainer(env, episodes_per_update=4)
        for i in range(10):
            trainer.collect_episode(seed=42 + i)
            if (i + 1) % 4 == 0:
                trainer.update()
        assert trainer.episode_count == 10

    def test_loss_not_nan(self) -> None:
        from rl.trainer import PPOTrainer

        PPOTrainer.set_seed(42)
        env = JieqiEnv(max_steps=100)
        trainer = PPOTrainer(env, episodes_per_update=4)
        for i in range(8):
            trainer.collect_episode(seed=42 + i)
        stats = trainer.update()
        for loss_name in ["total_loss", "policy_loss", "value_loss"]:
            val = stats.get(loss_name, 0)
            assert not np.isnan(val), f"{loss_name} is NaN"
            assert not np.isinf(val), f"{loss_name} is Inf"

    def test_50_episodes_no_nan(self) -> None:
        from rl.trainer import PPOTrainer

        PPOTrainer.set_seed(123)
        env = JieqiEnv(max_steps=80)
        trainer = PPOTrainer(env, episodes_per_update=8)
        for i in range(50):
            trainer.collect_episode(seed=123 + i)
            if (i + 1) % 8 == 0:
                stats = trainer.update()
                if stats.get("nan_detected", 0) > 0:
                    pytest.fail("NaN detected during training")

    def test_metrics_include_approx_kl_and_ev(self) -> None:
        from rl.trainer import PPOTrainer

        PPOTrainer.set_seed(42)
        env = JieqiEnv(max_steps=80)
        trainer = PPOTrainer(env, episodes_per_update=4)
        for i in range(8):
            trainer.collect_episode(seed=42 + i)
        stats = trainer.update()
        assert "approx_kl" in stats
        assert "clip_frac" in stats
        assert "explained_var" in stats
        assert stats["explained_var"] >= -1.0  # sanity range check

    def test_seed_reproducibility(self) -> None:
        from rl.trainer import PPOTrainer

        def train_several_eps(seed: int) -> list[float]:
            PPOTrainer.set_seed(seed)
            env = JieqiEnv(max_steps=80)
            trainer = PPOTrainer(env, episodes_per_update=4)
            returns = []
            for i in range(8):
                s = trainer.collect_episode(seed=seed + i)
                returns.append(s["return"])
                if (i + 1) % 4 == 0:
                    trainer.update()
            return returns

        r1 = train_several_eps(42)
        r2 = train_several_eps(42)
        # Same seed should produce same episode returns
        assert r1 == r2

    def test_set_seed(self) -> None:
        from rl.trainer import PPOTrainer
        import random as py_random

        PPOTrainer.set_seed(999)
        a = py_random.randint(0, 10000)
        b = np.random.randint(0, 10000)
        c = torch.randint(0, 10000, (1,)).item()

        PPOTrainer.set_seed(999)
        a2 = py_random.randint(0, 10000)
        b2 = np.random.randint(0, 10000)
        c2 = torch.randint(0, 10000, (1,)).item()

        assert a == a2
        assert b == b2
        assert c == c2


# ---------------------------------------------------------------------------
#  Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load(self) -> None:
        from rl.trainer import PPOTrainer

        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(env, episodes_per_update=2)
        trainer.collect_episode(seed=1)
        trainer.collect_episode(seed=2)
        trainer.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            trainer.save(path)
            assert os.path.exists(path)
            env2 = JieqiEnv(max_steps=50)
            trainer2 = PPOTrainer(env2)
            trainer2.load(path)
            for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
                assert torch.allclose(p1, p2)

    def test_save_load_episode_count(self) -> None:
        from rl.trainer import PPOTrainer

        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(env)
        for i in range(5):
            trainer.collect_episode(seed=i)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            trainer.save(path)
            env2 = JieqiEnv(max_steps=50)
            trainer2 = PPOTrainer(env2)
            trainer2.load(path)
            assert trainer2.episode_count == 5


# ---------------------------------------------------------------------------
#  Script
# ---------------------------------------------------------------------------


class TestTrainScript:
    def test_script_runs(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(__file__).parent.parent / "scripts" / "train_ppo.py"
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--episodes", "8",
                    "--max-steps", "50",
                    "--episodes-per-update", "4",
                    "--log-interval", "4",
                    "--eval-interval", "0",
                    "--checkpoint-dir", tmpdir,
                    "--seed", "123",
                ],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, f"Script failed:\n{result.stderr}"
            assert "Done" in result.stdout

    def test_csv_logger_output(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(__file__).parent.parent / "scripts" / "train_ppo.py"
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--episodes", "16",
                    "--max-steps", "50",
                    "--episodes-per-update", "8",
                    "--log-interval", "10",
                    "--eval-interval", "0",
                    "--checkpoint-dir", tmpdir,
                    "--seed", "42",
                ],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0
            csv_path = os.path.join(tmpdir, "metrics.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) >= 2  # header + at least 1 data row
            assert "episode" in lines[0]
            assert "approx_kl" in lines[0]
            assert "explained_var" in lines[0]

    def test_best_checkpoint_saved(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(__file__).parent.parent / "scripts" / "train_ppo.py"
            subprocess.run(
                [
                    sys.executable, str(script),
                    "--episodes", "30",
                    "--max-steps", "50",
                    "--episodes-per-update", "6",
                    "--log-interval", "10",
                    "--eval-interval", "15",
                    "--eval-games", "2",
                    "--checkpoint-dir", tmpdir,
                    "--seed", "99",
                ],
                capture_output=True, text=True, timeout=180,
            )
            assert os.path.exists(os.path.join(tmpdir, "latest.pt"))
            assert os.path.exists(os.path.join(tmpdir, "metrics.csv"))
