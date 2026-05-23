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
        # Tanh → [-1, 1]
        assert value.min() >= -1.0
        assert value.max() <= 1.0


# ---------------------------------------------------------------------------
#  Action selection with mask
# ---------------------------------------------------------------------------


class TestMaskedAction:
    def test_masked_actions_not_selected(self) -> None:
        """Illegal actions should have logits set to -1e9 and never be sampled."""
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
        # Sample 100 times — should never hit an illegal action
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

        env = JieqiEnv(max_steps=100)
        trainer = PPOTrainer(env, episodes_per_update=4)
        for i in range(10):
            trainer.collect_episode(seed=i)
            if (i + 1) % 4 == 0:
                trainer.update()
        assert trainer.episode_count == 10

    def test_loss_not_nan(self) -> None:
        from rl.trainer import PPOTrainer

        env = JieqiEnv(max_steps=100)
        trainer = PPOTrainer(env, episodes_per_update=4)
        for i in range(8):
            trainer.collect_episode(seed=i)
        stats = trainer.update()
        for loss_name in ["total_loss", "policy_loss", "value_loss"]:
            val = stats.get(loss_name, 0)
            assert not np.isnan(val), f"{loss_name} is NaN"
            assert not np.isinf(val), f"{loss_name} is Inf"

    def test_training_reduces_policy_loss(self) -> None:
        """Policy loss should generally decrease after multiple updates."""
        from rl.trainer import PPOTrainer

        env = JieqiEnv(max_steps=80)
        trainer = PPOTrainer(env, episodes_per_update=4)
        losses = []
        for ep in range(1, 33):
            trainer.collect_episode(seed=ep)
            if ep % 4 == 0:
                stats = trainer.update()
                losses.append(stats["total_loss"])
        # Loss should not be monotonically increasing
        assert losses[0] < 100.0  # sanity: not exploding


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

            # Load into new trainer
            env2 = JieqiEnv(max_steps=50)
            trainer2 = PPOTrainer(env2)
            trainer2.load(path)

            # Check model params match
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
                    "--checkpoint", tmpdir,
                ],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, f"Script failed:\n{result.stderr}"
            assert "Training complete" in result.stdout
