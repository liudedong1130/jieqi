from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from jieqi.env import JieqiEnv
from rl.model import PolicyValueNet, ResidualPolicyValueNet, create_model, ACTION_SPACE, NUM_CHANNELS


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
#  Value / return perspective correctness
# ---------------------------------------------------------------------------


class TestValuePerspective:
    """Verify value targets are from the correct player's viewpoint."""

    def test_red_wins_red_positive_black_negative(self) -> None:
        """Red wins → Red states target≈+1, Black states target≈-1."""
        from rl.trainer import PPOTrainer
        from jieqi.env import JieqiEnv

        PPOTrainer.set_seed(42)
        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(env, episodes_per_update=1, gamma=1.0)

        obs_dummy = env.reset(seed=42)
        trainer._obs_buf = [obs_dummy] * 3
        trainer._act_buf = [0, 0, 0]
        trainer._logp_buf = [0.0, 0.0, 0.0]
        trainer._val_buf = [0.1, -0.1, 0.2]     # Red, Black, Red
        trainer._rew_buf = [0.0, 0.0, 1.0]       # Red wins on last move
        trainer._done_buf = [False, False, True]
        trainer._player_buf = [0, 1, 0]          # Red, Black, Red (alternating!)

        advantages, returns = trainer._compute_gae()
        # Red's states (idx 0, 2) → target ≈ +1.0
        assert returns[0] > 0.5, f"Red step 0 return={returns[0]:.3f}, expected > 0.5"
        assert returns[2] > 0.5, f"Red step 2 return={returns[2]:.3f}, expected > 0.5"
        # Black's state (idx 1) → target ≈ -1.0
        assert returns[1] < -0.5, f"Black step 1 return={returns[1]:.3f}, expected < -0.5"

    def test_black_wins_black_positive_red_negative(self) -> None:
        """Black wins → Black states target≈+1, Red states target≈-1."""
        from rl.trainer import PPOTrainer
        from jieqi.env import JieqiEnv

        PPOTrainer.set_seed(42)
        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(env, episodes_per_update=1, gamma=1.0)

        obs_dummy = env.reset(seed=42)
        # Red(0) → Black(1) → Red(0) → Black(1, wins!)
        trainer._obs_buf = [obs_dummy] * 4
        trainer._act_buf = [0, 0, 0, 0]
        trainer._logp_buf = [0.0, 0.0, 0.0, 0.0]
        trainer._val_buf = [0.1, -0.1, 0.05, -0.2]  # Red, Black, Red, Black
        trainer._rew_buf = [0.0, 0.0, 0.0, 1.0]      # Black wins on last move
        trainer._done_buf = [False, False, False, True]
        trainer._player_buf = [0, 1, 0, 1]             # alternating!

        advantages, returns = trainer._compute_gae()
        # Red's states (idx 0, 2) → target ≈ -1.0
        assert returns[0] < -0.5, f"Red step 0 return={returns[0]:.3f}, expected < -0.5"
        assert returns[2] < -0.5, f"Red step 2 return={returns[2]:.3f}, expected < -0.5"
        # Black's states (idx 1, 3) → target ≈ +1.0
        assert returns[1] > 0.5, f"Black step 1 return={returns[1]:.3f}, expected > 0.5"
        assert returns[3] > 0.5, f"Black step 3 return={returns[3]:.3f}, expected > 0.5"

    def test_draw_all_targets_zero(self) -> None:
        """Draw/truncated → all targets ≈ 0."""
        from rl.trainer import PPOTrainer
        from jieqi.env import JieqiEnv

        PPOTrainer.set_seed(42)
        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(env, episodes_per_update=1, gamma=1.0)

        obs_dummy = env.reset(seed=42)
        trainer._obs_buf = [obs_dummy] * 3
        trainer._act_buf = [0, 0, 0]
        trainer._logp_buf = [0.0, 0.0, 0.0]
        trainer._val_buf = [0.1, -0.1, 0.05]  # all near zero
        trainer._rew_buf = [0.0, 0.0, 0.0]     # no winner
        trainer._done_buf = [False, False, True]
        trainer._player_buf = [0, 1, 0]

        advantages, returns = trainer._compute_gae()
        # All returns should be close to 0
        for i in range(3):
            assert abs(returns[i]) < 0.5, f"step {i} return={returns[i]:.3f}, expected near 0"

    def test_gae_with_discount(self) -> None:
        """With gamma < 1, returns should be appropriately discounted."""
        from rl.trainer import PPOTrainer
        from jieqi.env import JieqiEnv

        PPOTrainer.set_seed(42)
        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(env, episodes_per_update=1, gamma=0.9)

        obs_dummy = env.reset(seed=42)
        trainer._obs_buf = [obs_dummy] * 3
        trainer._act_buf = [0, 0, 0]
        trainer._logp_buf = [0.0, 0.0, 0.0]
        trainer._val_buf = [0.0, 0.0, 0.0]     # values at 0
        trainer._rew_buf = [0.0, 0.0, 1.0]     # Red wins
        trainer._done_buf = [False, False, True]
        trainer._player_buf = [0, 1, 0]          # Red, Black, Red

        advantages, returns = trainer._compute_gae()
        # Without discount: returns = [1, -1, 1]
        # With gamma=0.9:
        #   t=2: delta=1, gae=1, adv=1, ret=1
        #   t=1: delta=0, gae=0+0.9*0.95*(-1)=-0.855, adv=-0.855, ret=-0.855
        #   t=0: delta=0, gae=0+0.9*0.95*(-(-0.855))=0.731, adv=0.731, ret=0.731
        assert returns[2] == pytest.approx(1.0)
        assert returns[1] > -1.0  # discounted
        assert returns[0] > 0.5   # discounted but positive


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
#  ResNet model
# ---------------------------------------------------------------------------


class TestResNet:
    def test_forward_shapes(self) -> None:
        model = ResidualPolicyValueNet(channels=64, num_blocks=2)
        model.eval()
        x = torch.randn(4, NUM_CHANNELS, 10, 9)
        with torch.no_grad():
            logits, value = model(x)
        assert logits.shape == (4, ACTION_SPACE)
        assert value.shape == (4, 1)

    def test_value_in_range(self) -> None:
        model = ResidualPolicyValueNet(channels=64, num_blocks=2)
        model.eval()
        x = torch.randn(16, NUM_CHANNELS, 10, 9)
        with torch.no_grad():
            _, value = model(x)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_config(self) -> None:
        model = ResidualPolicyValueNet(channels=128, num_blocks=4)
        cfg = model.config()
        assert cfg["type"] == "resnet"
        assert cfg["channels"] == 128
        assert cfg["num_blocks"] == 4

    def test_create_model_factory(self) -> None:
        m1 = create_model("simple_cnn")
        assert isinstance(m1, PolicyValueNet)
        m2 = create_model("resnet", channels=64, num_blocks=2)
        assert isinstance(m2, ResidualPolicyValueNet)
        assert m2.channels == 64
        assert m2.num_blocks == 2


class TestResNetCheckpoint:
    def test_save_load_resnet(self) -> None:
        from rl.trainer import PPOTrainer

        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(
            env, episodes_per_update=2,
            model_type="resnet", model_kwargs={"channels": 64, "num_blocks": 2},
        )
        trainer.collect_episode(seed=1)
        trainer.collect_episode(seed=2)
        trainer.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resnet.pt")
            trainer.save(path)
            assert os.path.exists(path)

            env2 = JieqiEnv(max_steps=50)
            trainer2 = PPOTrainer(env2)
            trainer2.load(path)
            assert isinstance(trainer2.model, ResidualPolicyValueNet)
            assert trainer2.model.channels == 64
            for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
                assert torch.allclose(p1, p2)

    def test_legacy_simple_cnn_checkpoint_still_loads(self) -> None:
        """Checkpoints saved before model_config was added must still load."""
        from rl.trainer import PPOTrainer

        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(env, episodes_per_update=2)
        trainer.collect_episode(seed=1)
        trainer.collect_episode(seed=2)
        trainer.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pt")
            # Save without model_config (simulate old checkpoint)
            torch.save({
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "episode": 5,
            }, path)

            env2 = JieqiEnv(max_steps=50)
            trainer2 = PPOTrainer(env2)
            trainer2.load(path)
            assert trainer2.episode_count == 5

    def test_policy_agent_loads_resnet_checkpoint(self) -> None:
        from rl.trainer import PPOTrainer
        from agents.policy_agent import PolicyAgent

        env = JieqiEnv(max_steps=50)
        trainer = PPOTrainer(
            env, episodes_per_update=2,
            model_type="resnet", model_kwargs={"channels": 64, "num_blocks": 2},
        )
        trainer.collect_episode(seed=1)
        trainer.collect_episode(seed=2)
        trainer.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resnet.pt")
            trainer.save(path)
            agent = PolicyAgent(path)
            assert isinstance(agent.model, ResidualPolicyValueNet)
            action = agent.select_action(env)
            assert action >= 0


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

    def test_train_with_resnet(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(__file__).parent.parent / "scripts" / "train_ppo.py"
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--model", "resnet", "--channels", "32", "--blocks", "1",
                    "--episodes", "8", "--max-steps", "50",
                    "--episodes-per-update", "4", "--log-interval", "4",
                    "--eval-interval", "0", "--checkpoint-dir", tmpdir, "--seed", "42",
                ],
                capture_output=True, text=True, timeout=180,
            )
            assert result.returncode == 0, f"ResNet train failed:\n{result.stderr}"

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
