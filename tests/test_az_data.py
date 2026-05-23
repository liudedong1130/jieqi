from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from jieqi.env import JieqiEnv
from rl.az_data import AZSample, AZDataset


class TestAZSample:
    def test_policy_target_sums_to_one(self) -> None:
        policy = np.zeros(8100, dtype=np.float32)
        policy[42] = 1.0
        sample = AZSample(
            observation=np.zeros((28, 10, 9), dtype=np.float32),
            legal_mask=np.ones(8100, dtype=np.int8),
            policy_target=policy,
            value_target=1.0,
        )
        assert abs(sample.policy_target.sum() - 1.0) < 1e-6

    def test_value_target_range(self) -> None:
        sample = AZSample(
            observation=np.zeros((28, 10, 9), dtype=np.float32),
            legal_mask=np.ones(8100, dtype=np.int8),
            value_target=1.0,
        )
        assert -1.0 <= sample.value_target <= 1.0


class TestAZDataset:
    def test_save_load_roundtrip(self) -> None:
        ds = AZDataset()
        for i in range(5):
            policy = np.zeros(8100, dtype=np.float32)
            policy[i * 10] = 1.0
            ds.add(AZSample(
                observation=np.random.randn(28, 10, 9).astype(np.float32),
                legal_mask=np.ones(8100, dtype=np.int8),
                policy_target=policy,
                value_target=1.0 if i % 2 == 0 else -1.0,
                player=i % 2,
                game_id="test",
                move_index=i,
            ))

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.npz")
            ds.save(path)
            assert os.path.exists(path)

            ds2 = AZDataset()
            ds2.load(path)
            assert len(ds2) == 5
            for i in range(5):
                assert np.allclose(ds.samples[i].observation, ds2.samples[i].observation)
                assert ds.samples[i].value_target == ds2.samples[i].value_target
                assert ds.samples[i].player == ds2.samples[i].player

    def test_to_tensors(self) -> None:
        ds = AZDataset()
        for i in range(3):
            policy = np.zeros(8100, dtype=np.float32)
            policy[i] = 1.0
            ds.add(AZSample(
                observation=np.zeros((28, 10, 9), dtype=np.float32),
                legal_mask=np.ones(8100, dtype=np.int8),
                policy_target=policy,
                value_target=0.5,
            ))

        obs_t, mask_t, policy_t, value_t = ds.to_tensors()
        assert obs_t.shape == (3, 28, 10, 9)
        assert mask_t.shape == (3, 8100)
        assert policy_t.shape == (3, 8100)
        assert value_t.shape == (3,)


class TestGenerateAZData:
    def test_generate_and_save(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.npz")
            script = Path(__file__).parent.parent / "scripts" / "generate_az_data.py"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--agent", "random", "--games", "3", "--max-steps", "50",
                 "--output", path],
                capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0
            assert os.path.exists(path)

            ds = AZDataset()
            ds.load(path)
            assert len(ds) > 0
            for s in ds.samples:
                assert abs(s.policy_target.sum() - 1.0) < 1e-6
                assert -1.0 <= s.value_target <= 1.0


class TestSupervisedTrain:
    def test_train_runs(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            # First generate data
            data_path = os.path.join(d, "data.npz")
            gen_script = Path(__file__).parent.parent / "scripts" / "generate_az_data.py"
            subprocess.run(
                [sys.executable, str(gen_script),
                 "--agent", "random", "--games", "2", "--max-steps", "50",
                 "--output", data_path],
                capture_output=True, text=True, timeout=60,
            )

            # Then train
            train_script = Path(__file__).parent.parent / "scripts" / "train_supervised_policy.py"
            result = subprocess.run(
                [sys.executable, str(train_script),
                 "--data", data_path, "--epochs", "3", "--batch-size", "8"],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0
            assert "p_loss" in result.stdout


# ---------------------------------------------------------------------------
#  BeliefMCTS Pretrain pipeline
# ---------------------------------------------------------------------------


class TestPretrainPipeline:
    def test_pipeline_runs(self) -> None:
        import subprocess, sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "pretrained.pt")
            script = Path(__file__).parent.parent / "scripts" / "pretrain_from_belief_mcts.py"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--games", "5", "--max-steps", "50",
                 "--epochs", "2", "--batch-size", "32",
                 "--eval-games", "2",
                 "--checkpoint-out", ckpt_path, "--seed", "42"],
                capture_output=True, text=True, timeout=180,
            )
            assert result.returncode == 0
            assert os.path.exists(ckpt_path)

    def test_pretrained_loadable_by_policy_agent(self) -> None:
        import subprocess, sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "pretrained.pt")
            script = Path(__file__).parent.parent / "scripts" / "pretrain_from_belief_mcts.py"
            subprocess.run(
                [sys.executable, str(script),
                 "--games", "3", "--max-steps", "40",
                 "--epochs", "1", "--batch-size", "32",
                 "--eval-games", "2",
                 "--checkpoint-out", ckpt_path, "--seed", "42"],
                capture_output=True, text=True, timeout=120,
            )
            from agents.policy_agent import PolicyAgent
            from jieqi.env import JieqiEnv

            env = JieqiEnv()
            env.reset(seed=0)
            agent = PolicyAgent(ckpt_path, deterministic=True)
            action = agent.select_action(env)
            assert action in env.legal_actions()

    def test_pretrained_outputs_legal_actions(self) -> None:
        import subprocess, sys
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "pretrained.pt")
            script = Path(__file__).parent.parent / "scripts" / "pretrain_from_belief_mcts.py"
            subprocess.run(
                [sys.executable, str(script),
                 "--games", "3", "--max-steps", "40",
                 "--epochs", "1", "--batch-size", "32",
                 "--eval-games", "2",
                 "--checkpoint-out", ckpt_path, "--seed", "42"],
                capture_output=True, text=True, timeout=120,
            )
            from agents.policy_agent import PolicyAgent
            from jieqi.env import JieqiEnv

            env = JieqiEnv()
            env.reset(seed=0)
            agent = PolicyAgent(ckpt_path, deterministic=True)
            for _ in range(10):
                action = agent.select_action(env)
                assert action in env.legal_actions()
                env.step(action)
                if sum(env.legal_action_mask()) == 0:
                    break
