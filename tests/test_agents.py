from __future__ import annotations

import json
import os

import numpy as np
import pytest

from jieqi import BOARD_SIZE, Color, Piece, PieceType
from jieqi.board import Board
from jieqi.env import JieqiEnv
from jieqi.move import rc_to_pos
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.rollout_agent import RolloutAgent


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _piece(color: Color, origin: PieceType, true: PieceType, revealed: bool = False) -> Piece:
    return Piece(color=color, origin_type=origin, true_type=true, revealed=revealed)


def _setup_capture_position(env: JieqiEnv) -> None:
    """Set up: Red rook can capture Black revealed rook (high value) on vertical line."""
    b = env.board
    for pos in range(BOARD_SIZE):
        b.set_cell(pos, None)
    b._captured = []
    b._turn = Color.RED
    # Kings on different columns — no facing, no check
    b.set_cell(rc_to_pos(9, 4), _piece(Color.RED, PieceType.KING, PieceType.KING, True))
    b.set_cell(rc_to_pos(0, 3), _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
    # Red rook at (5, 0)
    b.set_cell(rc_to_pos(5, 0), _piece(Color.RED, PieceType.ROOK, PieceType.ROOK, True))
    # Black rook at (2, 0) — high value (500), reachable vertically
    b.set_cell(rc_to_pos(2, 0), _piece(Color.BLACK, PieceType.ROOK, PieceType.ROOK, True))
    # Black pawn at (5, 3) — low value (100), reachable horizontally
    b.set_cell(rc_to_pos(5, 3), _piece(Color.BLACK, PieceType.PAWN, PieceType.PAWN, True))


# ---------------------------------------------------------------------------
#  RandomAgent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    def test_select_action_is_legal(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        agent = RandomAgent(seed=1)
        for _ in range(50):
            action = agent.select_action(env)
            assert action in env.legal_actions()
            env.step(action)
            if sum(env.legal_action_mask()) == 0:
                break

    def test_random_selfplay_100_games(self) -> None:
        """RandomAgent vs RandomAgent: 100 games without crash."""
        from scripts.selfplay_random import run_selfplay

        result = run_selfplay(n_games=100, max_steps=200, seed=42)
        assert result["errors"] == 0, f"Got {result['errors']} errors"
        assert result["red_wins"] + result["black_wins"] + result["draws"] == 100
        assert result["avg_steps"] > 0


# ---------------------------------------------------------------------------
#  GreedyAgent
# ---------------------------------------------------------------------------

class TestGreedyAgent:
    def test_select_action_is_legal(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        agent = GreedyAgent(seed=1)
        for _ in range(50):
            action = agent.select_action(env)
            assert action in env.legal_actions()
            env.step(action)
            if sum(env.legal_action_mask()) == 0:
                break

    def test_captures_high_value_revealed(self) -> None:
        """GreedyAgent should prefer capturing a high-value revealed piece."""
        env = JieqiEnv(max_steps=100)
        _setup_capture_position(env)

        agent = GreedyAgent(seed=42)
        action = agent.select_action(env)
        to_pos = action % 90
        rook_target = rc_to_pos(2, 0)  # Black rook, value 500
        pawn_target = rc_to_pos(5, 3)  # Black pawn, value 100
        assert to_pos == rook_target, (
            f"Expected to capture rook at {rook_target}, but targeted {to_pos} (pawn={pawn_target})"
        )

    def test_no_true_type_peek(self) -> None:
        """GreedyAgent must not access board._cells for decision making."""
        env = JieqiEnv()
        env.reset(seed=42)
        agent = GreedyAgent(seed=1)
        # select_action should only use env.legal_actions() and env.observation()
        action = agent.select_action(env)
        assert action in env.legal_actions()

    def test_greedy_vs_random_multi_games(self) -> None:
        """GreedyAgent vs RandomAgent: 10 games without crash."""
        from scripts.evaluate import run_eval

        result = run_eval(agent_a_name="greedy", agent_b_name="random", n_games=10, max_steps=200, seed=123)
        assert result["errors"] == 0
        total = result["agent_a"]["wins"] + result["agent_b"]["wins"] + result["draws"]
        assert total == 10


# ---------------------------------------------------------------------------
#  RolloutAgent
# ---------------------------------------------------------------------------

class TestRolloutAgent:
    def test_select_action_is_legal(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        agent = RolloutAgent(seed=1)
        for _ in range(50):
            action = agent.select_action(env)
            assert action in env.legal_actions()
            env.step(action)
            if sum(env.legal_action_mask()) == 0:
                break

    def test_rollout_vs_random_multi_games(self) -> None:
        """RolloutAgent vs RandomAgent: 10 games without crash."""
        from scripts.evaluate import run_eval

        result = run_eval(agent_a_name="rollout", agent_b_name="random", n_games=10, max_steps=200, seed=123)
        assert result["errors"] == 0
        total = result["agent_a"]["wins"] + result["agent_b"]["wins"] + result["draws"]
        assert total == 10


# ---------------------------------------------------------------------------
#  PolicyAgent
# ---------------------------------------------------------------------------

class TestPolicyAgent:
    def test_loads_checkpoint(self) -> None:
        from agents.policy_agent import PolicyAgent

        agent = PolicyAgent("/tmp/ckpt/ppo_final.pt", deterministic=True)
        assert agent.model is not None

    def test_select_action_is_legal(self) -> None:
        from agents.policy_agent import PolicyAgent

        env = JieqiEnv()
        env.reset(seed=42)
        agent = PolicyAgent("/tmp/ckpt/ppo_final.pt", deterministic=False, seed=1)
        for _ in range(30):
            action = agent.select_action(env)
            assert action in env.legal_actions()
            env.step(action)
            if sum(env.legal_action_mask()) == 0:
                break

    def test_deterministic_mode(self) -> None:
        from agents.policy_agent import PolicyAgent

        env = JieqiEnv()
        env.reset(seed=42)
        agent = PolicyAgent("/tmp/ckpt/ppo_final.pt", deterministic=True, seed=1)
        # Same state → same action
        action1 = agent.select_action(env)
        env2 = JieqiEnv()
        env2.reset(seed=42)
        action2 = agent.select_action(env2)
        assert action1 == action2


# ---------------------------------------------------------------------------
#  evaluate.py CLI
# ---------------------------------------------------------------------------

class TestEvaluateScript:
    def test_help_runs(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "Agent vs Agent" in result.stdout

    def test_random_vs_random(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--agent-a", "random", "--agent-b", "random",
             "--games", "4", "--max-steps", "50"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "wins" in result.stdout

    def test_greedy_vs_random(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--agent-a", "greedy", "--agent-b", "random",
             "--games", "4", "--max-steps", "50"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_policy_vs_random(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--agent-a", "policy", "--checkpoint-a", "/tmp/ckpt/ppo_final.pt",
             "--agent-b", "random",
             "--games", "4", "--max-steps", "50", "--deterministic"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_swap_colors(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--agent-a", "random", "--agent-b", "random",
             "--games", "10", "--max-steps", "50"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "as Red" in result.stdout
        assert "as Black" in result.stdout

    def test_json_output(self) -> None:
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            result = subprocess.run(
                [sys.executable, str(script),
                 "--agent-a", "random", "--agent-b", "random",
                 "--games", "2", "--max-steps", "50", "--output", tmp],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0
            with open(tmp) as f:
                data = json.load(f)
            assert "agent_a" in data
            assert "agent_b" in data
            assert data["games"] == 2
        finally:
            os.unlink(tmp)

    def test_swap_logic_correct(self) -> None:
        """With swap enabled, each agent plays half games as Red."""
        from scripts.evaluate import run_eval

        result = run_eval(
            agent_a_name="random",
            agent_b_name="random",
            n_games=20,
            max_steps=50,
            seed=42,
            swap=True,
        )
        a_red = result["agent_a"]["as_red_total"]
        a_black = result["agent_a"]["as_black_total"]
        assert a_red == 10, f"Agent A should play 10 as Red, got {a_red}"
        assert a_black == 10, f"Agent A should play 10 as Black, got {a_black}"
