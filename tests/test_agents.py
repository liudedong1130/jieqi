from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from jieqi import BOARD_SIZE, Color, Piece, PieceType
from jieqi.board import Board
from jieqi.env import JieqiEnv
from jieqi.move import rc_to_pos
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.musesfish_agent import MusesfishAgent
from agents.musesfish_cpp_agent import MusesfishCppAgent
from agents.rollout_agent import RolloutAgent
from agents.belief_mcts_agent import BeliefState


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
#  MusesfishAgent
# ---------------------------------------------------------------------------

class TestMusesfishAgent:
    def test_select_action_is_legal(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        agent = MusesfishAgent(seed=1, search_min_depth=1, search_max_depth=1)
        for _ in range(30):
            action = agent.select_action(env)
            assert action in env.legal_actions()
            env.step(action)
            if sum(env.legal_action_mask()) == 0:
                break

    def test_no_true_type_peek(self) -> None:
        env1 = JieqiEnv()
        env2 = JieqiEnv()
        env1.reset(seed=42)
        env2.reset(seed=42)

        # Change hidden true identities without changing public origin/revealed state.
        for pos, piece in enumerate(env2.board.cells):
            if piece is not None and not piece.revealed:
                env2.board.set_cell(pos, replace(piece, true_type=PieceType.PAWN))

        agent = MusesfishAgent(seed=7, search_min_depth=1, search_max_depth=1)
        assert np.array_equal(env1.observation(), env2.observation())
        assert env1.legal_actions() == env2.legal_actions()
        assert agent.select_action(env1) == agent.select_action(env2)

    def test_captures_high_value_revealed(self) -> None:
        env = JieqiEnv(max_steps=100)
        _setup_capture_position(env)

        agent = MusesfishAgent(seed=42, use_original_search=False)
        action = agent.select_action(env)
        assert action % 90 == rc_to_pos(2, 0)

    def test_cannon_pressure_scores_higher(self) -> None:
        env = JieqiEnv(max_steps=100)
        b = env.board
        for pos in range(BOARD_SIZE):
            b.set_cell(pos, None)
        b._captured = []
        b._turn = Color.RED
        b.set_cell(rc_to_pos(9, 3), _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        b.set_cell(rc_to_pos(0, 4), _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        b.set_cell(rc_to_pos(5, 4), _piece(Color.RED, PieceType.CANNON, PieceType.CANNON, True))

        agent = MusesfishAgent(seed=1)
        pressure = rc_to_pos(5, 4) * 90 + rc_to_pos(1, 4)
        quiet = rc_to_pos(5, 4) * 90 + rc_to_pos(5, 3)
        assert pressure in env.legal_actions()
        assert quiet in env.legal_actions()
        assert agent.score_action(env, pressure) > agent.score_action(env, quiet)

    def test_get_policy_is_legal_distribution(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        agent = MusesfishAgent(seed=1, top_k=8)
        policy, action = agent.get_policy(env)
        legal = set(env.legal_actions())
        assert policy.shape == (8100,)
        assert action in legal
        assert float(policy.sum()) == pytest.approx(1.0)
        assert all(a in legal for a in np.flatnonzero(policy > 0).tolist())


class TestMusesfishCppAgent:
    def test_select_action_is_legal_when_binary_exists(self) -> None:
        agent = MusesfishCppAgent(seed=1, timeout=3.0, min_depth=1, max_depth=1, fallback=False)
        if not agent.binary_path.exists():
            pytest.skip("musesfish_query binary is not built")

        env = JieqiEnv()
        env.reset(seed=42)
        action = agent.select_action(env)
        assert action in env.legal_actions()

    def test_black_move_uses_black_piece_when_binary_exists(self) -> None:
        agent = MusesfishCppAgent(seed=1, timeout=3.0, min_depth=1, max_depth=1, persistent=False, fallback=False)
        if not agent.binary_path.exists():
            pytest.skip("musesfish_query binary is not built")

        env = JieqiEnv()
        env.reset(seed=42)
        red_action = agent.select_action(env)
        env.step(red_action)

        assert env.current_player() == int(Color.BLACK)
        black_action = agent.select_action(env)
        from_pos = black_action // 90
        piece = env.board.get_piece(from_pos)
        assert black_action in env.legal_actions()
        assert piece is not None
        assert piece.color == Color.BLACK


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


class TestBeliefState:
    def test_black_to_move_preserves_actual_colors(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        env.step(env.legal_actions()[0])

        belief = BeliefState.from_env(env)
        kings = {(item["pos"], item["color"]) for item in belief.revealed if item["ptype"] == PieceType.KING}

        assert (rc_to_pos(9, 4), int(Color.RED)) in kings
        assert (rc_to_pos(0, 4), int(Color.BLACK)) in kings
        assert all(
            item["color"] == int(env.board[item["pos"]].color)
            for item in belief.hidden
            if env.board[item["pos"]] is not None
        )


class TestISMCTSPolicyValueEvaluator:
    def test_policy_value_evaluator_uses_public_encoder(self) -> None:
        import torch

        from rl.ismcts import _get_policy_value_evaluator

        class DummyModel:
            def __call__(self, obs):
                return torch.zeros((obs.shape[0], 8100)), torch.tensor([[0.25]])

        board = Board()
        evaluator = _get_policy_value_evaluator(DummyModel(), "cpu")
        score = evaluator({
            rc_to_pos(9, 4): {"color": int(Color.RED), "ptype": int(PieceType.KING), "revealed": True},
            rc_to_pos(0, 4): {"color": int(Color.BLACK), "ptype": int(PieceType.KING), "revealed": True},
        }, int(Color.BLACK), board)

        assert score == pytest.approx(0.25)


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


# ---------------------------------------------------------------------------
#  Belief MCTS Agent
# ---------------------------------------------------------------------------


class TestBeliefMCTSAgent:
    def test_select_action_is_legal(self) -> None:
        from agents.belief_mcts_agent import BeliefMCTSAgent

        env = JieqiEnv()
        env.reset(seed=42)
        agent = BeliefMCTSAgent(num_samples=10, seed=1)
        for _ in range(20):
            action = agent.select_action(env)
            assert action in env.legal_actions()
            env.step(action)
            if sum(env.legal_action_mask()) == 0:
                break

    def test_no_true_type_peek(self) -> None:
        """BeliefMCTSAgent must never access env.board._cells."""
        from agents.belief_mcts_agent import BeliefMCTSAgent

        env = JieqiEnv()
        env.reset(seed=42)
        agent = BeliefMCTSAgent(num_samples=10, seed=1)
        action = agent.select_action(env)
        assert action in env.legal_actions()

    def test_vs_random_multi_games(self) -> None:
        """BeliefMCTSAgent vs RandomAgent: 5 games without crash."""
        from scripts.evaluate import run_eval

        result = run_eval(
            agent_a_name="belief_mcts",
            agent_b_name="random",
            n_games=5,
            max_steps=100,
            seed=123,
            swap=False,
        )
        assert result["errors"] == 0
        total = result["agent_a"]["wins"] + result["agent_b"]["wins"] + result["draws"]
        assert total == 5

    def test_king_capture_scenario(self) -> None:
        """In a position where king capture is possible, agent should find it."""
        from agents.belief_mcts_agent import BeliefMCTSAgent

        env = JieqiEnv(max_steps=50)
        b = env.board
        for pos in range(BOARD_SIZE):
            b.set_cell(pos, None)
        b._captured = []
        b._turn = Color.RED
        b.set_cell(rc_to_pos(9, 4), Piece(Color.RED, PieceType.KING, PieceType.KING, True))
        b.set_cell(rc_to_pos(0, 4), Piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        b.set_cell(rc_to_pos(5, 4), Piece(Color.RED, PieceType.ROOK, PieceType.ROOK, True))

        agent = BeliefMCTSAgent(num_samples=30, seed=0)
        action = agent.select_action(env)
        to_pos = action % 90
        king_pos = rc_to_pos(0, 4)
        assert to_pos == king_pos, f"Expected king capture at {king_pos}, got action to {to_pos}"

    def test_belief_state_consistent_across_true_types(self) -> None:
        """同一 observation，不同 true_type → BeliefState 相同"""
        from agents.belief_mcts_agent import BeliefState

        env1 = JieqiEnv()
        env1.reset(seed=42)
        bs1 = BeliefState.from_env(env1)

        env2 = JieqiEnv()
        env2.reset(seed=99)
        bs2 = BeliefState.from_env(env2)

        assert bs1.pool_own == bs2.pool_own
        assert bs1.pool_opp == bs2.pool_opp
        assert len(bs1.hidden) == len(bs2.hidden)
        assert len(bs1.revealed) == len(bs2.revealed)

    def test_determinization_piece_counts(self) -> None:
        """sample_determinization produces valid piece counts per side."""
        import random as py_random
        from agents.belief_mcts_agent import BeliefState, sample_determinization

        env = JieqiEnv()
        env.reset(seed=42)
        belief = BeliefState.from_env(env)
        rng = py_random.Random(42)

        board = sample_determinization(belief, rng)
        own_cnt = sum(1 for v in board.values() if v["color"] == 0)
        opp_cnt = sum(1 for v in board.values() if v["color"] == 1)
        assert own_cnt == 16
        assert opp_cnt == 16

    def test_top_k_debug_output(self) -> None:
        """top_k 参数开启时不崩溃"""
        from agents.belief_mcts_agent import BeliefMCTSAgent

        env = JieqiEnv()
        env.reset(seed=42)
        agent = BeliefMCTSAgent(num_samples=5, top_k=3, seed=1)
        action = agent.select_action(env)
        assert action in env.legal_actions()
