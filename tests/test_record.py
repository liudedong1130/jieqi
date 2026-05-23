from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from jieqi import Color, Piece, PieceType
from jieqi.env import JieqiEnv
from jieqi.record import build_record, load_from_file, replay_game, save_to_file
from jieqi.move import rc_to_pos


def _run_one_game(seed: int = 42, max_steps: int = 100) -> tuple[JieqiEnv, int, list[dict]]:
    """Run a random game and return (env_after, seed, moves_info)."""
    from agents.random_agent import RandomAgent

    env = JieqiEnv(max_steps=max_steps)
    env.reset(seed=seed)
    agent_red = RandomAgent(seed=seed)
    agent_black = RandomAgent(seed=seed + 1)

    moves_info = []
    done = False
    while not done:
        if env.current_player() == 0:
            action = agent_red.select_action(env)
        else:
            action = agent_black.select_action(env)

        player = env.current_player()
        obs, reward, terminated, truncated, _info = env.step(action)
        moves_info.append({
            "action": action,
            "player": player,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
        })
        done = terminated or truncated

    return env, seed, moves_info


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


class TestRecordSaveLoad:
    def test_save_and_load(self) -> None:
        env, seed, moves_info = _run_one_game(seed=42, max_steps=100)
        record = build_record(env, seed, moves_info)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "record.json")
            save_to_file(record, path)
            loaded = load_from_file(path)

        assert loaded["seed"] == 42
        assert loaded["total_steps"] == record["total_steps"]
        assert len(loaded["moves"]) == len(record["moves"])


class TestReplayConsistency:
    def test_replay_final_board_matches(self) -> None:
        env, seed, moves_info = _run_one_game(seed=42, max_steps=100)
        record = build_record(env, seed, moves_info)

        replayed = replay_game(record, step_by_step=False, speed=0)

        # Compare final boards
        original_cells = env.board.cells
        replayed_cells = replayed.board.cells
        for pos in range(90):
            o = original_cells[pos]
            r = replayed_cells[pos]
            if o is None:
                assert r is None, f"pos {pos}: original None, replayed has piece"
            else:
                assert r is not None, f"pos {pos}: original has piece, replayed None"
                assert o.color == r.color
                assert o.revealed == r.revealed
                assert o.true_type == r.true_type
                assert o.origin_type == r.origin_type

    def test_same_seed_same_actions_full_reproducibility(self) -> None:
        env, seed, moves_info = _run_one_game(seed=123, max_steps=50)

        env2 = JieqiEnv(max_steps=50)
        env2.reset(seed=123)
        for mi in moves_info:
            env2.step(mi["action"])

        # Compare final boards
        for pos in range(90):
            o = env.board.cells[pos]
            r = env2.board.cells[pos]
            if o is None:
                assert r is None
            else:
                assert r is not None
                assert o.true_type == r.true_type
                assert o.origin_type == r.origin_type
                assert o.color == r.color


class TestDebugMode:
    def test_non_debug_no_true_type(self) -> None:
        env, seed, moves_info = _run_one_game(seed=42, max_steps=50)
        record = build_record(env, seed, moves_info, debug=False)
        assert "debug" not in record
        # Verify moves don't contain true_type info
        for m in record["moves"]:
            assert "true_type" not in m

    def test_debug_contains_full_state(self) -> None:
        env, seed, moves_info = _run_one_game(seed=42, max_steps=50)
        record = build_record(env, seed, moves_info, debug=True)
        assert "debug" in record
        assert "initial_hidden" in record["debug"]
        for h in record["debug"]["initial_hidden"]:
            assert "true_type" in h
            assert "origin_type" in h
            assert "pos" in h

    def test_debug_is_explicit(self) -> None:
        """Debug is False by default — calling build_record without debug must not expose true_type."""
        env, seed, moves_info = _run_one_game(seed=42, max_steps=50)
        record = build_record(env, seed, moves_info)  # default debug=False
        assert "debug" not in record


class TestReplayScript:
    def test_script_runs(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        env, seed, moves_info = _run_one_game(seed=42, max_steps=50)
        record = build_record(env, seed, moves_info)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "record.json")
            save_to_file(record, path)
            script = Path(__file__).parent.parent / "scripts" / "replay_game.py"
            result = subprocess.run(
                [sys.executable, str(script), path, "--speed", "0"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0
            assert "Replay complete" in result.stdout
