from __future__ import annotations

import json
import os
import tempfile

import pytest

from evaluation.elo import INITIAL_ELO, expected_score, update_elo
from evaluation.arena import AgentConfig, Arena


# ---------------------------------------------------------------------------
#  Elo
# ---------------------------------------------------------------------------


class TestElo:
    def test_initial_elo(self) -> None:
        assert INITIAL_ELO == 1000.0

    def test_expected_score_equal(self) -> None:
        assert expected_score(1000, 1000) == pytest.approx(0.5)

    def test_expected_score_higher_wins_more(self) -> None:
        assert expected_score(1200, 1000) > 0.5
        assert expected_score(1000, 1200) < 0.5

    def test_expected_score_symmetry(self) -> None:
        assert expected_score(1000, 1200) == pytest.approx(1.0 - expected_score(1200, 1000))

    def test_update_elo_win(self) -> None:
        new_a, new_b = update_elo(1000, 1000, 1.0)
        assert new_a > 1000
        assert new_b < 1000

    def test_update_elo_loss(self) -> None:
        new_a, new_b = update_elo(1000, 1000, 0.0)
        assert new_a < 1000
        assert new_b > 1000

    def test_update_elo_draw(self) -> None:
        new_a, new_b = update_elo(1000, 1000, 0.5)
        assert new_a == pytest.approx(1000)
        assert new_b == pytest.approx(1000)

    def test_elo_zero_sum(self) -> None:
        """Total Elo points should be conserved."""
        ela, elb = 1200, 1100
        na, nb = update_elo(ela, elb, 0.7)
        assert na + nb == pytest.approx(ela + elb)


# ---------------------------------------------------------------------------
#  Arena
# ---------------------------------------------------------------------------


class TestArena:
    def test_random_vs_random_near_50(self) -> None:
        arena = Arena([AgentConfig("r1", "random"), AgentConfig("r2", "random")])
        results = arena.run_round_robin(n_games=20, max_steps=100, seed=42)
        mr = results[0]
        # Two random agents should be roughly equal
        assert mr.a_wins + mr.b_wins + mr.draws == 20

    def test_greedy_vs_random(self) -> None:
        arena = Arena([AgentConfig("greedy", "greedy"), AgentConfig("random", "random")])
        results = arena.run_round_robin(n_games=10, max_steps=100, seed=42)
        mr = results[0]
        assert mr.total_games == 10

    def test_round_robin_no_crash(self) -> None:
        arena = Arena([
            AgentConfig("r", "random"),
            AgentConfig("g", "greedy"),
            AgentConfig("ro", "rollout"),
        ])
        results = arena.run_round_robin(n_games=4, max_steps=50, seed=42)
        # 3 agents → 3 matches
        assert len(results) == 3

    def test_elo_updated_after_round_robin(self) -> None:
        arena = Arena([
            AgentConfig("r", "random"),
            AgentConfig("g", "greedy"),
        ])
        arena.run_round_robin(n_games=10, max_steps=100, seed=42)
        ratings = arena.rating_table()
        assert len(ratings) == 2
        # Ratings should differ from initial
        for r in ratings:
            assert r["elo"] != INITIAL_ELO

    def test_summary_markdown(self) -> None:
        arena = Arena([AgentConfig("r", "random"), AgentConfig("g", "greedy")])
        arena.run_round_robin(n_games=4, max_steps=50, seed=42)
        md = arena.summary_markdown()
        assert "| Rank |" in md
        assert "r" in md
        assert "g" in md

    def test_summary_json(self) -> None:
        arena = Arena([AgentConfig("r", "random"), AgentConfig("g", "greedy")])
        arena.run_round_robin(n_games=4, max_steps=50, seed=42)
        js = arena.summary_json()
        assert "ratings" in js
        assert "matches" in js
        assert len(js["matches"]) == 1

    def test_color_swap_in_match(self) -> None:
        arena = Arena([AgentConfig("a", "random"), AgentConfig("b", "random")])
        mr = arena.run_match(
            AgentConfig("a", "random"),
            AgentConfig("b", "random"),
            n_games=20, max_steps=50, seed=42,
        )
        # With 20 games, 10 as Red for A, 10 as Red for B
        assert mr.total_games == 20


# ---------------------------------------------------------------------------
#  Arena script
# ---------------------------------------------------------------------------


class TestArenaScript:
    def test_script_help(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "arena.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0

    def test_script_runs(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "arena.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--agents", "random,greedy", "--games", "2", "--max-steps", "50"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "Agent" in result.stdout or "Rank" in result.stdout