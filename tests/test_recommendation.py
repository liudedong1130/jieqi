from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jieqi.env import JieqiEnv
from analysis.recommendation import (
    generate_recommendations,
    recommendations_to_json,
    recommendations_to_text,
)


class TestRecommendation:
    def test_top_k_actions_legal(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        recs = generate_recommendations(env, agent_type="ismcts", top_k=5)
        assert len(recs) <= 5
        assert len(recs) > 0
        for r in recs:
            assert r.action in env.legal_actions()

    def test_json_format(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        recs = generate_recommendations(env, agent_type="ismcts", top_k=3)
        js = recommendations_to_json(recs)
        data = json.loads(js)
        assert isinstance(data, list)
        assert "move" in data[0]
        assert "score" in data[0]
        assert "reasons" in data[0]

    def test_text_format(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        recs = generate_recommendations(env, agent_type="ismcts", top_k=3)
        text = recommendations_to_text(recs)
        assert "Rank" in text
        assert "Move" in text

    def test_king_capture_recommended(self) -> None:
        from jieqi import Color, Piece, PieceType
        from jieqi.move import rc_to_pos

        env = JieqiEnv(max_steps=50)
        b = env.board
        for pos in range(90):
            b.set_cell(pos, None)
        b._captured = []
        b._turn = Color.RED
        b.set_cell(rc_to_pos(9, 4), Piece(Color.RED, PieceType.KING, PieceType.KING, True))
        b.set_cell(rc_to_pos(0, 4), Piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        b.set_cell(rc_to_pos(5, 4), Piece(Color.RED, PieceType.ROOK, PieceType.ROOK, True))

        recs = generate_recommendations(env, agent_type="belief_mcts", top_k=3)
        king_pos = rc_to_pos(0, 4)
        best_action = recs[0].action
        assert best_action % 90 == king_pos, f"Expected king capture, got {recs[0].move}"

    def test_uncertainty_nonzero(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        recs = generate_recommendations(env, agent_type="belief_mcts", top_k=5)
        # In Jieqi, most moves have some uncertainty due to hidden pieces
        uncertainties = [r.uncertainty for r in recs]
        assert any(u > 0 for u in uncertainties), "Expected some uncertainty"

    def test_reasons_no_true_type(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        recs = generate_recommendations(env, agent_type="ismcts", top_k=5)
        for r in recs:
            for reason in r.reasons:
                assert "true_type" not in reason.lower()


class TestRecommendScript:
    def test_script_runs(self) -> None:
        script = Path(__file__).parent.parent / "scripts" / "recommend.py"
        result = subprocess.run(
            [sys.executable, str(script), "--agent", "greedy", "--top-k", "3"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Rank" in result.stdout


    def test_script_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "recs.json")
            script = Path(__file__).parent.parent / "scripts" / "recommend.py"
            result = subprocess.run(
                [sys.executable, str(script), "--agent", "greedy", "--top-k", "3", "--output-json", out_path],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0
            with open(out_path) as f:
                data = json.load(f)
            assert len(data) == 3
