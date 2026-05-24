from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jieqi.env import JieqiEnv
from jieqi.constants import Color, HIDDEN_TRUE_TYPE_POOL, PieceType
from vision.adapter import (
    VisionBoardState,
    game_state_to_vision_state,
    validate_vision_state,
    vision_state_to_game_state,
)


def _make_sample_vision_json(env: JieqiEnv) -> dict:
    vs = game_state_to_vision_state(env)
    return vs.to_dict()


class TestVisionAdapter:
    def test_export_import_roundtrip(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        vs = game_state_to_vision_state(env)
        data = vs.to_dict()

        env2 = JieqiEnv()
        env2.reset()
        state2 = VisionBoardState.from_dict(data)
        vision_state_to_game_state(state2, env2)

        # Same piece count
        c1 = sum(1 for p in env.board.cells if p is not None)
        c2 = sum(1 for p in env2.board.cells if p is not None)
        assert c1 == c2 == 32

    def test_validate_valid_state(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        vs = game_state_to_vision_state(env)
        errs = validate_vision_state(vs)
        assert len(errs) == 0

    def test_validate_invalid_state(self) -> None:
        state = VisionBoardState(cells=[
            {"row": 0, "col": 0, "state": "red_open", "piece_type": 0},
            {"row": 0, "col": 0, "state": "black_open", "piece_type": 0},  # overlap!
        ])
        errs = validate_vision_state(state)
        assert len(errs) > 0

    def test_validate_missing_king(self) -> None:
        state = VisionBoardState(cells=[
            {"row": 0, "col": 0, "state": "red_open", "piece_type": 4},  # rook
        ])
        errs = validate_vision_state(state)
        assert any("King" in e for e in errs)

    def test_hidden_king_rejected(self) -> None:
        state = VisionBoardState(cells=[
            {"row": 0, "col": 0, "state": "red_hidden", "piece_type": 0},  # KING hidden!
        ])
        errs = validate_vision_state(state)
        assert any("KING" in e.upper() for e in errs)

    def test_no_true_type_leak(self) -> None:
        """Vision export must only use effective_type, not true_type."""
        env = JieqiEnv()
        env.reset(seed=42)
        vs = game_state_to_vision_state(env)
        data = vs.to_dict()
        for c in data["cells"]:
            assert "true_type" not in c
            if "hidden" in c.get("state", ""):
                # hidden piece_type must be origin_type (1-6), never KING(0)
                assert c["piece_type"] != 0
                # Verify it's the origin_type not true_type
                pos = c["row"] * 9 + c["col"]
                piece = env.board[pos]
                assert piece is not None
                if not piece.revealed:
                    assert c["piece_type"] == int(piece.origin_type)

    def test_import_samples_hidden_true_types_from_pool(self) -> None:
        env = JieqiEnv()
        state = VisionBoardState(cells=[
            {"row": 9, "col": 4, "state": "red_open", "piece_type": int(PieceType.KING)},
            {"row": 0, "col": 4, "state": "black_open", "piece_type": int(PieceType.KING)},
            {"row": 6, "col": 0, "state": "red_hidden", "piece_type": int(PieceType.PAWN)},
            {"row": 6, "col": 2, "state": "red_hidden", "piece_type": int(PieceType.PAWN)},
        ])

        vision_state_to_game_state(state, env, seed=0)
        red_hidden_true_types = [
            p.true_type
            for p in env.board.cells
            if p is not None and p.color == Color.RED and not p.revealed
        ]

        assert sorted(red_hidden_true_types) != [PieceType.PAWN, PieceType.PAWN]
        assert all(t in HIDDEN_TRUE_TYPE_POOL for t in red_hidden_true_types)


class TestAnalyzeScript:
    def test_script_runs(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        data = _make_sample_vision_json(env)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with open(path, "w") as f:
                json.dump(data, f)

            script = Path(__file__).parent.parent / "scripts" / "analyze_vision_state.py"
            result = subprocess.run(
                [sys.executable, str(script), path, "--agent", "greedy", "--top-k", "3"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0
            assert "Legal moves" in result.stdout
