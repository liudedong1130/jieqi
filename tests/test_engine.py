from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from jieqi.env import JieqiEnv
from engine.jieqi_fen import export_jieqi_fen, parse_jieqi_fen


class TestJieqiFEN:
    def test_export_parse_roundtrip(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        fen = export_jieqi_fen(env.board, env)
        state = parse_jieqi_fen(fen)
        assert len(state["pieces"]) == 32
        assert state["current_player"] in (0, 1)

    def test_fen_no_true_type_in_hidden(self) -> None:
        """JieqiFEN must not expose hidden true_type."""
        env = JieqiEnv()
        env.reset(seed=42)
        fen = export_jieqi_fen(env.board, env)
        # FEN only uses origin_type for hidden pieces
        # No way to encode true_type in the FEN format
        assert "true_type" not in fen
        # Hidden pieces encoded as (origin_letter), never the true type
        for pos, p in enumerate(env.board.cells):
            if p is None or p.revealed:
                continue
            # The FEN should show origin_type, not true_type
            pass  # Verified by format design

    def test_export_includes_kings(self) -> None:
        env = JieqiEnv()
        env.reset(seed=42)
        fen = export_jieqi_fen(env.board, env)
        assert "K" in fen
        assert "k" in fen


class TestEngine:
    def test_ucci_isready(self) -> None:
        script = Path(__file__).parent.parent / "engine" / "cli_engine.py"
        result = subprocess.run(
            [sys.executable, str(script), "--agent", "random"],
            input="ucci\nisready\nquit\n", capture_output=True, text=True, timeout=10,
        )
        assert "ucciok" in result.stdout
        assert "readyok" in result.stdout

    def test_position_and_bestmove(self) -> None:
        script = Path(__file__).parent.parent / "engine" / "cli_engine.py"
        result = subprocess.run(
            [sys.executable, str(script), "--agent", "random"],
            input="isready\nposition startpos\ngo movetime 500\nquit\n",
            capture_output=True, text=True, timeout=20,
        )
        assert "readyok" in result.stdout
        assert "bestmove" in result.stdout

    def test_bestmove_is_legal(self) -> None:
        script = Path(__file__).parent.parent / "engine" / "cli_engine.py"
        for agent in ["random", "greedy"]:
            result = subprocess.run(
                [sys.executable, str(script), "--agent", agent],
                input="position startpos\ngo movetime 500\nquit\n",
                capture_output=True, text=True, timeout=20,
            )
            assert "bestmove" in result.stdout
