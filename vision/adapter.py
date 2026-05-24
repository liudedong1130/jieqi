from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jieqi import BOARD_SIZE, Color, Piece, PieceType
from jieqi.board import Board
from jieqi.env import JieqiEnv
from jieqi.move import rc_to_pos

_VALID_ORIGINS = {1, 2, 3, 4, 5, 6}  # ADVISOR..PAWN, no KING


@dataclass
class VisionCell:
    """A single cell as a vision system would report it."""

    row: int
    col: int
    state: str = "empty"  # empty | red_open | black_open | red_hidden | black_hidden
    piece_type: int = 0    # effective_type (origin_type for hidden)


@dataclass
class VisionBoardState:
    """Public-information board state from vision."""

    cells: list[dict[str, Any]] = field(default_factory=list)
    current_player: int = 0  # 0=RED, 1=BLACK

    @classmethod
    def from_dict(cls, data: dict) -> VisionBoardState:
        return cls(
            cells=data.get("cells", []),
            current_player=data.get("current_player", 0),
        )

    def to_dict(self) -> dict:
        return {"cells": self.cells, "current_player": self.current_player}


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------


def validate_vision_state(state: VisionBoardState) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors = []
    seen = set()
    red_total = 0
    black_total = 0
    has_red_king = False
    has_black_king = False

    for c in state.cells:
        pos = rc_to_pos(c["row"], c["col"])
        if pos in seen:
            errors.append(f"Duplicate piece at ({c['row']},{c['col']})")
        seen.add(pos)

        st = c.get("state", "")
        pt = c.get("piece_type", 0)
        if st == "empty":
            continue
        if st in ("red_open", "red_hidden"):
            red_total += 1
        elif st in ("black_open", "black_hidden"):
            black_total += 1

        if st in ("red_open", "black_open") and pt == 0:
            has_red_king = has_red_king or (st == "red_open")
            has_black_king = has_black_king or (st == "black_open")
        if st in ("red_open", "black_open") and pt == 0:
            if st == "red_open":
                has_red_king = True
            else:
                has_black_king = True
        if st in ("red_hidden", "black_hidden") and pt == 0:
            errors.append(f"Hidden piece at ({c['row']},{c['col']}) cannot be KING")
        if st in ("red_hidden", "black_hidden") and pt not in _VALID_ORIGINS:
            errors.append(f"Invalid hidden origin_type {pt} at ({c['row']},{c['col']})")

    if red_total > 16:
        errors.append(f"Red has {red_total} pieces (>16)")
    if black_total > 16:
        errors.append(f"Black has {black_total} pieces (>16)")

    king_count = sum(1 for c in state.cells if c.get("piece_type") == 0 and "open" in c.get("state", ""))
    if king_count < 2:
        errors.append("Missing King(s)")
    if state.current_player not in (0, 1):
        errors.append(f"Invalid current_player: {state.current_player}")

    return errors


# ---------------------------------------------------------------------------
#  Conversion
# ---------------------------------------------------------------------------


def vision_state_to_game_state(state: VisionBoardState, env: JieqiEnv) -> None:
    """Apply a vision state to *env* (overwrites board).

    Hidden pieces are created with origin_type = true_type (the env does not
    know the real identities from vision alone).  This is safe because the
    env will only use effective_type for observations.
    """
    errs = validate_vision_state(state)
    if errs:
        raise ValueError("Invalid vision state:\n  " + "\n  ".join(errs))

    board = env.board
    for pos in range(BOARD_SIZE):
        board.set_cell(pos, None)
    board._captured = []
    board._turn = Color(state.current_player)

    for c in state.cells:
        st = c["state"]
        if st == "empty":
            continue
        pt = PieceType(c["piece_type"])
        pos = rc_to_pos(c["row"], c["col"])
        if st == "red_open":
            board.set_cell(pos, Piece(Color.RED, pt, pt, True))
        elif st == "black_open":
            board.set_cell(pos, Piece(Color.BLACK, pt, pt, True))
        elif st == "red_hidden":
            board.set_cell(pos, Piece(Color.RED, pt, pt, False))
        elif st == "black_hidden":
            board.set_cell(pos, Piece(Color.BLACK, pt, pt, False))


def game_state_to_vision_state(env: JieqiEnv) -> VisionBoardState:
    """Export the public board state as a VisionBoardState.

    Hidden pieces use ``effective_type`` (= origin_type), so **true_type
    is never exposed**.
    """
    cells = []
    for pos in range(BOARD_SIZE):
        p = env.board[pos]
        if p is None:
            continue
        r, c = pos // 9, pos % 9
        if p.revealed:
            st = "red_open" if p.color == Color.RED else "black_open"
        else:
            st = "red_hidden" if p.color == Color.RED else "black_hidden"
        cells.append({"row": r, "col": c, "state": st, "piece_type": int(p.effective_type)})
    return VisionBoardState(cells=cells, current_player=int(env.board.turn))
