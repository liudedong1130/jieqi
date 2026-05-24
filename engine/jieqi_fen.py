from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""JieqiFEN: compact string representation of Jieqi positions.

Format::
    <board> <turn> - - <hidden> <captured>

*board*  — 10 rows, "/" separated, each row 9 chars.
  Uppercase = Red revealed, Lowercase = Black revealed.
  Letters: K/A/E/H/R/C/P = King/Advisor/Elephant/Horse/Rook/Cannon/Pawn.
  Digit N = N consecutive empty squares.
  Parenthesised lowercase = hidden piece by origin_type: (r),(h),(e),(a),(c),(p).

*turn*  — "w" (Red) or "b" (Black).

*hidden* — "h[p1:t1,p2:t2,...]"  mapping hidden positions to origin_types.
  Only for initial state export; imported positions come from board notation.

*captured* — "c[K,R,P,...]"  captured pieces (debug only, never includes
  true_type of unrevealed captures).

**Hidden true_type is never exposed.**  The FEN only records *origin_type*
for hidden pieces.
"""

from typing import Any

_PIECE_MAP = {"K": 0, "A": 1, "E": 2, "H": 3, "R": 4, "C": 5, "P": 6}
_REV_MAP = {0: "K", 1: "A", 2: "E", 3: "H", 4: "R", 5: "C", 6: "P"}


def parse_jieqi_fen(fen: str) -> dict[str, Any]:
    """Parse a JieqiFEN string into a board-state dict."""
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid FEN: {fen}")

    board_str, turn = parts[0], parts[1]
    pieces: list[dict] = []
    rows = board_str.split("/")
    if len(rows) != 10:
        raise ValueError(f"Expected 10 rows, got {len(rows)}")

    for r, row in enumerate(rows):
        c = 0
        i = 0
        while i < len(row):
            ch = row[i]
            if ch.isdigit():
                c += int(ch)
                i += 1
                continue
            if ch == "(":
                end = row.index(")", i)
                origin_ch = row[i + 1]
                origin = _PIECE_MAP.get(origin_ch.upper(), 0)
                color = 0 if origin_ch.isupper() else 1  # Red=0, Black=1
                pieces.append({"pos": r * 9 + c, "color": color, "type": origin, "revealed": False})
                i = end + 1
                c += 1
                continue
            if ch.isalpha():
                ptype = _PIECE_MAP.get(ch.upper(), 0)
                color = 0 if ch.isupper() else 1
                pieces.append({"pos": r * 9 + c, "color": color, "type": ptype, "revealed": True})
                i += 1
                c += 1
                continue
            i += 1

    current_player = 0 if turn == "w" else 1
    return {"pieces": pieces, "current_player": current_player}


def export_jieqi_fen(board, env=None) -> str:
    """Export current board state as a JieqiFEN string.

    Parameters
    ----------
    board : Board
        The internal board (with full true_state).
    env : JieqiEnv | None
        If provided, only public info is exported (no true_type leak).

    Returns
    -------
    str
    """
    rows = []
    for r in range(10):
        row_chars = []
        empty = 0
        for c in range(9):
            pos = r * 9 + c
            p = board[pos]
            if p is None:
                empty += 1
            else:
                if empty > 0:
                    row_chars.append(str(empty))
                    empty = 0
                ch = _REV_MAP.get(int(p.effective_type), "?")
                if p.revealed:
                    row_chars.append(ch if p.color == 0 else ch.lower())
                else:
                    row_chars.append(f"({ch.lower()})" if p.color == 1 else f"({ch})")
        if empty > 0:
            row_chars.append(str(empty))
        rows.append("".join(row_chars))

    turn = "w" if board.turn == 0 else "b"
    return "/".join(rows) + f" {turn} - -"
