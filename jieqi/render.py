from __future__ import annotations

from .board import Board
from .constants import BOARD_COLS, BOARD_ROWS, Color, PieceType
from .pieces import Piece

# Mapping from PieceType to display character
_TYPE_CHAR: dict[PieceType, str] = {
    PieceType.KING: "K",
    PieceType.ADVISOR: "A",
    PieceType.ELEPHANT: "E",
    PieceType.HORSE: "H",
    PieceType.ROOK: "R",
    PieceType.CANNON: "C",
    PieceType.PAWN: "P",
}


def _cell_str(piece: Piece | None) -> str:
    """Return a 2-character string representation of a board cell."""
    if piece is None:
        return "· "  # middle dot + space
    ch = _TYPE_CHAR[piece.effective_type]
    if piece.revealed:
        # Revealed: uppercase=Red, lowercase=Black
        return (ch if piece.color == Color.RED else ch.lower()) + " "
    else:
        # Hidden: show origin_type char + "*"
        origin_ch = _TYPE_CHAR[piece.origin_type]
        return (origin_ch if piece.color == Color.RED else origin_ch.lower()) + "*"


def render(board: Board) -> str:
    """Return an ASCII rendering of the board.

    Conventions:
      - Empty cell: ``·``
      - Revealed Red: uppercase letter (K A E H R C P)
      - Revealed Black: lowercase letter (k a e h r c p)
      - Hidden: origin-type letter + ``*`` (uppercase=Red, lowercase=Black)
    """
    lines: list[str] = []

    # Column header
    header = "    " + "  ".join(str(c) for c in range(BOARD_COLS))
    lines.append(header)

    # Top border
    border = "  +" + "--+" * BOARD_COLS
    lines.append(border)

    for row in range(BOARD_ROWS):
        # Row content
        row_cells: list[str] = [str(row)]
        for col in range(BOARD_COLS):
            pos = row * BOARD_COLS + col
            row_cells.append(_cell_str(board[pos]))
        lines.append(" |".join(row_cells) + " |")

        # Separator / bottom border
        lines.append(border)

    return "\n".join(lines)
