from __future__ import annotations

import numpy as np

from .board import Board
from .constants import BOARD_COLS, BOARD_ROWS, Color, PieceType

NUM_CHANNELS = 28
"""Total number of observation channels."""

# ---------------------------------------------------------------------------
#  Channel layout (28 × 10 × 9, float32)
# ---------------------------------------------------------------------------
#   0 –  6:  own revealed pieces   (KING=0 … PAWN=6)
#   7 – 13:  opp revealed pieces   (KING=0 … PAWN=6)
#  14 – 19:  own hidden pieces by origin_type   (ADVISOR=1 … PAWN=6)
#  20 – 25:  opp hidden pieces by origin_type   (ADVISOR=1 … PAWN=6)
#  26:       side-to-move (filled with 1.0)
#  27:       reserved (all zeros)
#
#  Hidden channels map  origin_type → channel via  base + origin_type - 1,
#  since KING (=0) is never hidden.
#  Revealed channels map piece_type → channel via  base + piece_type.
# ---------------------------------------------------------------------------


def encode_observation(board: Board) -> np.ndarray:
    """Encode the board from the current player's perspective.

    Returns a ``(28, 10, 9)`` float32 tensor.  Hidden pieces are
    represented by their ``origin_type`` only — ``true_type`` is
    **never** exposed.
    """
    player = board.turn
    tensor = np.zeros((NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)

    for pos, p in enumerate(board.cells):
        if p is None:
            continue
        r, c = pos // BOARD_COLS, pos % BOARD_COLS

        if p.revealed:
            ch = int(p.effective_type)          # 0 … 6
            base = 0 if p.color == player else 7
        else:
            ch = int(p.origin_type) - 1         # 0 … 5  (ADVISOR=1→0 … PAWN=6→5)
            base = 14 if p.color == player else 20

        tensor[base + ch, r, c] = 1.0

    # side-to-move indicator
    tensor[26, :, :] = 1.0

    return tensor
