from __future__ import annotations

from enum import IntEnum

# Board dimensions
BOARD_ROWS: int = 10
BOARD_COLS: int = 9
BOARD_SIZE: int = BOARD_ROWS * BOARD_COLS  # 90

# Action space: from-to encoding, every position can move to every position
NUM_ACTIONS: int = BOARD_SIZE * BOARD_SIZE  # 8100

# Number of distinct piece types
NUM_PIECE_TYPES: int = 7

# Piece counts in standard Xiangqi (per side)
NUM_KINGS: int = 1
NUM_ADVISORS: int = 2
NUM_ELEPHANTS: int = 2
NUM_HORSES: int = 2
NUM_ROOKS: int = 2
NUM_CANNONS: int = 2
NUM_PAWNS: int = 5
TOTAL_PIECES_PER_SIDE: int = 16


class Color(IntEnum):
    RED = 0
    BLACK = 1

    def opposite(self) -> Color:
        return Color(1 - self.value)


class PieceType(IntEnum):
    KING = 0
    ADVISOR = 1
    ELEPHANT = 2
    HORSE = 3
    ROOK = 4
    CANNON = 5
    PAWN = 6


# Starting piece counts for standard Xiangqi layout (used for shuffle)
PIECE_COUNTS: dict[PieceType, int] = {
    PieceType.KING: NUM_KINGS,
    PieceType.ADVISOR: NUM_ADVISORS,
    PieceType.ELEPHANT: NUM_ELEPHANTS,
    PieceType.HORSE: NUM_HORSES,
    PieceType.ROOK: NUM_ROOKS,
    PieceType.CANNON: NUM_CANNONS,
    PieceType.PAWN: NUM_PAWNS,
}
