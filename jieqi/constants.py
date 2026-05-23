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


# True type pool for hidden pieces (15 pieces per side, excluding King)
# 2 Rook, 2 Horse, 2 Cannon, 2 Advisor, 2 Elephant, 5 Pawn
HIDDEN_TRUE_TYPE_POOL: list[PieceType] = [
    PieceType.ROOK, PieceType.ROOK,
    PieceType.HORSE, PieceType.HORSE,
    PieceType.CANNON, PieceType.CANNON,
    PieceType.ADVISOR, PieceType.ADVISOR,
    PieceType.ELEPHANT, PieceType.ELEPHANT,
    PieceType.PAWN, PieceType.PAWN, PieceType.PAWN, PieceType.PAWN, PieceType.PAWN,
]

# Standard Xiangqi starting layout: (row, col) -> (color, origin_type)
# Row 0 = Black's back rank, Row 9 = Red's back rank
STANDARD_LAYOUT: dict[tuple[int, int], tuple[Color, PieceType]] = {
    # Black back rank (row 0)
    (0, 0): (Color.BLACK, PieceType.ROOK),
    (0, 1): (Color.BLACK, PieceType.HORSE),
    (0, 2): (Color.BLACK, PieceType.ELEPHANT),
    (0, 3): (Color.BLACK, PieceType.ADVISOR),
    (0, 4): (Color.BLACK, PieceType.KING),
    (0, 5): (Color.BLACK, PieceType.ADVISOR),
    (0, 6): (Color.BLACK, PieceType.ELEPHANT),
    (0, 7): (Color.BLACK, PieceType.HORSE),
    (0, 8): (Color.BLACK, PieceType.ROOK),
    # Black cannons (row 2)
    (2, 1): (Color.BLACK, PieceType.CANNON),
    (2, 7): (Color.BLACK, PieceType.CANNON),
    # Black pawns (row 3)
    (3, 0): (Color.BLACK, PieceType.PAWN),
    (3, 2): (Color.BLACK, PieceType.PAWN),
    (3, 4): (Color.BLACK, PieceType.PAWN),
    (3, 6): (Color.BLACK, PieceType.PAWN),
    (3, 8): (Color.BLACK, PieceType.PAWN),
    # Red pawns (row 6)
    (6, 0): (Color.RED, PieceType.PAWN),
    (6, 2): (Color.RED, PieceType.PAWN),
    (6, 4): (Color.RED, PieceType.PAWN),
    (6, 6): (Color.RED, PieceType.PAWN),
    (6, 8): (Color.RED, PieceType.PAWN),
    # Red cannons (row 7)
    (7, 1): (Color.RED, PieceType.CANNON),
    (7, 7): (Color.RED, PieceType.CANNON),
    # Red back rank (row 9)
    (9, 0): (Color.RED, PieceType.ROOK),
    (9, 1): (Color.RED, PieceType.HORSE),
    (9, 2): (Color.RED, PieceType.ELEPHANT),
    (9, 3): (Color.RED, PieceType.ADVISOR),
    (9, 4): (Color.RED, PieceType.KING),
    (9, 5): (Color.RED, PieceType.ADVISOR),
    (9, 6): (Color.RED, PieceType.ELEPHANT),
    (9, 7): (Color.RED, PieceType.HORSE),
    (9, 8): (Color.RED, PieceType.ROOK),
}

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
