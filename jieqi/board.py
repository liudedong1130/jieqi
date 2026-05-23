from __future__ import annotations

import random
from typing import Optional

from .constants import (
    BOARD_SIZE,
    HIDDEN_TRUE_TYPE_POOL,
    STANDARD_LAYOUT,
    Color,
    PieceType,
)
from .move import Move, pos_to_rc
from .pieces import Piece


class Board:
    """Full-knowledge board state for Jieqi.

    Maintains the complete true state including true_type of all hidden pieces.
    External code (agents, encoders) must only access piece type through
    ``Piece.effective_type`` to prevent true_type leakage.
    """

    def __init__(self) -> None:
        self._cells: list[Piece | None] = [None] * BOARD_SIZE
        self._captured: list[Piece] = []
        self._turn: Color = Color.RED

    # ---- initialization ---------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset to the standard Jieqi starting position.

        Kings are placed revealed. The other 15 pieces per side are randomly
        assigned true types and placed face-down.
        """
        rng = random.Random(seed)
        self._cells = [None] * BOARD_SIZE
        self._captured = []
        self._turn = Color.RED
        self._init_side(Color.RED, rng)
        self._init_side(Color.BLACK, rng)

    def _init_side(self, color: Color, rng: random.Random) -> None:
        # Collect all starting positions for this side
        side_positions: list[tuple[int, int]] = [
            (r, c)
            for (r, c), (clr, _) in STANDARD_LAYOUT.items()
            if clr == color
        ]

        king_rc: tuple[int, int] | None = None
        hidden_rcs: list[tuple[int, int]] = []

        for rc in side_positions:
            _, origin = STANDARD_LAYOUT[rc]
            if origin == PieceType.KING:
                king_rc = rc
            else:
                hidden_rcs.append(rc)

        assert king_rc is not None, f"King position not found for {color}"
        assert len(hidden_rcs) == 15, f"Expected 15 hidden positions, got {len(hidden_rcs)}"

        # Place King (always revealed)
        kr, kc = king_rc
        king_pos = kr * 9 + kc
        self._cells[king_pos] = Piece(
            color=color,
            origin_type=PieceType.KING,
            true_type=PieceType.KING,
            revealed=True,
        )

        # Shuffle true types for hidden pieces
        true_types = list(HIDDEN_TRUE_TYPE_POOL)
        rng.shuffle(true_types)

        # Place hidden pieces
        for (r, c), true_type in zip(hidden_rcs, true_types):
            _, origin_type = STANDARD_LAYOUT[(r, c)]
            pos = r * 9 + c
            self._cells[pos] = Piece(
                color=color,
                origin_type=origin_type,
                true_type=true_type,
                revealed=False,
            )

    # ---- accessors --------------------------------------------------------

    @property
    def cells(self) -> tuple[Piece | None, ...]:
        """Immutable view of all 90 board cells."""
        return tuple(self._cells)

    def get_piece(self, pos: int) -> Piece | None:
        """Return the piece at the given linear position."""
        return self._cells[pos]

    def __getitem__(self, pos: int) -> Piece | None:
        return self._cells[pos]

    def pieces_of(self, color: Color) -> list[tuple[int, Piece]]:
        """Return ``[(pos, piece), ...]`` for all pieces of *color*."""
        return [
            (pos, p)
            for pos, p in enumerate(self._cells)
            if p is not None and p.color == color
        ]

    def king_pos(self, color: Color) -> int:
        """Return the linear position of *color*'s King."""
        for pos, piece in enumerate(self._cells):
            if piece is not None and piece.is_king and piece.color == color:
                return pos
        raise ValueError(f"King not found for {color}")

    def revealed_pieces(self) -> list[tuple[int, Piece]]:
        """Return all revealed pieces as ``[(pos, piece), ...]``."""
        return [
            (pos, p)
            for pos, p in enumerate(self._cells)
            if p is not None and p.revealed
        ]

    def set_cell(self, pos: int, piece: Piece | None) -> None:
        """Directly set a cell value (for testing / position setup)."""
        self._cells[pos] = piece

    # ---- state mutation ----------------------------------------------------

    def apply_move(self, move: Move) -> Piece | None:
        """Execute *move* on the board.

        Handles piece movement, hidden-piece reveal on first move, captures,
        and turn switching.  Returns the captured piece or ``None``.
        """
        from dataclasses import replace

        from_pos, to_pos = move.from_pos, move.to_pos
        moving = self._cells[from_pos]
        if moving is None:
            raise ValueError(f"No piece at position {from_pos}")

        # Reveal hidden piece on its first move
        if not moving.revealed:
            moving = replace(moving, revealed=True)

        captured = self._cells[to_pos]

        self._cells[to_pos] = moving
        self._cells[from_pos] = None

        if captured is not None:
            self._captured.append(captured)

        self._turn = self._turn.opposite()
        return captured

    @property
    def captured(self) -> list[Piece]:
        """List of pieces captured so far, in order."""
        return list(self._captured)

    @property
    def turn(self) -> Color:
        """The player whose turn it is to move."""
        return self._turn

    # ---- query helpers ------------------------------------------------------

    def hidden_pieces(self) -> list[tuple[int, Piece]]:
        """Return all hidden pieces as ``[(pos, piece), ...]``."""
        return [
            (pos, p)
            for pos, p in enumerate(self._cells)
            if p is not None and not p.revealed
        ]
