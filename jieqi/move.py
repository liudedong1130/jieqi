from __future__ import annotations

from dataclasses import dataclass

from .constants import BOARD_COLS, BOARD_ROWS, BOARD_SIZE, NUM_ACTIONS


def pos_to_rc(pos: int) -> tuple[int, int]:
    """Convert linear position (0–89) to (row, col)."""
    return pos // BOARD_COLS, pos % BOARD_COLS


def rc_to_pos(row: int, col: int) -> int:
    """Convert (row, col) to linear position (0–89)."""
    return row * BOARD_COLS + col


def encode_action(from_pos: int, to_pos: int) -> int:
    """Encode a from-to pair into a single action index [0, 8099]."""
    return from_pos * BOARD_SIZE + to_pos


def decode_action(action: int) -> tuple[int, int]:
    """Decode an action index into (from_pos, to_pos)."""
    return action // BOARD_SIZE, action % BOARD_SIZE


def is_valid_pos(pos: int) -> bool:
    """Check if a linear position is on the board."""
    return 0 <= pos < BOARD_SIZE


def is_valid_rc(row: int, col: int) -> bool:
    """Check if (row, col) is on the board."""
    return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS


@dataclass(frozen=True)
class Move:
    from_pos: int
    to_pos: int

    def __post_init__(self) -> None:
        if not is_valid_pos(self.from_pos):
            raise ValueError(f"from_pos out of range [0, {BOARD_SIZE}): {self.from_pos}")
        if not is_valid_pos(self.to_pos):
            raise ValueError(f"to_pos out of range [0, {BOARD_SIZE}): {self.to_pos}")

    def __repr__(self) -> str:
        fr, fc = pos_to_rc(self.from_pos)
        tr, tc = pos_to_rc(self.to_pos)
        return f"Move({fr},{fc} -> {tr},{tc})"
