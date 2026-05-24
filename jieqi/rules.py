from __future__ import annotations

from .board import Board
from .constants import BOARD_COLS, BOARD_ROWS, Color, PieceType
from .move import Move, is_valid_rc, pos_to_rc, rc_to_pos

# ==============================================================================
#  Palace / river helpers
# ==============================================================================


def _in_palace(row: int, col: int, color: Color) -> bool:
    """Check whether (row, col) is inside *color*'s palace."""
    if color == Color.RED:
        return 7 <= row <= 9 and 3 <= col <= 5
    else:
        return 0 <= row <= 2 and 3 <= col <= 5


def _has_crossed_river(row: int, color: Color) -> bool:
    """Check whether a pawn of *color* at *row* has crossed the river."""
    if color == Color.RED:
        return row <= 4
    else:
        return row >= 5


# ==============================================================================
#  Cell classification
# ==============================================================================


def _can_occupy(board: Board, pos: int, color: Color) -> bool:
    """True if *pos* is empty or occupied by an opponent piece."""
    p = board[pos]
    return p is None or p.color != color


def _is_enemy(board: Board, pos: int, color: Color) -> bool:
    p = board[pos]
    return p is not None and p.color != color


def _is_empty(board: Board, pos: int) -> bool:
    return board[pos] is None


# ==============================================================================
#  Ray-cast helper (Rook / Cannon)
# ==============================================================================


def _count_between(board: Board, from_pos: int, to_pos: int) -> int:
    """Count pieces strictly between *from_pos* and *to_pos* on a straight line."""
    fr, fc = pos_to_rc(from_pos)
    tr, tc = pos_to_rc(to_pos)
    count = 0
    if fr == tr:
        step = 1 if tc > fc else -1
        for c in range(fc + step, tc, step):
            if board[rc_to_pos(fr, c)] is not None:
                count += 1
    elif fc == tc:
        step = 1 if tr > fr else -1
        for r in range(fr + step, tr, step):
            if board[rc_to_pos(r, fc)] is not None:
                count += 1
    return count


# ==============================================================================
#  Per-piece-type pseudo-legal move generators
# ==============================================================================

_KING_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _king_moves(board: Board, row: int, col: int, color: Color, revealed: bool = True) -> list[Move]:
    moves: list[Move] = []
    from_pos = rc_to_pos(row, col)
    for dr, dc in _KING_DIRS:
        nr, nc = row + dr, col + dc
        if not is_valid_rc(nr, nc):
            continue
        # Palace restriction only for revealed king (§Jieqi: hidden can leave)
        if revealed and not _in_palace(nr, nc, color):
            continue
        to_pos = rc_to_pos(nr, nc)
        if _can_occupy(board, to_pos, color):
            moves.append(Move(from_pos, to_pos))
    return moves


_ADVISOR_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _advisor_moves(board: Board, row: int, col: int, color: Color, revealed: bool = True) -> list[Move]:
    moves: list[Move] = []
    from_pos = rc_to_pos(row, col)
    for dr, dc in _ADVISOR_DIRS:
        nr, nc = row + dr, col + dc
        if not is_valid_rc(nr, nc):
            continue
        # §Jieqi: advisors can leave the palace (unlike standard Xiangqi)
        to_pos = rc_to_pos(nr, nc)
        if _can_occupy(board, to_pos, color):
            moves.append(Move(from_pos, to_pos))
    return moves


_ELEPHANT_DIRS = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
_ELEPHANT_EYES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _elephant_moves(board: Board, row: int, col: int, color: Color, revealed: bool = True) -> list[Move]:
    moves: list[Move] = []
    from_pos = rc_to_pos(row, col)
    for (dr, dc), (er, ec) in zip(_ELEPHANT_DIRS, _ELEPHANT_EYES):
        nr, nc = row + dr, col + dc
        eye_r, eye_c = row + er, col + ec
        if not is_valid_rc(nr, nc):
            continue
        # §Jieqi: elephants CAN cross the river (unlike standard Xiangqi)
        # Eye blocked
        if board[rc_to_pos(eye_r, eye_c)] is not None:
            continue
        to_pos = rc_to_pos(nr, nc)
        if _can_occupy(board, to_pos, color):
            moves.append(Move(from_pos, to_pos))
    return moves


# Horse: (leg_dr, leg_dc, target_dr, target_dc)
_HORSE_STEPS = [
    (-1, 0, -2, -1),  # up leg  → up-left
    (-1, 0, -2, 1),   # up leg  → up-right
    (1, 0, 2, -1),    # down leg → down-left
    (1, 0, 2, 1),     # down leg → down-right
    (0, -1, -1, -2),  # left leg  → up-left
    (0, -1, 1, -2),   # left leg  → down-left
    (0, 1, -1, 2),    # right leg → up-right
    (0, 1, 1, 2),     # right leg → down-right
]


def _horse_moves(board: Board, row: int, col: int, color: Color, revealed: bool = True) -> list[Move]:
    moves: list[Move] = []
    from_pos = rc_to_pos(row, col)
    for lr, lc, tr, tc in _HORSE_STEPS:
        leg_r, leg_c = row + lr, col + lc
        target_r, target_c = row + tr, col + tc
        if not is_valid_rc(target_r, target_c):
            continue
        # Leg blocked → cannot jump
        if board[rc_to_pos(leg_r, leg_c)] is not None:
            continue
        to_pos = rc_to_pos(target_r, target_c)
        if _can_occupy(board, to_pos, color):
            moves.append(Move(from_pos, to_pos))
    return moves


_ROOK_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _rook_moves(board: Board, row: int, col: int, color: Color, revealed: bool = True) -> list[Move]:
    moves: list[Move] = []
    from_pos = rc_to_pos(row, col)
    for dr, dc in _ROOK_DIRS:
        nr, nc = row + dr, col + dc
        while is_valid_rc(nr, nc):
            to_pos = rc_to_pos(nr, nc)
            if _is_empty(board, to_pos):
                moves.append(Move(from_pos, to_pos))
            else:
                if _is_enemy(board, to_pos, color):
                    moves.append(Move(from_pos, to_pos))
                break  # blocked by any piece
            nr += dr
            nc += dc
    return moves


def _cannon_moves(board: Board, row: int, col: int, color: Color, revealed: bool = True) -> list[Move]:
    moves: list[Move] = []
    from_pos = rc_to_pos(row, col)
    for dr, dc in _ROOK_DIRS:
        nr, nc = row + dr, col + dc
        # Phase 1: non-capture movement (stop before first piece)
        while is_valid_rc(nr, nc):
            to_pos = rc_to_pos(nr, nc)
            if not _is_empty(board, to_pos):
                break  # hit the screen (or blocker)
            moves.append(Move(from_pos, to_pos))
            nr += dr
            nc += dc
        # Phase 2: capture behind the screen
        if is_valid_rc(nr, nc):
            # Skip the screen piece itself
            nr += dr
            nc += dc
            # Look for the first piece behind the screen
            while is_valid_rc(nr, nc):
                to_pos = rc_to_pos(nr, nc)
                if _is_empty(board, to_pos):
                    nr += dr
                    nc += dc
                    continue
                if _is_enemy(board, to_pos, color):
                    moves.append(Move(from_pos, to_pos))
                break  # first piece found (or blocked by own piece)
    return moves


def _pawn_moves(board: Board, row: int, col: int, color: Color, revealed: bool = True) -> list[Move]:
    moves: list[Move] = []
    from_pos = rc_to_pos(row, col)
    crossed = _has_crossed_river(row, color)
    if color == Color.RED:
        forward = (row - 1, col)
        if is_valid_rc(*forward) and _can_occupy(board, rc_to_pos(*forward), color):
            moves.append(Move(from_pos, rc_to_pos(*forward)))
        if crossed:
            for dc in (-1, 1):
                side = (row, col + dc)
                if is_valid_rc(*side) and _can_occupy(board, rc_to_pos(*side), color):
                    moves.append(Move(from_pos, rc_to_pos(*side)))
    else:
        forward = (row + 1, col)
        if is_valid_rc(*forward) and _can_occupy(board, rc_to_pos(*forward), color):
            moves.append(Move(from_pos, rc_to_pos(*forward)))
        if crossed:
            for dc in (-1, 1):
                side = (row, col + dc)
                if is_valid_rc(*side) and _can_occupy(board, rc_to_pos(*side), color):
                    moves.append(Move(from_pos, rc_to_pos(*side)))
    return moves


_GENERATORS = {
    PieceType.KING: _king_moves,
    PieceType.ADVISOR: _advisor_moves,
    PieceType.ELEPHANT: _elephant_moves,
    PieceType.HORSE: _horse_moves,
    PieceType.ROOK: _rook_moves,
    PieceType.CANNON: _cannon_moves,
    PieceType.PAWN: _pawn_moves,
}

# ==============================================================================
#  Public API
# ==============================================================================


def generate_piece_moves(board: Board, row: int, col: int) -> list[Move]:
    """Generate pseudo-legal moves for the piece at (row, col).

    Pseudo-legal means the move obeys piece-specific rules (movement pattern,
    blocking, palace/river constraints) but may leave the own king in check
    or cause the two kings to face each other.

    In Jieqi, hidden pieces CAN cross the river / leave the palace —
    they follow the movement *pattern* of their origin_type but without
    zone restrictions, since their true identity is unknown.
    """
    piece = board[rc_to_pos(row, col)]
    if piece is None:
        return []

    gen = _GENERATORS.get(piece.effective_type)
    if gen is None:
        return []
    return gen(board, row, col, piece.color, piece.revealed)


# ==============================================================================
#  Legality filtering
# ==============================================================================


def _is_in_check(board: Board, color: Color) -> bool:
    """Return True if *color*'s king is under attack."""
    try:
        king_pos = board.king_pos(color)
    except ValueError:
        return True  # king missing → effectively in check
    opponent = color.opposite()
    for _pos, piece in board.pieces_of(opponent):
        r, c = pos_to_rc(_pos)
        for mv in generate_piece_moves(board, r, c):
            if mv.to_pos == king_pos:
                return True
    return False


def _kings_are_facing(board: Board) -> bool:
    """Return True if the two kings face each other on the same file with nothing between."""
    try:
        rp = board.king_pos(Color.RED)
        bp = board.king_pos(Color.BLACK)
    except ValueError:
        return False
    rr, rc = pos_to_rc(rp)
    br, bc = pos_to_rc(bp)
    if rc != bc:
        return False
    lo, hi = (rr, br) if rr < br else (br, rr)
    for r in range(lo + 1, hi):
        if board[rc_to_pos(r, rc)] is not None:
            return False
    return True


def _is_legal_after_move(board: Board, move: Move, color: Color) -> bool:
    """Check whether executing *move* leaves *color*'s king safe."""
    cells = board._cells  # internal list — mutated temporarily
    fpos, tpos = move.from_pos, move.to_pos

    moving = cells[fpos]
    captured = cells[tpos]

    cells[tpos] = moving
    cells[fpos] = None

    try:
        if _is_in_check(board, color):
            return False
        if _kings_are_facing(board):
            return False
        return True
    finally:
        cells[fpos] = moving
        cells[tpos] = captured


def generate_legal_moves(board: Board, color: Color) -> list[Move]:
    """Return all fully legal moves for *color*.

    A move is legal iff it is pseudo-legal for the piece AND does not leave
    the moving side's king in check or cause the two kings to face each other.
    """
    legal: list[Move] = []
    for _pos, piece in board.pieces_of(color):
        r, c = pos_to_rc(_pos)
        for mv in generate_piece_moves(board, r, c):
            if _is_legal_after_move(board, mv, color):
                legal.append(mv)
    return legal
