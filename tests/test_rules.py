from __future__ import annotations

import pytest

from jieqi import BOARD_SIZE, Color, Piece, PieceType
from jieqi.board import Board
from jieqi.move import pos_to_rc, rc_to_pos
from jieqi.rules import (
    _is_in_check,
    _kings_are_facing,
    generate_legal_moves,
    generate_piece_moves,
)


# ---------------------------------------------------------------------------
#  Test helpers
# ---------------------------------------------------------------------------


def _make_empty_board() -> Board:
    """Return a completely empty Board."""
    b = Board()
    for pos in range(BOARD_SIZE):
        b.set_cell(pos, None)
    return b


def _piece(color: Color, ptype: PieceType, revealed: bool = True) -> Piece:
    """Shorthand to create a Piece with origin_type == true_type."""
    return Piece(color=color, origin_type=ptype, true_type=ptype, revealed=revealed)


def _place(board: Board, row: int, col: int, piece: Piece) -> None:
    board.set_cell(rc_to_pos(row, col), piece)


# ---------------------------------------------------------------------------
#  Horse (马) tests
# ---------------------------------------------------------------------------


class TestHorse:
    def test_horse_leg_blocked_cannot_move(self) -> None:
        """马腿被堵不能走"""
        board = _make_empty_board()
        # Red horse at center (5, 4)
        _place(board, 5, 4, _piece(Color.RED, PieceType.HORSE))
        # Blocker at up-leg (4, 4)
        _place(board, 4, 4, _piece(Color.RED, PieceType.PAWN))

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}

        # Up-left (3,3) and up-right (3,5) blocked by leg at (4,4)
        assert rc_to_pos(3, 3) not in targets
        assert rc_to_pos(3, 5) not in targets

    def test_horse_leg_clear_can_move(self) -> None:
        """马腿没堵可以走"""
        board = _make_empty_board()
        _place(board, 5, 4, _piece(Color.RED, PieceType.HORSE))

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}

        # All 8 positions should be reachable (board edges permitting)
        # Since (5,4) is center, all 8 are on board
        expected = {
            rc_to_pos(3, 3), rc_to_pos(3, 5),  # up-left, up-right
            rc_to_pos(7, 3), rc_to_pos(7, 5),  # down-left, down-right
            rc_to_pos(4, 2), rc_to_pos(6, 2),  # left-up, left-down
            rc_to_pos(4, 6), rc_to_pos(6, 6),  # right-up, right-down
        }
        assert targets == expected

    def test_horse_near_edge_truncates(self) -> None:
        """马在边角时走法被棋盘截断"""
        board = _make_empty_board()
        # Horse at corner (0, 0)
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.HORSE))

        moves = generate_piece_moves(board, 0, 0)
        targets = {m.to_pos for m in moves}
        # From (0,0): left leg blocked by board edge, up leg blocked by edge
        # Only down-right (2,1) and right-down (1,2) reachable
        expected = {rc_to_pos(2, 1), rc_to_pos(1, 2)}
        assert targets == expected

    def test_horse_cannot_capture_own(self) -> None:
        """马不能吃己方棋子"""
        board = _make_empty_board()
        _place(board, 5, 4, _piece(Color.RED, PieceType.HORSE))
        _place(board, 3, 3, _piece(Color.RED, PieceType.PAWN))  # own piece at target

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(3, 3) not in targets

    def test_horse_can_capture_enemy(self) -> None:
        """马可以吃对方棋子"""
        board = _make_empty_board()
        _place(board, 5, 4, _piece(Color.RED, PieceType.HORSE))
        _place(board, 3, 3, _piece(Color.BLACK, PieceType.PAWN))  # enemy at target

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(3, 3) in targets


# ---------------------------------------------------------------------------
#  Elephant (象) tests
# ---------------------------------------------------------------------------


class TestElephant:
    def test_elephant_eye_blocked(self) -> None:
        """象眼被堵不能走"""
        board = _make_empty_board()
        _place(board, 7, 2, _piece(Color.RED, PieceType.ELEPHANT))
        # Blocker at eye (6, 1) — blocks (5, 0) target
        _place(board, 6, 1, _piece(Color.RED, PieceType.PAWN))

        moves = generate_piece_moves(board, 7, 2)
        targets = {m.to_pos for m in moves}

        assert rc_to_pos(5, 0) not in targets  # eye at (6,1) blocked
        # Other targets should still be reachable
        assert rc_to_pos(5, 4) in targets  # eye (6,3) clear
        assert rc_to_pos(9, 0) in targets  # eye (8,1) clear
        assert rc_to_pos(9, 4) in targets  # eye (8,3) clear

    def test_elephant_cannot_cross_river(self) -> None:
        """象不能过河"""
        board = _make_empty_board()
        # Red elephant at row 5 (right at river edge)
        _place(board, 5, 2, _piece(Color.RED, PieceType.ELEPHANT))

        moves = generate_piece_moves(board, 5, 2)
        targets = {m.to_pos for m in moves}
        # Target (3,0) would be across the river → not allowed
        assert rc_to_pos(3, 0) not in targets
        # Target (3,4) → row 3 is across the river → not allowed
        assert rc_to_pos(3, 4) not in targets
        # Target (7,0) and (7,4) → rows 7 are on own side → allowed
        assert rc_to_pos(7, 0) in targets
        assert rc_to_pos(7, 4) in targets

    def test_elephant_eye_clear_and_on_own_side(self) -> None:
        """象眼干净且不过河时可以走"""
        board = _make_empty_board()
        _place(board, 7, 2, _piece(Color.RED, PieceType.ELEPHANT))

        moves = generate_piece_moves(board, 7, 2)
        targets = {m.to_pos for m in moves}
        # From (7,2), all targets are on Red side (rows 5-9)
        assert rc_to_pos(5, 0) in targets
        assert rc_to_pos(5, 4) in targets
        assert rc_to_pos(9, 0) in targets
        assert rc_to_pos(9, 4) in targets


# ---------------------------------------------------------------------------
#  Advisor (士/仕) tests
# ---------------------------------------------------------------------------


class TestAdvisor:
    def test_advisor_only_diagonal_in_palace(self) -> None:
        """士只能走九宫斜线"""
        board = _make_empty_board()
        _place(board, 8, 4, _piece(Color.RED, PieceType.ADVISOR))

        moves = generate_piece_moves(board, 8, 4)
        targets = {m.to_pos for m in moves}
        # From (8,4) in Red palace, diagonal to (7,3), (7,5), (9,3), (9,5)
        expected = {rc_to_pos(7, 3), rc_to_pos(7, 5), rc_to_pos(9, 3), rc_to_pos(9, 5)}
        assert targets == expected

    def test_advisor_cannot_leave_palace(self) -> None:
        """士不能离开九宫"""
        board = _make_empty_board()
        # Advisor at palace edge (9, 3)
        _place(board, 9, 3, _piece(Color.RED, PieceType.ADVISOR))

        moves = generate_piece_moves(board, 9, 3)
        targets = {m.to_pos for m in moves}
        # Only diagonal to (8, 4) — (10, 2) and (10, 4) are off-board
        # (8, 2) is outside palace (col 2 < 3) — wait, (8,2) is diagonal from (9,3)?
        # (9,3) → (8,2) = (-1, -1) ✓ but col 2 is outside palace.
        # (9,3) → (8,4) = (-1, +1) ✓ col 4 is in palace.
        # (9,3) → (10,2) = (+1, -1) off-board.
        # (9,3) → (10,4) = (+1, +1) off-board.
        assert rc_to_pos(8, 2) not in targets  # outside palace col
        assert rc_to_pos(8, 4) in targets

    def test_black_advisor_in_black_palace(self) -> None:
        board = _make_empty_board()
        _place(board, 1, 4, _piece(Color.BLACK, PieceType.ADVISOR))

        moves = generate_piece_moves(board, 1, 4)
        targets = {m.to_pos for m in moves}
        expected = {rc_to_pos(0, 3), rc_to_pos(0, 5), rc_to_pos(2, 3), rc_to_pos(2, 5)}
        assert targets == expected


# ---------------------------------------------------------------------------
#  King (帅/将) tests
# ---------------------------------------------------------------------------


class TestKing:
    def test_king_stays_in_palace(self) -> None:
        """帅/将只能在九宫"""
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))

        moves = generate_piece_moves(board, 9, 4)
        targets = {m.to_pos for m in moves}
        # (9,4) → can go to (8,4), (9,3), (9,5) — (10,4) off-board
        expected = {rc_to_pos(8, 4), rc_to_pos(9, 3), rc_to_pos(9, 5)}
        assert targets == expected

    def test_king_cannot_leave_palace(self) -> None:
        board = _make_empty_board()
        # King at palace corner (9, 3)
        _place(board, 9, 3, _piece(Color.RED, PieceType.KING))

        moves = generate_piece_moves(board, 9, 3)
        targets = {m.to_pos for m in moves}
        # (9,2) is outside palace
        assert rc_to_pos(9, 2) not in targets
        # Valid: (8,3), (9,4)
        assert rc_to_pos(8, 3) in targets
        assert rc_to_pos(9, 4) in targets

    def test_king_cannot_move_into_check(self) -> None:
        """King cannot move into a position attacked by opponent."""
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        # Black rook on same file (row 0, col 4) attacks row 8 and row 9
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.ROOK))

        legal = generate_legal_moves(board, Color.RED)
        targets = {m.to_pos for m in legal}
        # King at (9,4); (8,4) is attacked by rook → illegal
        assert rc_to_pos(8, 4) not in targets
        # (9,3) and (9,5) should be legal (king moves off the file)
        assert rc_to_pos(9, 3) in targets
        assert rc_to_pos(9, 5) in targets


# ---------------------------------------------------------------------------
#  Rook (车) tests
# ---------------------------------------------------------------------------


class TestRook:
    def test_rook_cannot_jump_over_pieces(self) -> None:
        """车不能越子"""
        board = _make_empty_board()
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK))
        _place(board, 5, 3, _piece(Color.RED, PieceType.PAWN))  # blocker

        moves = generate_piece_moves(board, 5, 0)
        targets = {m.to_pos for m in moves}

        # Can move up to (not including) blocker: (5,1), (5,2)
        assert rc_to_pos(5, 1) in targets
        assert rc_to_pos(5, 2) in targets
        # Cannot pass blocker
        assert rc_to_pos(5, 3) not in targets  # own piece
        assert rc_to_pos(5, 4) not in targets
        assert rc_to_pos(5, 8) not in targets

    def test_rook_can_capture_enemy(self) -> None:
        board = _make_empty_board()
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK))
        _place(board, 5, 3, _piece(Color.BLACK, PieceType.PAWN))

        moves = generate_piece_moves(board, 5, 0)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(5, 3) in targets  # can capture enemy
        assert rc_to_pos(5, 4) not in targets  # can't go beyond

    def test_rook_vertical_movement(self) -> None:
        board = _make_empty_board()
        _place(board, 5, 4, _piece(Color.RED, PieceType.ROOK))

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}
        # Vertical: all rows except own row
        for r in range(10):
            if r != 5:
                assert rc_to_pos(r, 4) in targets
        # Horizontal: all cols except own col
        for c in range(9):
            if c != 4:
                assert rc_to_pos(5, c) in targets

    def test_rook_blocked_by_piece_stops_before(self) -> None:
        """车在对方棋子前停止"""
        board = _make_empty_board()
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.ROOK))
        _place(board, 5, 0, _piece(Color.BLACK, PieceType.PAWN))

        moves = generate_piece_moves(board, 0, 0)
        targets = {m.to_pos for m in moves}
        # Can reach own pawn, but not beyond
        assert rc_to_pos(5, 0) not in targets  # own piece
        for r in range(1, 5):
            assert rc_to_pos(r, 0) in targets
        assert rc_to_pos(6, 0) not in targets


# ---------------------------------------------------------------------------
#  Cannon (炮) tests
# ---------------------------------------------------------------------------


class TestCannon:
    def test_cannon_non_capture_cannot_jump(self) -> None:
        """炮不吃子时不能越子"""
        board = _make_empty_board()
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.CANNON))
        _place(board, 3, 0, _piece(Color.BLACK, PieceType.PAWN))  # screen

        moves = generate_piece_moves(board, 0, 0)
        targets = {m.to_pos for m in moves}
        # Can move to (1,0), (2,0)
        assert rc_to_pos(1, 0) in targets
        assert rc_to_pos(2, 0) in targets
        # Cannot move to (3,0) or beyond as non-capture
        assert rc_to_pos(3, 0) not in targets
        assert rc_to_pos(4, 0) not in targets

    def test_cannon_capture_must_have_exactly_one_screen(self) -> None:
        """炮吃子必须隔一个炮架"""
        board = _make_empty_board()
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.CANNON))
        _place(board, 3, 0, _piece(Color.RED, PieceType.PAWN))  # screen (any piece)
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK))  # target

        moves = generate_piece_moves(board, 0, 0)
        targets = {m.to_pos for m in moves}
        # Can capture at (5,0) via screen at (3,0)
        assert rc_to_pos(5, 0) in targets

    def test_cannon_cannot_capture_without_screen(self) -> None:
        board = _make_empty_board()
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.CANNON))
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK))  # no screen!

        moves = generate_piece_moves(board, 0, 0)
        targets = {m.to_pos for m in moves}
        # Can move up to (4,0), but cannot capture at (5,0)
        for r in range(1, 5):
            assert rc_to_pos(r, 0) in targets
        assert rc_to_pos(5, 0) not in targets  # no screen, can't capture

    def test_cannon_cannot_capture_with_two_screens(self) -> None:
        board = _make_empty_board()
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.CANNON))
        _place(board, 2, 0, _piece(Color.BLACK, PieceType.PAWN))
        _place(board, 4, 0, _piece(Color.RED, PieceType.PAWN))  # second piece
        _place(board, 6, 0, _piece(Color.RED, PieceType.ROOK))

        moves = generate_piece_moves(board, 0, 0)
        targets = {m.to_pos for m in moves}
        # Non-capture: can reach (1,0) only (stopped by first piece at (2,0))
        assert rc_to_pos(1, 0) in targets
        # Capture: screen at (2,0), target behind it at (4,0) is reachable (first piece after screen)
        assert rc_to_pos(4, 0) in targets  # can capture at (4,0)
        # But (6,0) is NOT reachable because the cannon stops at the first piece after the screen
        assert rc_to_pos(6, 0) not in targets

    def test_cannon_cannot_capture_own_piece_behind_screen(self) -> None:
        board = _make_empty_board()
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.CANNON))
        _place(board, 3, 0, _piece(Color.RED, PieceType.PAWN))  # screen
        _place(board, 5, 0, _piece(Color.BLACK, PieceType.ROOK))  # own piece behind

        moves = generate_piece_moves(board, 0, 0)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(5, 0) not in targets  # own piece, can't capture


# ---------------------------------------------------------------------------
#  Pawn (兵/卒) tests
# ---------------------------------------------------------------------------


class TestPawn:
    def test_pawn_before_river_no_sideways(self) -> None:
        """兵过河前不能左右走"""
        # Red pawn at row 6 (before crossing; river crossing at row <= 4)
        board = _make_empty_board()
        _place(board, 6, 4, _piece(Color.RED, PieceType.PAWN))

        moves = generate_piece_moves(board, 6, 4)
        targets = {m.to_pos for m in moves}
        # Only forward: (5, 4)
        assert rc_to_pos(5, 4) in targets
        assert rc_to_pos(6, 3) not in targets  # left
        assert rc_to_pos(6, 5) not in targets  # right

    def test_pawn_after_river_can_sideways(self) -> None:
        """兵过河后可以左右走"""
        board = _make_empty_board()
        # Red pawn at row 4 (has crossed to row 4)
        _place(board, 4, 4, _piece(Color.RED, PieceType.PAWN))

        moves = generate_piece_moves(board, 4, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(3, 4) in targets  # forward
        assert rc_to_pos(4, 3) in targets  # left
        assert rc_to_pos(4, 5) in targets  # right

    def test_pawn_never_backward(self) -> None:
        """兵永远不能后退"""
        board = _make_empty_board()
        _place(board, 3, 4, _piece(Color.RED, PieceType.PAWN))

        moves = generate_piece_moves(board, 3, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(4, 4) not in targets  # backward

    def test_black_pawn_moves_downward(self) -> None:
        board = _make_empty_board()
        _place(board, 3, 4, _piece(Color.BLACK, PieceType.PAWN))

        moves = generate_piece_moves(board, 3, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(4, 4) in targets  # forward (row+1 for Black)

    def test_black_pawn_after_river_can_sideways(self) -> None:
        board = _make_empty_board()
        _place(board, 5, 4, _piece(Color.BLACK, PieceType.PAWN))

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(6, 4) in targets  # forward
        assert rc_to_pos(5, 3) in targets  # left
        assert rc_to_pos(5, 5) in targets  # right


# ---------------------------------------------------------------------------
#  Kings-facing (将帅照面) tests
# ---------------------------------------------------------------------------


class TestKingsFacing:
    def test_kings_facing_detected(self) -> None:
        """两个王在同列无障碍物时检测为照面"""
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING))

        assert _kings_are_facing(board) is True

    def test_kings_not_facing_with_blocker(self) -> None:
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING))
        _place(board, 5, 4, _piece(Color.RED, PieceType.PAWN))  # blocker

        assert _kings_are_facing(board) is False

    def test_kings_not_facing_different_columns(self) -> None:
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 3, _piece(Color.BLACK, PieceType.KING))

        assert _kings_are_facing(board) is False

    def test_move_causing_kings_facing_is_illegal(self) -> None:
        """将帅照面时相关走法非法"""
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING))
        # Rook at (5, 4) is the ONLY blocker between the two kings on col 4
        _place(board, 5, 4, _piece(Color.RED, PieceType.ROOK))

        legal = generate_legal_moves(board, Color.RED)
        legal_targets = {m.to_pos for m in legal}

        # Rook is pinned: moving off col 4 would leave kings facing → ILLEGAL
        assert rc_to_pos(5, 0) not in legal_targets
        assert rc_to_pos(5, 8) not in legal_targets
        # Rook can move along col 4, staying between the kings
        assert rc_to_pos(3, 4) in legal_targets
        assert rc_to_pos(7, 4) in legal_targets


# ---------------------------------------------------------------------------
#  Capture rules
# ---------------------------------------------------------------------------


class TestCapture:
    def test_cannot_capture_own_piece(self) -> None:
        """不能吃己方棋子"""
        board = _make_empty_board()
        _place(board, 5, 4, _piece(Color.RED, PieceType.ROOK))
        _place(board, 5, 6, _piece(Color.RED, PieceType.PAWN))

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(5, 6) not in targets

    def test_can_capture_enemy_piece(self) -> None:
        """可以吃对方棋子"""
        board = _make_empty_board()
        _place(board, 5, 4, _piece(Color.RED, PieceType.ROOK))
        _place(board, 5, 6, _piece(Color.BLACK, PieceType.PAWN))

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(5, 6) in targets


# ---------------------------------------------------------------------------
#  Legal moves — check safety
# ---------------------------------------------------------------------------


class TestLegalMoves:
    def test_king_cannot_move_into_check(self) -> None:
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.ROOK))  # attacks file 4

        legal = generate_legal_moves(board, Color.RED)
        targets = {m.to_pos for m in legal}
        assert rc_to_pos(8, 4) not in targets  # stays on attacked file

    def test_pinned_piece_cannot_move_off_line(self) -> None:
        """Absolute pin: a piece shielding king from rook cannot leave the line."""
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 7, 4, _piece(Color.RED, PieceType.ADVISOR))  # shields king
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.ROOK))

        legal = generate_legal_moves(board, Color.RED)
        legal_moves = {(m.from_pos, m.to_pos) for m in legal}
        advisor_pos = rc_to_pos(7, 4)

        # Advisor at (7,4) is pinned on file 4. Can it move off the file?
        off_file_moves = [
            t for f, t in legal_moves
            if f == advisor_pos and pos_to_rc(t)[1] != 4
        ]
        assert len(off_file_moves) == 0, "Pinned advisor should not move off the file"

    def test_legal_moves_includes_moves_for_all_revealed_pieces(self) -> None:
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 9, 0, _piece(Color.RED, PieceType.ROOK))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING))
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.ROOK))

        red_legal = generate_legal_moves(board, Color.RED)
        assert len(red_legal) > 0

    def test_is_in_check_detected(self) -> None:
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.ROOK))

        assert _is_in_check(board, Color.RED) is True

    def test_is_not_in_check(self) -> None:
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 0, _piece(Color.BLACK, PieceType.ROOK))

        assert _is_in_check(board, Color.RED) is False


# ---------------------------------------------------------------------------
#  Hidden pieces / empty cell
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_hidden_piece_generates_moves_by_origin_type(self) -> None:
        """Hidden pieces generate moves according to origin_type."""
        board = _make_empty_board()
        # Place Kings to avoid check/kings-facing issues
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING))
        # Hidden rook with origin_type=ROOK
        _place(board, 5, 4, _piece(Color.RED, PieceType.ROOK, revealed=False))

        moves = generate_piece_moves(board, 5, 4)
        # Should have rook-like moves (straight lines)
        targets = {m.to_pos for m in moves}
        assert len(moves) > 0
        # Rook at (5,4) on empty board: can reach all rows on col 4 and all cols on row 5
        # But (0,4) has Black King, (9,4) has Red King
        assert rc_to_pos(0, 4) in targets  # can capture Black King
        assert rc_to_pos(9, 4) not in targets  # can't capture own King
        assert rc_to_pos(5, 0) in targets  # left horizontal

    def test_empty_cell_generates_no_moves(self) -> None:
        board = _make_empty_board()
        moves = generate_piece_moves(board, 0, 0)
        assert moves == []

    def test_hidden_piece_contributes_legal_moves(self) -> None:
        """Hidden pieces now contribute to legal moves."""
        board = _make_empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING))
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, revealed=False))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING))

        legal = generate_legal_moves(board, Color.RED)
        from_positions = {m.from_pos for m in legal}
        # Both hidden rook and king should contribute
        assert rc_to_pos(5, 0) in from_positions  # hidden rook used
        assert rc_to_pos(9, 4) in from_positions  # king used
