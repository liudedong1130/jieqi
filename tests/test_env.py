from __future__ import annotations

import pytest

from jieqi import BOARD_SIZE, Color, Piece, PieceType
from jieqi.board import Board
from jieqi.move import Move, pos_to_rc, rc_to_pos
from jieqi.rules import generate_legal_moves, generate_piece_moves


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _piece(color: Color, origin: PieceType, true: PieceType, revealed: bool = False) -> Piece:
    return Piece(color=color, origin_type=origin, true_type=true, revealed=revealed)


def _empty_board() -> Board:
    b = Board()
    for pos in range(BOARD_SIZE):
        b.set_cell(pos, None)
    return b


def _place(board: Board, row: int, col: int, piece: Piece) -> None:
    board.set_cell(rc_to_pos(row, col), piece)


# ---------------------------------------------------------------------------
#  Hidden piece move generation
# ---------------------------------------------------------------------------


class TestHiddenMoveGeneration:
    def test_hidden_rook_moves_like_rook(self) -> None:
        """暗车位第一次移动时可以按车走"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        # Hidden piece at rook position
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False))

        moves = generate_piece_moves(board, 5, 0)
        targets = {m.to_pos for m in moves}
        # Rook-like: can move vertically
        for r in range(10):
            if r != 5:
                assert rc_to_pos(r, 0) in targets
        # Can move horizontally
        assert rc_to_pos(5, 8) in targets

    def test_hidden_piece_reveals_after_move(self) -> None:
        """暗车 true_type=PAWN，移动揭开后变成兵"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))

        move = Move(rc_to_pos(5, 0), rc_to_pos(5, 3))
        board.apply_move(move)

        moved = board[rc_to_pos(5, 3)]
        assert moved is not None
        assert moved.revealed is True
        assert moved.true_type == PieceType.PAWN
        assert moved.effective_type == PieceType.PAWN
        # Origin position is now empty
        assert board[rc_to_pos(5, 0)] is None

    def test_revealed_pawn_moves_like_pawn_not_rook(self) -> None:
        """揭开后只能按兵走，不能继续按车走"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))

        # First move: reveals as PAWN
        board.apply_move(Move(rc_to_pos(5, 0), rc_to_pos(5, 1)))

        # Second move: should move like pawn, NOT rook
        moves = generate_piece_moves(board, 5, 1)
        targets = {m.to_pos for m in moves}
        # Pawn at row 5 (not crossed): can only go forward to row 4
        assert rc_to_pos(4, 1) in targets
        # Should NOT have rook-like horizontal moves
        assert rc_to_pos(5, 8) not in targets
        # Should NOT have rook-like vertical moves (beyond one step)
        assert rc_to_pos(8, 1) not in targets

    def test_hidden_cannon_moves_like_cannon(self) -> None:
        """暗炮位第一次按炮规则走"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        # Hidden cannon at cannon position; true_type = HORSE
        _place(board, 7, 1, _piece(Color.RED, PieceType.CANNON, PieceType.HORSE, False))
        # Screen piece and target for cannon capture
        _place(board, 4, 1, _piece(Color.BLACK, PieceType.PAWN, PieceType.PAWN, True))  # screen
        _place(board, 2, 1, _piece(Color.BLACK, PieceType.ROOK, PieceType.ROOK, True))  # target

        moves = generate_piece_moves(board, 7, 1)
        targets = {m.to_pos for m in moves}
        # Non-capture: can reach (6,1) and (5,1) but NOT (4,1)
        assert rc_to_pos(6, 1) in targets
        assert rc_to_pos(5, 1) in targets
        assert rc_to_pos(4, 1) not in targets  # screen — cannot move here non-capture
        # Capture: can capture at (2,1) via screen at (4,1)
        assert rc_to_pos(2, 1) in targets

    def test_hidden_horse_respects_leg_block(self) -> None:
        """暗马位第一次受马腿限制"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        # Hidden horse at horse position
        _place(board, 5, 4, _piece(Color.RED, PieceType.HORSE, PieceType.ADVISOR, False))
        # Blocker at up-leg
        _place(board, 4, 4, _piece(Color.RED, PieceType.PAWN, PieceType.PAWN, True))

        moves = generate_piece_moves(board, 5, 4)
        targets = {m.to_pos for m in moves}
        assert rc_to_pos(3, 3) not in targets  # blocked by leg at (4,4)
        assert rc_to_pos(3, 5) not in targets  # blocked by leg at (4,4)
        # Other targets still reachable
        assert rc_to_pos(7, 3) in targets
        assert rc_to_pos(7, 5) in targets


# ---------------------------------------------------------------------------
#  Capture scenarios
# ---------------------------------------------------------------------------


class TestCaptureScenarios:
    def test_revealed_captures_hidden(self) -> None:
        """明子可以吃暗子"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        # Red revealed rook
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.ROOK, True))
        # Black hidden piece
        _place(board, 5, 3, _piece(Color.BLACK, PieceType.PAWN, PieceType.CANNON, False))

        # Capture
        move = Move(rc_to_pos(5, 0), rc_to_pos(5, 3))
        board.apply_move(move)

        # Hidden piece is removed from board
        assert board[rc_to_pos(5, 3)] is not None  # rook now here
        assert len(board.captured) == 1
        assert board.captured[0].true_type == PieceType.CANNON  # captured was cannon

    def test_hidden_captures_revealed(self) -> None:
        """暗子可以吃明子"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        # Red hidden piece at rook position
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))
        # Black revealed piece
        _place(board, 5, 3, _piece(Color.BLACK, PieceType.PAWN, PieceType.PAWN, True))

        # Hidden captures revealed
        move = Move(rc_to_pos(5, 0), rc_to_pos(5, 3))
        board.apply_move(move)

        moved = board[rc_to_pos(5, 3)]
        assert moved is not None
        assert moved.revealed is True  # attacker is now revealed
        assert moved.true_type == PieceType.PAWN  # revealed as pawn
        assert board[rc_to_pos(5, 0)] is None
        assert len(board.captured) == 1

    def test_hidden_captures_hidden(self) -> None:
        """暗子吃暗子后，移动方揭开为 true_type，目标暗子被移除"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        # Red hidden rook (true_type=PAWN)
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))
        # Black hidden pawn (true_type=CANNON)
        _place(board, 5, 3, _piece(Color.BLACK, PieceType.PAWN, PieceType.CANNON, False))

        move = Move(rc_to_pos(5, 0), rc_to_pos(5, 3))
        board.apply_move(move)

        moved = board[rc_to_pos(5, 3)]
        assert moved is not None
        assert moved.revealed is True
        assert moved.true_type == PieceType.PAWN
        assert board[rc_to_pos(5, 0)] is None
        # Captured piece recorded
        assert len(board.captured) == 1
        assert board.captured[0].true_type == PieceType.CANNON

    def test_legal_moves_include_hidden_capture(self) -> None:
        """legal_moves includes capture of hidden pieces by revealed pieces."""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 3, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))  # diff col
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.ROOK, True))
        _place(board, 5, 3, _piece(Color.BLACK, PieceType.PAWN, PieceType.CANNON, False))

        legal = generate_legal_moves(board, Color.RED)
        capt_targets = {m.to_pos for m in legal}
        assert rc_to_pos(5, 3) in capt_targets


# ---------------------------------------------------------------------------
#  Board consistency after apply_move
# ---------------------------------------------------------------------------


class TestBoardConsistency:
    def test_piece_count_after_move(self) -> None:
        """apply_move 后棋子总数正确"""
        board = Board()
        board.reset(seed=42)
        initial_count = sum(1 for p in board.cells if p is not None)

        # Find a legal Red move
        legal = generate_legal_moves(board, Color.RED)
        assert len(legal) > 0, "Expected at least one legal move"
        move = legal[0]

        board.apply_move(move)
        after_count = sum(1 for p in board.cells if p is not None)
        # Non-capture move: count unchanged
        if len(board.captured) == 0:
            assert after_count == initial_count
        # Capture move: count decreases by 1
        else:
            assert after_count == initial_count - 1

    def test_no_overlap_after_apply_move(self) -> None:
        """apply_move 后棋盘没有格子同时包含两个棋子"""
        board = Board()
        board.reset(seed=42)
        legal = generate_legal_moves(board, Color.RED)
        move = legal[0]
        board.apply_move(move)
        # Every cell has at most one piece
        assert sum(1 for p in board.cells if p is not None) <= 32
        # from_pos is empty
        assert board[move.from_pos] is None

    def test_captured_and_turn_reset_on_reset(self) -> None:
        """reset 后 captured 和 turn 重置"""
        board = Board()
        board.reset(seed=42)
        assert board.turn == Color.RED
        assert board.captured == []

        legal = generate_legal_moves(board, Color.RED)
        board.apply_move(legal[0])
        assert board.turn == Color.BLACK

        board.reset(seed=42)
        assert board.turn == Color.RED
        assert board.captured == []

    def test_hidden_count_decreases_after_reveal(self) -> None:
        """暗子揭开后 hidden_pieces 数量递减"""
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))

        assert len(board.hidden_pieces()) == 1

        move = Move(rc_to_pos(5, 0), rc_to_pos(5, 1))
        board.apply_move(move)

        assert len(board.hidden_pieces()) == 0
        assert len(board.revealed_pieces()) == 3  # two kings + revealed attacker

    def test_revealed_pieces_increase_after_reveal(self) -> None:
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))
        _place(board, 3, 3, _piece(Color.BLACK, PieceType.HORSE, PieceType.ROOK, False))

        assert len(board.revealed_pieces()) == 2

        board.apply_move(Move(rc_to_pos(5, 0), rc_to_pos(5, 1)))

        assert len(board.revealed_pieces()) == 3  # +1 for the revealed piece
