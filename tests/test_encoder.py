from __future__ import annotations

import numpy as np
import pytest

from jieqi import BOARD_SIZE, Color, Piece, PieceType
from jieqi.board import Board
from jieqi.encoder import NUM_CHANNELS, encode_observation
from jieqi.move import rc_to_pos
from jieqi.rules import generate_legal_moves


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _piece(color: Color, origin: PieceType, true: PieceType, revealed: bool = False) -> Piece:
    return Piece(color=color, origin_type=origin, true_type=true, revealed=revealed)


def _empty_board() -> Board:
    b = Board()
    for pos in range(BOARD_SIZE):
        b.set_cell(pos, None)
    b._captured = []
    b._turn = Color.RED
    return b


def _place(board: Board, row: int, col: int, piece: Piece) -> None:
    board.set_cell(rc_to_pos(row, col), piece)


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


class TestShape:
    def test_observation_shape(self) -> None:
        board = Board()
        board.reset(seed=42)
        tensor = encode_observation(board)
        assert tensor.shape == (NUM_CHANNELS, 10, 9)
        assert tensor.dtype == np.float32

    def test_side_to_move_channel_all_ones(self) -> None:
        board = Board()
        board.reset(seed=42)
        tensor = encode_observation(board)
        assert np.all(tensor[26, :, :] == 1.0)

    def test_reserved_channel_all_zeros(self) -> None:
        board = Board()
        board.reset(seed=42)
        tensor = encode_observation(board)
        assert np.all(tensor[27, :, :] == 0.0)


class TestTrueTypeIndistinguishability:
    def test_different_true_type_same_encoding(self) -> None:
        """Two hidden pieces with same origin_type and position but different
        true_type must produce identical encodings."""
        b1 = _empty_board()
        _place(b1, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(b1, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(b1, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False))

        b2 = _empty_board()
        _place(b2, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(b2, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(b2, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))

        t1 = encode_observation(b1)
        t2 = encode_observation(b2)
        assert np.array_equal(t1, t2), (
            "Encodings must be identical when only true_type differs"
        )

    def test_all_hidden_channels_identical_under_swap(self) -> None:
        """Even when true_types are globally shuffled, hidden channels
        only depend on origin_type and must be invariant."""
        board = Board()
        board.reset(seed=42)

        # Record all hidden pieces' origin_types and positions
        hidden_info = [
            (pos, p.origin_type, p.color)
            for pos, p in enumerate(board.cells)
            if p is not None and not p.revealed
        ]

        # Reset with different seed → different true_type assignment
        board2 = Board()
        board2.reset(seed=99)

        t1 = encode_observation(board)
        t2 = encode_observation(board2)

        # Hidden channels (14-25) should be identical because origin_types
        # at each position are fixed by STANDARD_LAYOUT
        assert np.array_equal(t1[14:26], t2[14:26])


class TestRevealTransition:
    def test_after_reveal_moves_to_revealed_channel(self) -> None:
        """After a hidden piece moves (reveals), its encoding moves from
        hidden origin channel to revealed true_type channel."""
        b = _empty_board()
        _place(b, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(b, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        # Hidden rook at (5,4); true_type = PAWN, revealed = False
        hidden_pos = rc_to_pos(5, 4)
        _place(b, 5, 4, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))
        b._turn = Color.RED

        before = encode_observation(b)
        # Before: it should be in hidden channel (channel 17 = ROOK origin)
        ch_hidden_rook = 14 + (int(PieceType.ROOK) - 1)  # = 17
        assert before[ch_hidden_rook, 5, 4] == 1.0
        # Not in revealed PAWN channel (channel 6)
        assert before[6, 5, 4] == 0.0

        # Move to reveal (apply_move swaps turn to BLACK)
        from jieqi.move import Move
        b.apply_move(Move(hidden_pos, rc_to_pos(5, 3)))

        after = encode_observation(b)
        # After: no longer in hidden origin channel
        assert after[ch_hidden_rook, 5, 3] == 0.0
        # Now in opponent's revealed PAWN channel (ch 7+6=13) because turn=BLACK
        assert after[13, 5, 3] == 1.0


class TestPlayerPerspective:
    def test_own_opponent_channels_swap_with_player(self) -> None:
        """When turn changes, own and opponent channels should swap."""
        b = _empty_board()
        _place(b, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(b, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))

        # Red's perspective
        b._turn = Color.RED
        t_red = encode_observation(b)
        # Red king should be in "own revealed KING" channel 0
        assert t_red[0, 9, 4] == 1.0
        # Black king should be in "opp revealed KING" channel 7
        assert t_red[7, 0, 4] == 1.0

        # Black's perspective
        b._turn = Color.BLACK
        t_black = encode_observation(b)
        # Black king should be in "own revealed KING" channel 0
        assert t_black[0, 0, 4] == 1.0
        # Red king should be in "opp revealed KING" channel 7
        assert t_black[7, 9, 4] == 1.0

    def test_hidden_perspective_swaps(self) -> None:
        """Hidden pieces also swap between own/opp hidden channels."""
        b = _empty_board()
        _place(b, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(b, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(b, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.ROOK, False))
        _place(b, 3, 0, _piece(Color.BLACK, PieceType.HORSE, PieceType.HORSE, False))

        b._turn = Color.RED
        t_red = encode_observation(b)
        ch_own_hidden_rook = 14 + (int(PieceType.ROOK) - 1)  # 17
        ch_opp_hidden_horse = 20 + (int(PieceType.HORSE) - 1)  # 22
        assert t_red[ch_own_hidden_rook, 5, 0] == 1.0
        assert t_red[ch_opp_hidden_horse, 3, 0] == 1.0

        b._turn = Color.BLACK
        t_black = encode_observation(b)
        # Now Red rook is opponent's hidden
        assert t_black[ch_opp_hidden_horse, 5, 0] == 0.0  # Red rook is not horse
        assert t_black[20 + (int(PieceType.ROOK) - 1), 5, 0] == 1.0  # opp hidden rook


class TestKingInCorrectChannel:
    def test_kings_always_in_revealed_channels(self) -> None:
        """Kings are always revealed; never appear in hidden channels."""
        board = Board()
        board.reset(seed=42)
        tensor = encode_observation(board)
        # All hidden channels (14-25) should be zero at king positions
        red_king_pos = board.king_pos(Color.RED)
        black_king_pos = board.king_pos(Color.BLACK)
        rkr, rkc = red_king_pos // 9, red_king_pos % 9
        bkr, bkc = black_king_pos // 9, black_king_pos % 9
        for ch in range(14, 26):
            assert tensor[ch, rkr, rkc] == 0.0
            assert tensor[ch, bkr, bkc] == 0.0
