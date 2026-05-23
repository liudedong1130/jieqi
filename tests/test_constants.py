import pytest

from jieqi.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    BOARD_SIZE,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    TOTAL_PIECES_PER_SIDE,
    Color,
    PieceType,
)


class TestBoardConstants:
    def test_board_rows(self) -> None:
        assert BOARD_ROWS == 10

    def test_board_cols(self) -> None:
        assert BOARD_COLS == 9

    def test_board_size(self) -> None:
        assert BOARD_SIZE == 90

    def test_num_actions(self) -> None:
        assert NUM_ACTIONS == 90 * 90
        assert NUM_ACTIONS == 8100

    def test_num_piece_types(self) -> None:
        assert NUM_PIECE_TYPES == 7

    def test_total_pieces_per_side(self) -> None:
        assert TOTAL_PIECES_PER_SIDE == 16


class TestColorEnum:
    def test_red_value(self) -> None:
        assert Color.RED == 0

    def test_black_value(self) -> None:
        assert Color.BLACK == 1

    def test_opposite_red(self) -> None:
        assert Color.RED.opposite() == Color.BLACK

    def test_opposite_black(self) -> None:
        assert Color.BLACK.opposite() == Color.RED

    def test_opposite_is_involution(self) -> None:
        for c in Color:
            assert c.opposite().opposite() == c

    def test_opposite_returns_color_not_int(self) -> None:
        result = Color.RED.opposite()
        assert isinstance(result, Color)


class TestPieceTypeEnum:
    def test_values_in_order(self) -> None:
        assert PieceType.KING == 0
        assert PieceType.ADVISOR == 1
        assert PieceType.ELEPHANT == 2
        assert PieceType.HORSE == 3
        assert PieceType.ROOK == 4
        assert PieceType.CANNON == 5
        assert PieceType.PAWN == 6

    def test_enum_length(self) -> None:
        assert len(PieceType) == NUM_PIECE_TYPES


class TestPackageImports:
    def test_top_level_imports(self) -> None:
        import jieqi

        assert jieqi.BOARD_ROWS == 10
        assert jieqi.BOARD_COLS == 9
        assert jieqi.BOARD_SIZE == 90
        assert jieqi.NUM_ACTIONS == 8100
        assert jieqi.Color.RED == Color.RED
        assert jieqi.PieceType.KING == PieceType.KING

    def test_piece_import(self) -> None:
        from jieqi import Piece

        p = Piece(Color.RED, PieceType.KING, PieceType.KING, revealed=True)
        assert p.is_king

    def test_move_import(self) -> None:
        from jieqi import Move, encode_action, decode_action

        m = Move(0, 1)
        assert encode_action(0, 1) == 1
        assert decode_action(1) == (0, 1)
