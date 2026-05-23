import pytest
from dataclasses import FrozenInstanceError

from jieqi.constants import BOARD_COLS, BOARD_ROWS, BOARD_SIZE, NUM_ACTIONS
from jieqi.move import (
    Move,
    decode_action,
    encode_action,
    is_valid_pos,
    is_valid_rc,
    pos_to_rc,
    rc_to_pos,
)


class TestPosToRC:
    def test_origin(self) -> None:
        assert pos_to_rc(0) == (0, 0)

    def test_top_right(self) -> None:
        assert pos_to_rc(8) == (0, 8)

    def test_bottom_left(self) -> None:
        assert pos_to_rc(81) == (9, 0)

    def test_bottom_right(self) -> None:
        assert pos_to_rc(89) == (9, 8)

    def test_second_row(self) -> None:
        assert pos_to_rc(9) == (1, 0)
        assert pos_to_rc(17) == (1, 8)

    def test_middle(self) -> None:
        assert pos_to_rc(40) == (4, 4)


class TestRCToPos:
    def test_origin(self) -> None:
        assert rc_to_pos(0, 0) == 0

    def test_top_right(self) -> None:
        assert rc_to_pos(0, 8) == 8

    def test_bottom_left(self) -> None:
        assert rc_to_pos(9, 0) == 81

    def test_bottom_right(self) -> None:
        assert rc_to_pos(9, 8) == 89

    def test_second_row(self) -> None:
        assert rc_to_pos(1, 0) == 9
        assert rc_to_pos(1, 8) == 17


class TestPosRCRoundtrip:
    def test_all_positions(self) -> None:
        for pos in range(BOARD_SIZE):
            row, col = pos_to_rc(pos)
            assert rc_to_pos(row, col) == pos

    def test_all_rows_cols(self) -> None:
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                pos = rc_to_pos(row, col)
                assert pos_to_rc(pos) == (row, col)


class TestEncodeAction:
    def test_from_0_to_0(self) -> None:
        assert encode_action(0, 0) == 0

    def test_from_0_to_89(self) -> None:
        assert encode_action(0, 89) == 89

    def test_from_89_to_0(self) -> None:
        assert encode_action(89, 0) == 89 * 90

    def test_from_0_to_1(self) -> None:
        assert encode_action(0, 1) == 1

    def test_from_1_to_0(self) -> None:
        assert encode_action(1, 0) == 90

    def test_last_action(self) -> None:
        assert encode_action(89, 89) == 8099


class TestDecodeAction:
    def test_action_0(self) -> None:
        assert decode_action(0) == (0, 0)

    def test_action_89(self) -> None:
        assert decode_action(89) == (0, 89)

    def test_action_90(self) -> None:
        assert decode_action(90) == (1, 0)

    def test_action_8010(self) -> None:
        assert decode_action(8010) == (89, 0)

    def test_action_8099(self) -> None:
        assert decode_action(8099) == (89, 89)


class TestEncodeDecodeRoundtrip:
    def test_all_actions(self) -> None:
        for from_pos in range(BOARD_SIZE):
            for to_pos in range(BOARD_SIZE):
                action = encode_action(from_pos, to_pos)
                decoded_from, decoded_to = decode_action(action)
                assert decoded_from == from_pos
                assert decoded_to == to_pos

    def test_action_range(self) -> None:
        for from_pos in range(BOARD_SIZE):
            for to_pos in range(BOARD_SIZE):
                action = encode_action(from_pos, to_pos)
                assert 0 <= action < NUM_ACTIONS


class TestMoveCreation:
    def test_valid_move(self) -> None:
        move = Move(0, 1)
        assert move.from_pos == 0
        assert move.to_pos == 1

    def test_corner_to_corner(self) -> None:
        move = Move(0, 89)
        assert move.from_pos == 0
        assert move.to_pos == 89

    def test_same_positions(self) -> None:
        move = Move(5, 5)
        assert move.from_pos == 5
        assert move.to_pos == 5

    def test_invalid_negative_from(self) -> None:
        with pytest.raises(ValueError, match="from_pos"):
            Move(-1, 0)

    def test_invalid_too_large_to(self) -> None:
        with pytest.raises(ValueError, match="to_pos"):
            Move(0, 90)

    def test_invalid_both(self) -> None:
        with pytest.raises(ValueError):
            Move(-5, 100)


class TestMoveImmutability:
    def test_frozen(self) -> None:
        move = Move(3, 7)
        with pytest.raises(FrozenInstanceError):
            move.from_pos = 0  # type: ignore[misc]

    def test_frozen_to(self) -> None:
        move = Move(3, 7)
        with pytest.raises(FrozenInstanceError):
            move.to_pos = 0  # type: ignore[misc]


class TestMoveRepr:
    def test_repr_format(self) -> None:
        move = Move(0, 9)
        r = repr(move)
        assert "Move" in r
        assert "0,0" in r
        assert "1,0" in r

    def test_repr_does_not_raise(self) -> None:
        for pos in range(BOARD_SIZE):
            move = Move(0, pos)
            assert isinstance(repr(move), str)


class TestMoveEquality:
    def test_equal(self) -> None:
        assert Move(0, 1) == Move(0, 1)

    def test_not_equal(self) -> None:
        assert Move(0, 1) != Move(0, 2)

    def test_hashable(self) -> None:
        s = {Move(0, 1), Move(0, 1), Move(1, 2)}
        assert len(s) == 2


class TestIsValidPos:
    def test_valid(self) -> None:
        assert is_valid_pos(0) is True
        assert is_valid_pos(89) is True
        assert is_valid_pos(45) is True

    def test_invalid(self) -> None:
        assert is_valid_pos(-1) is False
        assert is_valid_pos(90) is False
        assert is_valid_pos(100) is False


class TestIsValidRC:
    def test_valid(self) -> None:
        assert is_valid_rc(0, 0) is True
        assert is_valid_rc(9, 8) is True
        assert is_valid_rc(5, 4) is True

    def test_invalid_row(self) -> None:
        assert is_valid_rc(-1, 0) is False
        assert is_valid_rc(10, 0) is False

    def test_invalid_col(self) -> None:
        assert is_valid_rc(0, -1) is False
        assert is_valid_rc(0, 9) is False
