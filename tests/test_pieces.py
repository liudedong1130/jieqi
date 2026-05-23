import pytest
from dataclasses import FrozenInstanceError

from jieqi.constants import Color, PieceType
from jieqi.pieces import Piece


class TestPieceCreation:
    def test_create_hidden_piece(self) -> None:
        piece = Piece(
            color=Color.RED,
            origin_type=PieceType.ROOK,
            true_type=PieceType.CANNON,
            revealed=False,
        )
        assert piece.color == Color.RED
        assert piece.origin_type == PieceType.ROOK
        assert piece.true_type == PieceType.CANNON
        assert piece.revealed is False

    def test_create_revealed_piece(self) -> None:
        piece = Piece(
            color=Color.BLACK,
            origin_type=PieceType.KING,
            true_type=PieceType.KING,
            revealed=True,
        )
        assert piece.revealed is True

    def test_default_revealed_is_false(self) -> None:
        piece = Piece(
            color=Color.RED,
            origin_type=PieceType.HORSE,
            true_type=PieceType.ADVISOR,
        )
        assert piece.revealed is False


class TestEffectiveType:
    def test_hidden_piece_uses_origin_type(self) -> None:
        """Hidden piece effective_type must be origin_type, NOT true_type."""
        piece = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, revealed=False)
        assert piece.effective_type == PieceType.ROOK

    def test_revealed_piece_uses_true_type(self) -> None:
        piece = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, revealed=True)
        assert piece.effective_type == PieceType.CANNON

    def test_king_starts_revealed(self) -> None:
        piece = Piece(Color.RED, PieceType.KING, PieceType.KING, revealed=True)
        assert piece.effective_type == PieceType.KING

    def test_true_type_not_leaked_via_effective_when_hidden(self) -> None:
        """effective_type must NEVER equal true_type when origin_type differs."""
        piece = Piece(Color.BLACK, PieceType.ELEPHANT, PieceType.HORSE, revealed=False)
        assert piece.effective_type != piece.true_type

    def test_same_origin_and_true_type_hidden(self) -> None:
        """When origin_type == true_type, effective_type equals both (trivial case)."""
        piece = Piece(Color.RED, PieceType.PAWN, PieceType.PAWN, revealed=False)
        assert piece.effective_type == PieceType.PAWN
        assert piece.effective_type == piece.true_type


class TestPieceImmutability:
    def test_cannot_modify_revealed(self) -> None:
        piece = Piece(Color.RED, PieceType.ROOK, PieceType.ROOK)
        with pytest.raises(FrozenInstanceError):
            piece.revealed = True  # type: ignore[misc]

    def test_cannot_modify_color(self) -> None:
        piece = Piece(Color.RED, PieceType.ROOK, PieceType.ROOK)
        with pytest.raises(FrozenInstanceError):
            piece.color = Color.BLACK  # type: ignore[misc]

    def test_cannot_modify_true_type(self) -> None:
        piece = Piece(Color.RED, PieceType.ROOK, PieceType.ROOK)
        with pytest.raises(FrozenInstanceError):
            piece.true_type = PieceType.CANNON  # type: ignore[misc]

    def test_replace_creates_new_piece(self) -> None:
        """dataclasses.replace should work for creating revealed copies."""
        from dataclasses import replace

        piece = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, revealed=False)
        revealed_piece = replace(piece, revealed=True)
        assert revealed_piece is not piece
        assert revealed_piece.revealed is True
        assert revealed_piece.effective_type == PieceType.CANNON
        assert piece.revealed is False


class TestPieceEquality:
    def test_equal_same_fields(self) -> None:
        p1 = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False)
        p2 = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False)
        assert p1 == p2

    def test_not_equal_different_revealed(self) -> None:
        p1 = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False)
        p2 = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, True)
        assert p1 != p2

    def test_not_equal_different_true_type(self) -> None:
        p1 = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False)
        p2 = Piece(Color.RED, PieceType.ROOK, PieceType.HORSE, False)
        assert p1 != p2

    def test_hashable(self) -> None:
        """Frozen dataclass should be hashable for use in sets/dicts."""
        p1 = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False)
        p2 = Piece(Color.RED, PieceType.ROOK, PieceType.CANNON, False)
        s = {p1, p2}
        assert len(s) == 1


class TestIsKing:
    def test_king_is_king(self) -> None:
        piece = Piece(Color.RED, PieceType.KING, PieceType.KING, True)
        assert piece.is_king is True

    def test_non_king_is_not_king(self) -> None:
        for pt in PieceType:
            if pt == PieceType.KING:
                continue
            piece = Piece(Color.RED, pt, pt, True)
            assert piece.is_king is False

    def test_hidden_king_is_king(self) -> None:
        piece = Piece(Color.RED, PieceType.KING, PieceType.KING, False)
        assert piece.is_king is True

    def test_piece_with_king_true_type_is_king(self) -> None:
        """is_king checks true_type, not effective_type."""
        piece = Piece(Color.RED, PieceType.PAWN, PieceType.KING, False)
        assert piece.is_king is True
