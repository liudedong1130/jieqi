import pytest

from jieqi import (
    BOARD_SIZE,
    HIDDEN_TRUE_TYPE_POOL,
    STANDARD_LAYOUT,
    Color,
    PieceType,
)
from jieqi.board import Board
from jieqi.render import render


class TestBoardInit:
    def test_cells_length(self) -> None:
        board = Board()
        board.reset()
        assert len(board.cells) == BOARD_SIZE

    def test_total_pieces(self) -> None:
        board = Board()
        board.reset()
        non_none = sum(1 for p in board.cells if p is not None)
        assert non_none == 32

    def test_red_has_16_pieces(self) -> None:
        board = Board()
        board.reset()
        red = board.pieces_of(Color.RED)
        assert len(red) == 16

    def test_black_has_16_pieces(self) -> None:
        board = Board()
        board.reset()
        black = board.pieces_of(Color.BLACK)
        assert len(black) == 16

    def test_red_king_revealed(self) -> None:
        board = Board()
        board.reset()
        kings = [(p, piece) for p, piece in board.pieces_of(Color.RED) if piece.is_king]
        assert len(kings) == 1
        king_pos, king = kings[0]
        assert king.revealed is True
        assert king.origin_type == PieceType.KING
        assert king.true_type == PieceType.KING

    def test_black_king_revealed(self) -> None:
        board = Board()
        board.reset()
        kings = [(p, piece) for p, piece in board.pieces_of(Color.BLACK) if piece.is_king]
        assert len(kings) == 1
        _king_pos, king = kings[0]
        assert king.revealed is True
        assert king.origin_type == PieceType.KING
        assert king.true_type == PieceType.KING

    def test_king_positions_match_layout(self) -> None:
        board = Board()
        board.reset()
        assert board.king_pos(Color.RED) == 9 * 9 + 4  # (9, 4)
        assert board.king_pos(Color.BLACK) == 0 * 9 + 4  # (0, 4)

    def test_hidden_count_per_side(self) -> None:
        board = Board()
        board.reset()
        for color in (Color.RED, Color.BLACK):
            hidden = [(p, piece) for p, piece in board.pieces_of(color) if not piece.revealed]
            assert len(hidden) == 15

    def test_revealed_count(self) -> None:
        board = Board()
        board.reset()
        revealed = board.revealed_pieces()
        assert len(revealed) == 2  # only the two Kings

    def test_hidden_count_total(self) -> None:
        board = Board()
        board.reset()
        hidden = board.hidden_pieces()
        assert len(hidden) == 30

    def test_true_type_distribution_per_side(self) -> None:
        board = Board()
        board.reset(seed=123)
        for color in (Color.RED, Color.BLACK):
            pieces = [piece for _pos, piece in board.pieces_of(color)]
            true_types = [p.true_type for p in pieces]
            assert true_types.count(PieceType.KING) == 1
            assert true_types.count(PieceType.ROOK) == 2
            assert true_types.count(PieceType.HORSE) == 2
            assert true_types.count(PieceType.CANNON) == 2
            assert true_types.count(PieceType.ADVISOR) == 2
            assert true_types.count(PieceType.ELEPHANT) == 2
            assert true_types.count(PieceType.PAWN) == 5

    def test_origin_type_matches_layout(self) -> None:
        board = Board()
        board.reset()
        for pos, piece in enumerate(board.cells):
            if piece is None:
                continue
            r, c = pos // 9, pos % 9
            expected_origin = STANDARD_LAYOUT[(r, c)][1]
            assert piece.origin_type == expected_origin, (
                f"Position ({r},{c}) origin_type mismatch: "
                f"expected {expected_origin}, got {piece.origin_type}"
            )

    def test_no_overlapping_pieces(self) -> None:
        board = Board()
        board.reset()
        occupied = [pos for pos, p in enumerate(board.cells) if p is not None]
        assert len(occupied) == 32
        assert len(set(occupied)) == 32  # all unique

    def test_all_other_cells_empty(self) -> None:
        board = Board()
        board.reset()
        empty = sum(1 for p in board.cells if p is None)
        assert empty == 58  # 90 - 32

    def test_seed_reproducibility(self) -> None:
        b1 = Board()
        b1.reset(seed=42)
        b2 = Board()
        b2.reset(seed=42)
        for pos in range(BOARD_SIZE):
            p1, p2 = b1[pos], b2[pos]
            if p1 is None:
                assert p2 is None
            else:
                assert p2 is not None
                assert p1.true_type == p2.true_type
                assert p1.origin_type == p2.origin_type
                assert p1.color == p2.color
                assert p1.revealed == p2.revealed

    def test_different_seeds_produce_different_assignments(self) -> None:
        """With 15! possible shuffles, different seeds are overwhelmingly likely to differ."""
        b1 = Board()
        b1.reset(seed=1)
        b2 = Board()
        b2.reset(seed=2)
        # Compare true_type sequences for Red hidden pieces
        seq1 = tuple(
            p.true_type
            for p in b1.cells
            if p is not None and p.color == Color.RED and not p.revealed
        )
        seq2 = tuple(
            p.true_type
            for p in b2.cells
            if p is not None and p.color == Color.RED and not p.revealed
        )
        assert seq1 != seq2

    def test_hidden_pieces_not_revealed(self) -> None:
        board = Board()
        board.reset()
        for _pos, piece in board.hidden_pieces():
            assert piece.revealed is False
            assert piece.effective_type == piece.origin_type

    def test_get_piece(self) -> None:
        board = Board()
        board.reset()
        # Red King at position (9,4) = 85
        king = board.get_piece(85)
        assert king is not None
        assert king.is_king
        assert king.color == Color.RED

    def test_empty_cell(self) -> None:
        board = Board()
        board.reset()
        assert board.get_piece(0) is not None  # Black rook
        assert board.get_piece(10) is None  # empty

    def test_hidden_true_type_pool_matches_spec(self) -> None:
        """Verify the constant pool has the correct composition."""
        assert len(HIDDEN_TRUE_TYPE_POOL) == 15
        assert HIDDEN_TRUE_TYPE_POOL.count(PieceType.ROOK) == 2
        assert HIDDEN_TRUE_TYPE_POOL.count(PieceType.HORSE) == 2
        assert HIDDEN_TRUE_TYPE_POOL.count(PieceType.CANNON) == 2
        assert HIDDEN_TRUE_TYPE_POOL.count(PieceType.ADVISOR) == 2
        assert HIDDEN_TRUE_TYPE_POOL.count(PieceType.ELEPHANT) == 2
        assert HIDDEN_TRUE_TYPE_POOL.count(PieceType.PAWN) == 5
        assert PieceType.KING not in HIDDEN_TRUE_TYPE_POOL

    def test_standard_layout_has_32_entries(self) -> None:
        assert len(STANDARD_LAYOUT) == 32


class TestRender:
    def test_render_contains_red_king(self) -> None:
        board = Board()
        board.reset()
        output = render(board)
        assert "K " in output or "K|" in output or "K*" in output
        # Red King should be "K " (revealed, no *)
        has_red_king_revealed = False
        for line in output.split("\n"):
            if "K " in line or "K|" in line:
                has_red_king_revealed = True
        assert has_red_king_revealed

    def test_render_contains_black_king(self) -> None:
        board = Board()
        board.reset()
        output = render(board)
        has_black_king = any("k " in line or "k|" in line for line in output.split("\n"))
        assert has_black_king

    def test_render_contains_hidden_markers(self) -> None:
        board = Board()
        board.reset()
        output = render(board)
        assert "*" in output, "Hidden pieces should show * marker"

    def test_render_hidden_pieces_show_origin_type(self) -> None:
        board = Board()
        board.reset()
        output = render(board)
        # Hidden pieces at rook positions should show r* or R*
        # Black back rank rook at (0,0): hidden with rook origin → "r*"
        assert "r*" in output or "R*" in output

    def test_render_has_10_data_rows(self) -> None:
        board = Board()
        board.reset()
        output = render(board)
        lines = output.split("\n")
        # Data rows start with a single digit followed by " |" (e.g., "0 |r* |...")
        data_rows = [l for l in lines if len(l) >= 3 and l[0].isdigit() and l[1] == " " and l[2] == "|"]
        assert len(data_rows) == 10

    def test_render_does_not_crash(self) -> None:
        board = Board()
        board.reset()
        output = render(board)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_kings_no_star(self) -> None:
        """Kings should NOT have * marker since they're always revealed."""
        board = Board()
        board.reset()
        output = render(board)
        assert "K*" not in output
        assert "k*" not in output
