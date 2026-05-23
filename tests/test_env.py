from __future__ import annotations

import numpy as np
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

# ---------------------------------------------------------------------------
#  JieqiEnv interface tests
# ---------------------------------------------------------------------------


class TestJieqiEnvReset:
    def test_reset_returns_tensor(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        obs = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (28, 10, 9)

    def test_observation_does_not_leak_true_type(self) -> None:
        """Hidden pieces in observation show origin_type, not true_type."""
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        obs = env.observation()
        # Check: for each hidden piece, the revealed channel (0-13) is 0
        # and the hidden origin channel shows origin_type, not true_type
        for pos, p in enumerate(env.board.cells):
            if p is None or p.revealed:
                continue
            r, c = pos // 9, pos % 9
            # Hidden piece should NOT appear in any revealed channel (0-13)
            for ch in range(14):  # 0 through 13
                assert obs[ch, r, c] == 0.0, (
                    f"Hidden piece at ({r},{c}) leaked into revealed channel {ch}"
                )
            # Should appear in the correct hidden origin channel
            origin_ch = 14 + int(p.origin_type) - 1
            own_base = 14 if p.color == env.board.turn else 20
            actual_ch = own_base + int(p.origin_type) - 1
            assert obs[actual_ch, r, c] == 1.0


class TestJieqiEnvLegalActions:
    def test_legal_actions_returns_list(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        actions = env.legal_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(a, int) for a in actions)

    def test_legal_action_mask_shape(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        mask = env.legal_action_mask()
        assert mask.shape == (8100,)

    def test_legal_action_mask_consistency(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        actions = env.legal_actions()
        mask = env.legal_action_mask()
        assert mask.sum() == len(actions)
        for a in actions:
            assert mask[a] == 1


class TestJieqiEnvStep:
    def test_step_switches_current_player(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        player_before = env.current_player()
        actions = env.legal_actions()
        obs, reward, terminated, truncated, info = env.step(actions[0])
        assert env.current_player() != player_before

    def test_step_illegal_action_raises(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        # Find an illegal action (action 0 might be legal)
        legal = set(env.legal_actions())
        illegal = next(a for a in range(8100) if a not in legal)
        with pytest.raises(ValueError):
            env.step(illegal)

    def test_step_returns_expected_tuple(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        actions = env.legal_actions()
        obs, reward, terminated, truncated, info = env.step(actions[0])
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (28, 10, 9)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestJieqiEnvTermination:
    def test_king_capture_terminates(self) -> None:
        """When a king is captured, terminated=True with reward=1.0."""
        from jieqi.env import JieqiEnv

        env = JieqiEnv(max_steps=100)
        _setup_king_capture(env)
        # Red rook at (5,0) can capture Black king at (0,0)
        # The action: rook moves from (5,0) to (0,0)
        action = _encode(5, 0, 0, 0)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert reward == 1.0

    def test_max_steps_truncation(self) -> None:
        from jieqi.env import JieqiEnv

        env = JieqiEnv(max_steps=10)
        env.reset(seed=42)
        truncated_seen = False
        for _ in range(15):
            actions = env.legal_actions()
            if not actions:
                break
            obs, reward, terminated, truncated, info = env.step(actions[0])
            if truncated:
                truncated_seen = True
                break
            if terminated:
                break
        assert truncated_seen, "Should have been truncated after max_steps"


class TestObservationNoLeak:
    def test_hidden_true_type_not_in_observation(self) -> None:
        """Observation tensor must not encode true_type of hidden pieces."""
        from jieqi.env import JieqiEnv

        env = JieqiEnv()
        env.reset(seed=42)
        obs = env.observation()
        assert isinstance(obs, np.ndarray)
        # For each hidden piece, verify it only appears in its origin_type channel
        # and the channel index is derived from origin_type, never from true_type
        for pos, p in enumerate(env.board.cells):
            if p is None or p.revealed:
                continue
            r, c = pos // 9, pos % 9
            base = 14 if p.color == env.board.turn else 20
            expected_ch = base + int(p.origin_type) - 1
            # Must be 1.0 in the correct hidden origin channel
            assert obs[expected_ch, r, c] == 1.0
            # Must not appear in the hidden channel corresponding to true_type
            # (when true_type != origin_type)
            if p.true_type != p.origin_type:
                true_ch = base + int(p.true_type) - 1
                assert obs[true_ch, r, c] == 0.0


class TestRandomSelfPlay:
    def test_random_agent_plays_full_episode(self) -> None:
        """RandomAgent completes a full episode without errors."""
        from jieqi.env import JieqiEnv
        from agents.random_agent import RandomAgent

        env = JieqiEnv(max_steps=300)
        env.reset(seed=123)
        agent_red = RandomAgent(seed=1)
        agent_black = RandomAgent(seed=2)

        done = False
        move_count = 0
        while not done:
            if env.current_player() == 0:
                action = agent_red.act(env)
            else:
                action = agent_black.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            move_count += 1
            done = terminated or truncated

        assert move_count > 0
        assert done
        # Episode should not run forever
        assert move_count <= 300, f"Episode took {move_count} moves"


# ---------------------------------------------------------------------------
#  Helpers for env tests
# ---------------------------------------------------------------------------


def _encode(fr: int, fc: int, tr: int, tc: int) -> int:
    from jieqi.move import encode_action, rc_to_pos

    return encode_action(rc_to_pos(fr, fc), rc_to_pos(tr, tc))


def _setup_king_capture(env: "JieqiEnv") -> None:
    """Set up a position where Red can immediately capture Black's king."""
    b = env.board
    for pos in range(BOARD_SIZE):
        b.set_cell(pos, None)
    b._captured = []
    b._turn = Color.RED
    # Black king exposed
    b.set_cell(rc_to_pos(0, 0), _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
    # Red king safe
    b.set_cell(rc_to_pos(9, 4), _piece(Color.RED, PieceType.KING, PieceType.KING, True))
    # Red rook can capture Black king
    b.set_cell(rc_to_pos(5, 0), _piece(Color.RED, PieceType.ROOK, PieceType.ROOK, True))


class TestBoardConsistencyMore:
    def test_revealed_pieces_increase_after_reveal(self) -> None:
        board = _empty_board()
        _place(board, 9, 4, _piece(Color.RED, PieceType.KING, PieceType.KING, True))
        _place(board, 0, 4, _piece(Color.BLACK, PieceType.KING, PieceType.KING, True))
        _place(board, 5, 0, _piece(Color.RED, PieceType.ROOK, PieceType.PAWN, False))
        _place(board, 3, 3, _piece(Color.BLACK, PieceType.HORSE, PieceType.ROOK, False))

        assert len(board.revealed_pieces()) == 2

        board.apply_move(Move(rc_to_pos(5, 0), rc_to_pos(5, 1)))

        assert len(board.revealed_pieces()) == 3  # +1 for the revealed piece
