from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from jieqi.constants import BOARD_COLS, BOARD_ROWS, NUM_ACTIONS, PieceType
from jieqi.move import decode_action, pos_to_rc

# Inspired by miaosiSari/Jieqi (GPL v3), rewritten for this project's public
# observation tensor.  This module deliberately avoids Board internals and
# hidden true_type values.

PIECE_VALUES: dict[int, float] = {
    int(PieceType.KING): 10000.0,
    int(PieceType.ADVISOR): 150.0,
    int(PieceType.ELEPHANT): 150.0,
    int(PieceType.HORSE): 320.0,
    int(PieceType.ROOK): 600.0,
    int(PieceType.CANNON): 380.0,
    int(PieceType.PAWN): 120.0,
}

HIDDEN_POOL: dict[int, int] = {
    int(PieceType.ADVISOR): 2,
    int(PieceType.ELEPHANT): 2,
    int(PieceType.HORSE): 2,
    int(PieceType.ROOK): 2,
    int(PieceType.CANNON): 2,
    int(PieceType.PAWN): 5,
}


@dataclass(frozen=True)
class PublicPiece:
    side: int  # 0 = current player, 1 = opponent
    ptype: int
    revealed: bool


def public_board_from_obs(obs: np.ndarray) -> list[PublicPiece | None]:
    board: list[PublicPiece | None] = [None] * (BOARD_ROWS * BOARD_COLS)

    for ch in range(14):
        side = 0 if ch < 7 else 1
        ptype = ch % 7
        for r, c in zip(*np.where(obs[ch] > 0.5)):
            board[int(r) * BOARD_COLS + int(c)] = PublicPiece(side, ptype, True)

    for ch in range(14, 26):
        side = 0 if ch < 20 else 1
        ptype = (ch - (14 if side == 0 else 20)) + 1
        for r, c in zip(*np.where(obs[ch] > 0.5)):
            board[int(r) * BOARD_COLS + int(c)] = PublicPiece(side, ptype, False)

    return board


def remaining_pools(board: list[PublicPiece | None]) -> dict[int, dict[int, int]]:
    pools = {0: dict(HIDDEN_POOL), 1: dict(HIDDEN_POOL)}
    for piece in board:
        if piece is None or not piece.revealed or piece.ptype == int(PieceType.KING):
            continue
        pool = pools[piece.side]
        pool[piece.ptype] = max(0, pool.get(piece.ptype, 0) - 1)
    return pools


def expected_hidden_value(pool: dict[int, int]) -> float:
    total = sum(pool.values())
    if total <= 0:
        return 0.0
    return sum(PIECE_VALUES[p] * n for p, n in pool.items()) / total


def score_public_action(
    obs: np.ndarray,
    action: int,
    *,
    current_player: int,
    legal_actions: list[int],
) -> float:
    board = public_board_from_obs(obs)
    from_pos, to_pos = decode_action(action)
    moving = board[from_pos]
    if moving is None or moving.side != 0:
        return -1e9

    target = board[to_pos]
    pools = remaining_pools(board)
    own_hidden_avg = expected_hidden_value(pools[0])
    opp_hidden_avg = expected_hidden_value(pools[1])

    score = 0.0
    score += _capture_score(target, opp_hidden_avg)
    score += _piece_square_delta(moving, from_pos, to_pos, current_player)
    score += _reveal_score(moving, own_hidden_avg, pools[0], from_pos, to_pos, current_player)
    score += _cannon_pressure_score(board, moving, from_pos, to_pos, current_player)
    score += _rook_rib_score(moving, from_pos, to_pos, board)
    score += _advisor_elephant_score(moving, from_pos, to_pos, board, current_player)
    score += _mobility_tiebreak(action, legal_actions)
    return float(score)


def soft_policy_from_scores(
    scores: dict[int, float],
    *,
    top_k: int = 12,
    temperature: float = 60.0,
) -> tuple[np.ndarray, int]:
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if not scores:
        return policy, 0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    chosen = ranked[0][0]
    kept = ranked[: max(1, top_k)]
    max_s = kept[0][1]
    weights = [(a, exp((s - max_s) / max(temperature, 1e-6))) for a, s in kept]
    total = sum(w for _, w in weights)
    if total <= 0:
        policy[chosen] = 1.0
    else:
        for action, weight in weights:
            policy[action] = float(weight / total)
    return policy, chosen


def _capture_score(target: PublicPiece | None, opp_hidden_avg: float) -> float:
    if target is None:
        return 0.0
    base = PIECE_VALUES[target.ptype] if target.revealed else opp_hidden_avg
    if not target.revealed and target.ptype == int(PieceType.ROOK):
        base += 70.0
    if target.ptype == int(PieceType.KING):
        base += 5000.0
    return base


def _piece_square_delta(piece: PublicPiece, from_pos: int, to_pos: int, current_player: int) -> float:
    fr, fc = pos_to_rc(from_pos)
    tr, tc = pos_to_rc(to_pos)
    forward = -1 if current_player == 0 else 1
    progress = (tr - fr) * forward
    center_before = abs(fc - 4)
    center_after = abs(tc - 4)

    score = (center_before - center_after) * 4.0
    if piece.ptype == int(PieceType.PAWN):
        score += progress * 12.0
        if (current_player == 0 and tr <= 4) or (current_player == 1 and tr >= 5):
            score += 18.0
    elif piece.ptype == int(PieceType.ROOK):
        score += progress * 3.0
    elif piece.ptype == int(PieceType.CANNON):
        score += progress * 5.0
    elif piece.ptype == int(PieceType.HORSE):
        score += (center_before - center_after) * 5.0
    return score


def _reveal_score(
    piece: PublicPiece,
    own_hidden_avg: float,
    pool: dict[int, int],
    from_pos: int,
    to_pos: int,
    current_player: int,
) -> float:
    if piece.revealed:
        return 0.0
    _, fc = pos_to_rc(from_pos)
    tr, tc = pos_to_rc(to_pos)
    expected = own_hidden_avg - PIECE_VALUES.get(piece.ptype, own_hidden_avg) * 0.25
    score = expected * 0.08

    if piece.ptype == int(PieceType.CANNON):
        forward_distance = (from_pos // BOARD_COLS - tr) if current_player == 0 else (tr - from_pos // BOARD_COLS)
        if forward_distance >= 4:
            score += 45.0
        if forward_distance >= 6:
            score += 35.0
    elif piece.ptype == int(PieceType.ADVISOR):
        pawn_pressure = pool.get(int(PieceType.PAWN), 0) / max(sum(pool.values()), 1)
        score -= 25.0 * pawn_pressure
        if tc in (3, 5):
            score += 12.0
    elif piece.ptype == int(PieceType.ELEPHANT):
        if tc in (2, 4, 6):
            score += 16.0
    elif piece.ptype == int(PieceType.HORSE):
        if abs(tc - 4) < abs(fc - 4):
            score += 18.0
    return score


def _cannon_pressure_score(
    board: list[PublicPiece | None],
    piece: PublicPiece,
    from_pos: int,
    to_pos: int,
    current_player: int,
) -> float:
    if piece.ptype != int(PieceType.CANNON):
        return 0.0
    simulated = board.copy()
    simulated[to_pos] = PublicPiece(0, piece.ptype, True)
    simulated[from_pos] = None
    tr, tc = pos_to_rc(to_pos)
    opp_back = 0 if current_player == 0 else BOARD_ROWS - 1
    score = 0.0
    if tr == opp_back:
        hidden_back = sum(
            1 for c in range(BOARD_COLS)
            if (p := simulated[opp_back * BOARD_COLS + c]) is not None and p.side == 1 and not p.revealed
        )
        score += 30.0 + hidden_back * 8.0

    king_pos = _find_piece(simulated, 1, int(PieceType.KING))
    if king_pos is not None and king_pos % BOARD_COLS == tc:
        between = _count_between(simulated, to_pos, king_pos)
        if between == 0:
            score += 90.0
        elif between == 1:
            score += 45.0
    return score


def _rook_rib_score(
    piece: PublicPiece,
    from_pos: int,
    to_pos: int,
    board: list[PublicPiece | None],
) -> float:
    if piece.ptype != int(PieceType.ROOK):
        return 0.0
    _, fc = pos_to_rc(from_pos)
    _, tc = pos_to_rc(to_pos)
    score = 0.0
    if tc in (3, 5) and fc not in (3, 5):
        score += 28.0
    if fc in (3, 5) and tc not in (3, 5):
        score -= 18.0
    target = board[to_pos]
    if target is not None and not target.revealed and target.ptype == int(PieceType.ROOK):
        score += 45.0
    return score


def _advisor_elephant_score(
    piece: PublicPiece,
    from_pos: int,
    to_pos: int,
    board: list[PublicPiece | None],
    current_player: int,
) -> float:
    if piece.ptype not in (int(PieceType.ADVISOR), int(PieceType.ELEPHANT)):
        return 0.0
    tr, tc = pos_to_rc(to_pos)
    own_back = BOARD_ROWS - 1 if current_player == 0 else 0
    palace_cols = {3, 4, 5}
    score = 0.0
    if tr == own_back and tc in palace_cols:
        score += 12.0
    enemy_rooks_on_rib = 0
    for pos, other in enumerate(board):
        if other is None or other.side != 1 or other.ptype != int(PieceType.ROOK):
            continue
        _, oc = pos_to_rc(pos)
        if oc in (3, 5):
            enemy_rooks_on_rib += 1
    if enemy_rooks_on_rib and tc in (3, 5):
        score += 20.0
    return score


def _mobility_tiebreak(action: int, legal_actions: list[int]) -> float:
    from_pos, to_pos = decode_action(action)
    same_piece_moves = sum(1 for a in legal_actions if a // 90 == from_pos)
    distance = abs((from_pos // 9) - (to_pos // 9)) + abs((from_pos % 9) - (to_pos % 9))
    return min(same_piece_moves, 8) * 0.5 + min(distance, 9) * 0.2


def _find_piece(board: list[PublicPiece | None], side: int, ptype: int) -> int | None:
    for pos, piece in enumerate(board):
        if piece is not None and piece.side == side and piece.ptype == ptype:
            return pos
    return None


def _count_between(board: list[PublicPiece | None], a: int, b: int) -> int:
    ar, ac = pos_to_rc(a)
    br, bc = pos_to_rc(b)
    if ac == bc:
        step = 1 if br > ar else -1
        return sum(1 for r in range(ar + step, br, step) if board[r * BOARD_COLS + ac] is not None)
    if ar == br:
        step = 1 if bc > ac else -1
        return sum(1 for c in range(ac + step, bc, step) if board[ar * BOARD_COLS + c] is not None)
    return 0
