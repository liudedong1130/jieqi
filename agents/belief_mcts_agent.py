from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jieqi import Color, PieceType
from jieqi.board import Board
from jieqi.env import JieqiEnv
from jieqi.move import pos_to_rc, rc_to_pos
from jieqi.rules import generate_piece_moves

PIECE_VALUES: dict[int, int] = {
    0: 10000, 1: 150, 2: 150, 3: 300, 4: 500, 5: 350, 6: 100,
}

INITIAL_POOL: dict[int, int] = {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 5}

# ---------------------------------------------------------------------------
#  BeliefState
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """Public-information-only view of the game state.

    Constructed exclusively from ``env.observation()`` — **never** peeks
    at hidden ``true_type``.
    """

    revealed: list[dict] = field(default_factory=list)  # [{pos, color, ptype}]
    hidden: list[dict] = field(default_factory=list)     # [{pos, color, origin}]
    pool_own: dict[int, int] = field(default_factory=dict)
    pool_opp: dict[int, int] = field(default_factory=dict)
    current_player: int = 0

    @classmethod
    def from_env(cls, env: JieqiEnv) -> BeliefState:
        """Extract belief from public observation."""
        obs = env.observation()
        player = env.current_player()
        revealed: list[dict] = []
        hidden: list[dict] = []
        pool_own = dict(INITIAL_POOL)
        pool_opp = dict(INITIAL_POOL)

        # Revealed channels 0-13
        for ch in range(14):
            pts = _hot_positions(obs, ch)
            ptype = ch % 7
            color = 0 if ch < 7 else 1  # 0=own(Red when player==0), 1=opp
            for r, c in pts:
                pos = r * 9 + c
                revealed.append({"pos": pos, "color": color, "ptype": ptype})
                if ch < 7:
                    pool_own[ptype] = max(0, pool_own.get(ptype, 0) - 1)
                else:
                    pool_opp[ptype] = max(0, pool_opp.get(ptype, 0) - 1)

        # Hidden channels 14-25
        for ch in range(14, 26):
            pts = _hot_positions(obs, ch)
            for r, c in pts:
                pos = r * 9 + c
                if ch < 20:
                    hidden.append({"pos": pos, "color": 0, "origin": (ch - 14) + 1})
                else:
                    hidden.append({"pos": pos, "color": 1, "origin": (ch - 20) + 1})

        return cls(
            revealed=revealed,
            hidden=hidden,
            pool_own=pool_own,
            pool_opp=pool_opp,
            current_player=player,
        )


def _hot_positions(obs: np.ndarray, ch: int) -> list[tuple[int, int]]:
    rows, cols = np.where(obs[ch] > 0.5)
    return list(zip(rows.tolist(), cols.tolist()))


# ---------------------------------------------------------------------------
#  Determinization
# ---------------------------------------------------------------------------


def sample_determinization(belief: BeliefState, rng: random.Random) -> dict[int, dict]:
    """Sample a complete board state by assigning true_types to hidden pieces.

    Returns dict[pos] → {color, ptype, revealed}.
    """
    board: dict[int, dict] = {}

    for r in belief.revealed:
        board[r["pos"]] = {"color": r["color"], "ptype": r["ptype"], "revealed": True}

    own_flat = _pool_to_list(belief.pool_own)
    opp_flat = _pool_to_list(belief.pool_opp)
    rng.shuffle(own_flat)
    rng.shuffle(opp_flat)

    oi, bi = 0, 0
    for h in belief.hidden:
        if h["color"] == 0:
            t = own_flat[oi] if oi < len(own_flat) else h["origin"]
            oi += 1
        else:
            t = opp_flat[bi] if bi < len(opp_flat) else h["origin"]
            bi += 1
        board[h["pos"]] = {"color": h["color"], "ptype": t, "revealed": False}

    return board


def _pool_to_list(pool: dict[int, int]) -> list[int]:
    out: list[int] = []
    for ptype, cnt in pool.items():
        out.extend([ptype] * cnt)
    return out


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

_MOBILITY_MAX = 100


def evaluate_determinized(
    board: dict[int, dict],
    current_player: int,
    board_obj: Board,
) -> float:
    """Evaluate board from *current_player*'s perspective.

    Returns score in [-inf, +inf]; higher = better for current_player.
    Components: material + 0.05 * mobility + king_safety.
    """
    b = board_obj
    _apply_to_board(board, b)

    # Material
    own_mat = 0.0
    opp_mat = 0.0
    for info in board.values():
        v = PIECE_VALUES.get(info["ptype"], 0)
        if info["color"] == current_player:
            own_mat += v
        else:
            opp_mat += v

    # Mobility (count pseudo-legal moves for current_player)
    own_moves = 0
    opp_moves_approx = 0
    for pos, info in board.items():
        r, c = pos_to_rc(pos)
        color = Color(info["color"])
        if info["revealed"] or info["color"] == current_player:
            # For mobility, we generate moves from the perspective of the
            # current player (both own and revealed pieces)
            pass
    # Simple mobility: count own legal moves
    for pos, info in board.items():
        if info["color"] != current_player:
            continue
        r, c = pos_to_rc(pos)
        b._turn = Color(current_player)
        moves = generate_piece_moves(b, r, c)
        own_moves += len(moves)
    mobility = min(own_moves / _MOBILITY_MAX, 1.0) * 50

    # King safety
    king_safety = 0.0
    try:
        opp_color = 1 - current_player
        opp_king_pos = b.king_pos(Color(opp_color))
        # Check if current_player's pieces attack opponent king
        for pos, info in board.items():
            if info["color"] != current_player:
                continue
            r, c = pos_to_rc(pos)
            for mv in generate_piece_moves(b, r, c):
                if mv.to_pos == opp_king_pos:
                    king_safety += 500  # opponent king under attack
                    break
    except ValueError:
        king_safety += 10000  # opponent king captured → huge bonus

    # Check if own king is safe
    try:
        own_king_pos = b.king_pos(Color(current_player))
        for pos, info in board.items():
            if info["color"] == current_player:
                continue
            r, c = pos_to_rc(pos)
            for mv in generate_piece_moves(b, r, c):
                if mv.to_pos == own_king_pos:
                    king_safety -= 2000  # own king in check → huge penalty
                    break
    except ValueError:
        king_safety -= 10000

    _clear_board(b)
    return (own_mat - opp_mat) + mobility + king_safety


def _apply_to_board(state: dict[int, dict], board: Board) -> None:
    from jieqi.pieces import Piece
    for pos, info in state.items():
        color = Color(info["color"])
        ptype = PieceType(info["ptype"])
        board.set_cell(pos, Piece(color, ptype, ptype, info.get("revealed", True)))


def _clear_board(board: Board) -> None:
    for pos in range(90):
        board.set_cell(pos, None)


# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------


class BeliefMCTSAgent:
    """Root-level belief-sampling agent with multi-component evaluation.

    Parameters
    ----------
    num_samples : int
        Number of determinization samples (default 30).
    mean_weight : float
        Weight for mean score across samples.
    risk_weight : float
        Weight for p10 (worst-case) score.
    std_weight : float
        Penalty weight for score variance.
    top_k : int
        Number of top actions to print for debugging (0 = off).
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        num_samples: int = 30,
        mean_weight: float = 0.6,
        risk_weight: float = 0.3,
        std_weight: float = 0.1,
        top_k: int = 0,
        seed: int | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.mean_weight = mean_weight
        self.risk_weight = risk_weight
        self.std_weight = std_weight
        self.top_k = top_k
        self._rng = random.Random(seed)
        self._board = Board()  # reusable temporary board

    # ------------------------------------------------------------------
    #  Action selection
    # ------------------------------------------------------------------

    def select_action(self, env: JieqiEnv) -> int:
        actions = env.legal_actions()
        if len(actions) == 1:
            return actions[0]

        belief = BeliefState.from_env(env)

        # Generate samples once — reused across all candidate actions
        samples = [
            sample_determinization(belief, self._rng)
            for _ in range(self.num_samples)
        ]

        # Cache: for each (sample_idx, action), evaluate the resulting board
        action_scores: dict[int, list[float]] = {}
        player = env.current_player()

        for action in actions:
            scores: list[float] = []
            from_pos = action // 90
            to_pos = action % 90
            for bd in samples:
                sim = _simulate_action(bd, from_pos, to_pos)
                score = evaluate_determinized(sim, player, self._board)
                scores.append(score)
            action_scores[action] = scores

        # Compute composite scores
        scored: list[tuple[int, float, float, float, float]] = []
        for action, scores in action_scores.items():
            arr = np.array(scores, dtype=np.float64)
            mean_s = float(np.mean(arr))
            p10_s = float(np.percentile(arr, 10))
            std_s = float(np.std(arr))
            composite = (
                self.mean_weight * mean_s
                + self.risk_weight * p10_s
                - self.std_weight * std_s
            )
            scored.append((action, composite, mean_s, p10_s, std_s))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Debug output
        if self.top_k > 0:
            print(f"\n--- BeliefMCTS top {self.top_k} ---")
            for rank, (action, comp, mean_s, p10_s, std_s) in enumerate(scored[:self.top_k]):
                fpos, tpos = action // 90, action % 90
                print(
                    f"  {rank+1}. action={action} ({fpos//9},{fpos%9}->{tpos//9},{tpos%9}) "
                    f"comp={comp:.1f} mean={mean_s:.1f} p10={p10_s:.1f} std={std_s:.1f}"
                )

        return scored[0][0]


def _simulate_action(
    board: dict[int, dict], from_pos: int, to_pos: int
) -> dict[int, dict]:
    """Return a new board dict after executing the action."""
    sim = {pos: dict(info) for pos, info in board.items()}
    mover = sim.pop(from_pos, None)
    if mover is None:
        return sim
    mover = dict(mover)
    mover["revealed"] = True
    sim.pop(to_pos, None)  # captured
    sim[to_pos] = mover
    return sim
