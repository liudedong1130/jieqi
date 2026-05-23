from __future__ import annotations

import random
from typing import Any

import numpy as np

from jieqi.env import JieqiEnv

# Material values for revealed pieces
PIECE_VALUES: dict[int, int] = {
    0: 10000,  # KING
    1: 150,    # ADVISOR
    2: 150,    # ELEPHANT
    3: 300,    # HORSE
    4: 500,    # ROOK
    5: 350,    # CANNON
    6: 100,    # PAWN
}

# Initial piece counts per side (same as HIDDEN_TRUE_TYPE_POOL + KING)
INITIAL_POOL: dict[int, int] = {
    0: 1,   # KING
    1: 2,   # ADVISOR
    2: 2,   # ELEPHANT
    3: 2,   # HORSE
    4: 2,   # ROOK
    5: 2,   # CANNON
    6: 5,   # PAWN
}


class BeliefMCTSAgent:
    """Root-level belief-sampling agent for Jieqi.

    For each legal action the agent samples *num_samples* possible hidden-
    piece assignments (from the public belief pool only — **never** peeking
    at ``board._cells``), simulates the resulting material balance, and
    picks the action with the best mean/p10 composite score.

    Parameters
    ----------
    num_samples : int
        Number of belief samples per candidate action (default 20).
    seed : int | None
        Random seed.
    """

    def __init__(self, num_samples: int = 20, seed: int | None = None) -> None:
        self._num_samples = num_samples
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def select_action(self, env: JieqiEnv) -> int:
        actions = env.legal_actions()
        if len(actions) == 1:
            return actions[0]

        obs = env.observation()
        player = env.current_player()

        pool_own, pool_opp = self._build_pools(obs)
        hidden = self._extract_hidden(obs)

        best_action = actions[0]
        best_score = -float("inf")

        for action in actions:
            scores: list[float] = []
            for _ in range(self._num_samples):
                assignment = self._sample(hidden, pool_own, pool_opp)
                score = self._simulate(obs, action, hidden, assignment)
                scores.append(score)

            arr = np.array(scores, dtype=np.float64)
            mean_s = float(np.mean(arr))
            p10_s = float(np.percentile(arr, 10))
            composite = 0.7 * mean_s + 0.3 * p10_s

            if composite > best_score:
                best_score = composite
                best_action = action

        return best_action

    # ------------------------------------------------------------------
    #  Belief pool
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pools(obs: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
        """Build remaining-piece pools from the observation tensor.

        Scans revealed channels (0–13).  Hidden-piece true identities
        are **never** used — only what the observation publicly shows.
        """
        pool_own = dict(INITIAL_POOL)
        pool_opp = dict(INITIAL_POOL)

        for ch in range(14):  # own revealed 0-6, opp revealed 7-13
            pts = _find_hot_positions(obs, ch)
            for _r, _c in pts:
                ptype = ch % 7
                if ch < 7:
                    pool_own[ptype] = max(0, pool_own[ptype] - 1)
                else:
                    pool_opp[ptype] = max(0, pool_opp[ptype] - 1)

        return pool_own, pool_opp

    @staticmethod
    def _extract_hidden(obs: np.ndarray) -> list[dict[str, Any]]:
        """Extract hidden-piece positions and origin_types from observation."""
        hidden: list[dict[str, Any]] = []
        for ch in range(14, 26):
            pts = _find_hot_positions(obs, ch)
            for r, c in pts:
                pos = r * 9 + c
                if ch < 20:
                    color = "own"
                    origin = (ch - 14) + 1  # ADVISOR=1 .. PAWN=6
                else:
                    color = "opp"
                    origin = (ch - 20) + 1
                hidden.append({"pos": pos, "color": color, "origin": origin})
        return hidden

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------

    def _sample(
        self,
        hidden: list[dict[str, Any]],
        pool_own: dict[int, int],
        pool_opp: dict[int, int],
    ) -> dict[int, int]:
        """Sample a *true_type* for every hidden piece without replacement."""
        own_flat = _pool_to_list(pool_own)
        opp_flat = _pool_to_list(pool_opp)
        self._rng.shuffle(own_flat)
        self._rng.shuffle(opp_flat)

        assignment: dict[int, int] = {}
        oi = 0
        bi = 0
        for h in hidden:
            if h["color"] == "own":
                assignment[h["pos"]] = own_flat[oi] if oi < len(own_flat) else h["origin"]
                oi += 1
            else:
                assignment[h["pos"]] = opp_flat[bi] if bi < len(opp_flat) else h["origin"]
                bi += 1
        return assignment

    # ------------------------------------------------------------------
    #  Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate(
        obs: np.ndarray,
        action: int,
        hidden: list[dict[str, Any]],
        assignment: dict[int, int],
    ) -> float:
        """Simulate *action* on a sampled board state, return material score."""
        from_pos = action // 90
        to_pos = action % 90

        # Build virtual board: pieces[0] = own pieces, pieces[1] = opp pieces
        own_mat = 0
        opp_mat = 0

        # Helper: add piece value to the right side
        def add_mat(color: str, ptype: int) -> None:
            nonlocal own_mat, opp_mat
            v = PIECE_VALUES.get(ptype, 0)
            if color == "own":
                own_mat += v
            else:
                opp_mat += v

        # Add revealed pieces from observation
        for ch in range(14):
            pts = _find_hot_positions(obs, ch)
            ptype = ch % 7
            color = "own" if ch < 7 else "opp"
            for r, c in pts:
                pos = r * 9 + c
                if pos == to_pos:
                    continue  # will be captured
                add_mat(color, ptype)

        # Add hidden pieces with sampled true_type
        for h in hidden:
            pos = h["pos"]
            if pos == from_pos:
                # Mover: revealed, moved to to_pos
                t = assignment.get(pos, h["origin"])
                add_mat(h["color"], t)
                continue
            if pos == to_pos:
                continue  # captured (hidden)
            t = assignment.get(pos, h["origin"])
            add_mat(h["color"], t)

        return float(own_mat - opp_mat)


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------


def _find_hot_positions(obs: np.ndarray, ch: int) -> list[tuple[int, int]]:
    """Return ``[(row, col), ...]`` where ``obs[ch, row, col] > 0.5``."""
    rows, cols = np.where(obs[ch] > 0.5)
    return list(zip(rows.tolist(), cols.tolist()))


def _pool_to_list(pool: dict[int, int]) -> list[int]:
    """Flatten a ``{type: count}`` pool into a list of individual types."""
    out: list[int] = []
    for ptype, cnt in pool.items():
        out.extend([ptype] * cnt)
    return out
