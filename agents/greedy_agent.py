from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jieqi.env import JieqiEnv

# Material values for revealed pieces
PIECE_VALUE: dict[int, int] = {
    0: 10000,  # KING
    1: 150,    # ADVISOR
    2: 150,    # ELEPHANT
    3: 300,    # HORSE
    4: 500,    # ROOK
    5: 350,    # CANNON
    6: 100,    # PAWN
}

HIDDEN_ESTIMATE = 250


class GreedyAgent:
    """Agent that prioritises captures using only public information.

    The agent inspects the observation tensor to detect captures and
    estimate their value.  It **never** accesses ``board._cells``
    directly, so true identities of hidden pieces cannot leak.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, env: JieqiEnv) -> int:
        actions = env.legal_actions()
        obs = env.observation()
        return self._choose(actions, obs)

    def _choose(self, actions: list[int], obs: np.ndarray) -> int:
        best_actions: list[int] = []
        best_value = -1

        for action in actions:
            to_pos = action % 90
            r, c = to_pos // 9, to_pos % 9
            value = self._target_value(obs, r, c)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return self._rng.choice(best_actions)

    @staticmethod
    def _target_value(obs: np.ndarray, r: int, c: int) -> int:
        """Estimate the value of the piece at board position (r, c).

        Returns 0 if the cell is empty.
        """
        # Check revealed channels first (0-13)
        for ch in range(14):
            if obs[ch, r, c] > 0.5:
                ptype = ch % 7
                return PIECE_VALUE.get(ptype, 0)

        # Check hidden channels (14-25)
        for ch in range(14, 26):
            if obs[ch, r, c] > 0.5:
                return HIDDEN_ESTIMATE

        return 0
