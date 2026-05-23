from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jieqi.env import JieqiEnv

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


class RolloutAgent:
    """1-ply material-evaluation agent with random tie-breaking.

    For each legal action the agent estimates the immediate material gain
    (value of a captured piece) using only public information from the
    observation tensor.  Hidden pieces are valued at a fixed estimate.
    Small random noise is added to break ties.
    """

    def __init__(self, seed: int | None = None, noise_scale: float = 10.0) -> None:
        self._rng = random.Random(seed)
        self._noise_scale = noise_scale

    def select_action(self, env: JieqiEnv) -> int:
        actions = env.legal_actions()
        obs = env.observation()

        best_action = actions[0]
        best_score = -float("inf")

        for action in actions:
            to_pos = action % 90
            r, c = to_pos // 9, to_pos % 9
            gain = self._material_gain(obs, r, c)
            noise = self._rng.uniform(0, self._noise_scale)
            score = gain + noise
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @staticmethod
    def _material_gain(obs: np.ndarray, r: int, c: int) -> float:
        # Check revealed (channels 0-13)
        for ch in range(14):
            if obs[ch, r, c] > 0.5:
                return float(PIECE_VALUE.get(ch % 7, 0))

        # Check hidden (channels 14-25)
        for ch in range(14, 26):
            if obs[ch, r, c] > 0.5:
                return float(HIDDEN_ESTIMATE)

        return 0.0
