from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jieqi.env import JieqiEnv


class RandomAgent:
    """Agent that selects a random legal action."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, env: JieqiEnv) -> int:
        actions = env.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")
        return self._rng.choice(actions)
