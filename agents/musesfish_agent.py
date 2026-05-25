from __future__ import annotations

import random

import numpy as np

from agents.musesfish_eval import score_public_action, soft_policy_from_scores
from agents.musesfish_original import OriginalMusesfishSearch
from jieqi.env import JieqiEnv

# Based on miaosiSari/Jieqi (GPL v3).  The default action selector now wraps the
# vendored original Python PVS engine; the lightweight public evaluator remains
# as a fallback and for fast policy targets.  It must not inspect hidden Board
# true_type values.


class MusesfishAgent:
    """Rule-based Jieqi teacher using public-information heuristics.

    The agent scores every legal move with a static evaluator inspired by the
    Musesfish/Jieqi rule engine, then either chooses the best move or exposes a
    top-k soft policy for supervised pretraining.
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        top_k: int = 4,
        temperature: float = 30.0,
        stochastic: bool = False,
        use_original_search: bool = True,
        think_time: float = 1.0,
        search_min_depth: int = 4,
        search_max_depth: int = 5,
    ) -> None:
        self._rng = random.Random(seed)
        self.top_k = top_k
        self.temperature = temperature
        self.stochastic = stochastic
        self.use_original_search = use_original_search
        self._original = (
            OriginalMusesfishSearch(
                think_time=think_time, min_depth=search_min_depth, max_depth=search_max_depth,
            )
            if use_original_search else None
        )

    def select_action(self, env: JieqiEnv) -> int:
        if self._original is not None and not self.stochastic:
            action = self._original.select_action(env)
            if action in env.legal_actions():
                return int(action)
        policy, action = self.get_policy(env)
        if not self.stochastic:
            return action
        nonzero = np.flatnonzero(policy > 0)
        if len(nonzero) == 0:
            legal = env.legal_actions()
            return self._rng.choice(legal) if legal else 0
        weights = policy[nonzero].astype(float)
        pick = self._rng.choices(nonzero.tolist(), weights=weights.tolist(), k=1)[0]
        return int(pick)

    def score_action(self, env: JieqiEnv, action: int) -> float:
        return score_public_action(
            env.observation(),
            action,
            current_player=env.current_player(),
            legal_actions=env.legal_actions(),
        )

    def get_policy(self, env: JieqiEnv) -> tuple[np.ndarray, int]:
        actions = env.legal_actions()
        if not actions:
            return np.zeros(8100, dtype=np.float32), 0
        obs = env.observation()
        player = env.current_player()
        scores = {
            action: score_public_action(
                obs, action, current_player=player, legal_actions=actions,
            )
            for action in actions
        }
        policy, action = soft_policy_from_scores(
            scores, top_k=self.top_k, temperature=self.temperature,
        )
        return policy, action
