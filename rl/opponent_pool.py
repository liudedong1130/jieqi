from __future__ import annotations

import os
import random
from typing import Any

from agents.belief_mcts_agent import BeliefMCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.musesfish_agent import MusesfishAgent
from agents.musesfish_cpp_agent import MusesfishCppAgent
from agents.policy_agent import PolicyAgent
from agents.random_agent import RandomAgent


class OpponentPool:
    """Pool of opponent agents for diverse self-play training.

    Supports baseline agents (random, greedy, belief_mcts) and
    historical policy checkpoints.  Agents are sampled uniformly
    by default.
    """

    def __init__(
        self,
        pool_dir: str | None = None,
        include_musesfish: bool = False,
        opponents: list[str] | None = None,
    ) -> None:
        self._opponents: list[dict[str, Any]] = []
        self._include_musesfish = include_musesfish
        self._configured_opponents = opponents
        if opponents is None:
            self._add_baselines()
        else:
            self._add_named_opponents(opponents)

    # ---- baseline agents ---------------------------------------------------

    def _add_baselines(self) -> None:
        self._opponents.append({"type": "random", "name": "random"})
        self._opponents.append({"type": "greedy", "name": "greedy"})
        if self._include_musesfish:
            self._opponents.append({"type": "musesfish", "name": "musesfish"})

    def _add_named_opponents(self, opponents: list[str]) -> None:
        for name in opponents:
            if name not in {"random", "greedy", "belief_mcts", "musesfish", "musesfish_cpp"}:
                raise ValueError(f"Unknown built-in opponent: {name}")
            self._opponents.append({"type": name, "name": name})

    # ---- policy checkpoints ------------------------------------------------

    def add_policy(self, checkpoint_path: str, name: str | None = None) -> None:
        if name is None:
            name = os.path.basename(checkpoint_path).replace(".pt", "")
        self._opponents.append({
            "type": "policy",
            "checkpoint": checkpoint_path,
            "name": name,
        })

    # ---- sampling ----------------------------------------------------------

    def sample(self, rng: random.Random, strategy: str = "uniform") -> dict[str, Any]:
        """Return a random opponent config dict."""
        if not self._opponents:
            return {"type": "random", "name": "random"}
        idx = rng.randint(0, len(self._opponents) - 1)
        return dict(self._opponents[idx])

    def __len__(self) -> int:
        return len(self._opponents)

    # ---- agent factory -----------------------------------------------------

    @staticmethod
    def make_agent(config: dict[str, Any], seed: int) -> Any:
        t = config["type"]
        if t == "random":
            return RandomAgent(seed=seed)
        elif t == "greedy":
            return GreedyAgent(seed=seed)
        elif t == "belief_mcts":
            return BeliefMCTSAgent(num_samples=10, seed=seed)
        elif t == "musesfish":
            return MusesfishAgent(seed=seed)
        elif t == "musesfish_cpp":
            return MusesfishCppAgent(seed=seed)
        elif t == "policy":
            return PolicyAgent(config["checkpoint"], deterministic=False, seed=seed)
        else:
            raise ValueError(f"Unknown opponent type: {t}")

    # ---- persistence -------------------------------------------------------

    def state_dict(self) -> dict:
        return {"opponents": self._opponents}

    def load_state_dict(self, state: dict) -> None:
        self._opponents = state.get("opponents", [])
        if self._configured_opponents is None:
            self._add_baselines_if_missing()

    def _add_baselines_if_missing(self) -> None:
        has_random = any(o["type"] == "random" for o in self._opponents)
        has_greedy = any(o["type"] == "greedy" for o in self._opponents)
        has_musesfish = any(o["type"] == "musesfish" for o in self._opponents)
        if not has_random:
            self._opponents.insert(0, {"type": "random", "name": "random"})
        if not has_greedy:
            self._opponents.insert(1, {"type": "greedy", "name": "greedy"})
        if self._include_musesfish and not has_musesfish:
            self._opponents.append({"type": "musesfish", "name": "musesfish"})
