from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

from agents.belief_mcts_agent import BeliefMCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.policy_agent import PolicyAgent
from agents.random_agent import RandomAgent
from agents.rollout_agent import RolloutAgent
from rl.ismcts import ISMCTSAgent
from evaluation.elo import INITIAL_ELO, update_elo
from jieqi.env import JieqiEnv

_SIMPLE_AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "rollout": RolloutAgent,
    "belief_mcts": BeliefMCTSAgent,
    "ismcts": ISMCTSAgent,
}

# ---------------------------------------------------------------------------
#  Multiprocessing workers (top-level for pickling)
# ---------------------------------------------------------------------------

GameResult = tuple[str, str, int, bool, str, int]  # (a_name, b_name, g, a_is_red, winner, steps)


def _dataclass_to_dict(obj: Any) -> dict:
    """Convert a dataclass instance to a plain dict for pickling."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    return obj


def _mp_run_game(args: tuple) -> GameResult:
    """Run a single game in a worker process."""
    (a_cfg_dict, b_cfg_dict, g, half, n_games, max_steps, seed) = args
    a_cfg = AgentConfig(**a_cfg_dict)
    b_cfg = AgentConfig(**b_cfg_dict)

    a_is_red = g < half
    if a_is_red:
        red_cfg, black_cfg = a_cfg, b_cfg
    else:
        red_cfg, black_cfg = b_cfg, a_cfg

    env = JieqiEnv(max_steps=max_steps)
    env.reset(seed=seed + g)
    red = Arena._make_agent_static(red_cfg, seed=seed + g * 2)
    black = Arena._make_agent_static(black_cfg, seed=seed + g * 2 + 1)

    steps = 0
    done = False
    while not done:
        agent = red if env.current_player() == 0 else black
        action = agent.select_action(env)
        if action not in env.legal_actions():
            action = env.legal_actions()[0]
        _obs, reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if terminated:
            if reward > 0:
                winner = "red" if env.current_player() == 1 else "black"
            else:
                winner = "draw"
            return a_cfg.name, b_cfg.name, g, a_is_red, winner, steps
        elif truncated:
            return a_cfg.name, b_cfg.name, g, a_is_red, "draw", steps

    return a_cfg.name, b_cfg.name, g, a_is_red, "draw", steps


# ---------------------------------------------------------------------------
#  Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    name: str
    type: str  # random | greedy | rollout | belief_mcts | policy
    checkpoint: str | None = None
    deterministic: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    agent_a: str
    agent_b: str
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_as_red_wins: int = 0
    a_as_black_wins: int = 0
    b_as_red_wins: int = 0
    b_as_black_wins: int = 0
    total_games: int = 0
    avg_steps: float = 0.0
    illegal_actions: int = 0

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / max(self.total_games, 1)

    @property
    def b_win_rate(self) -> float:
        return self.b_wins / max(self.total_games, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.total_games, 1)


# ---------------------------------------------------------------------------
#  Arena
# ---------------------------------------------------------------------------


class Arena:
    """Round-robin tournament manager."""

    def __init__(self, agents: list[AgentConfig]) -> None:
        self.agents = agents
        self._ratings: dict[str, float] = {a.name: INITIAL_ELO for a in agents}
        self._results: list[MatchResult] = []

    # ---- agent factory ----------------------------------------------------

    @staticmethod
    def _make_agent(config: AgentConfig, seed: int) -> Any:
        return Arena._make_agent_static(config, seed)

    @staticmethod
    def _make_agent_static(config: AgentConfig, seed: int) -> Any:
        cls = _SIMPLE_AGENTS.get(config.type)
        if cls is not None:
            return cls(seed=seed, **config.params)
        if config.type == "policy":
            if config.checkpoint is None:
                raise ValueError(f"checkpoint required for policy agent '{config.name}'")
            return PolicyAgent(
                config.checkpoint, deterministic=config.deterministic, seed=seed
            )
        raise ValueError(f"Unknown agent type: {config.type}")

    # ---- match runner -----------------------------------------------------

    def run_match(
        self,
        a: AgentConfig,
        b: AgentConfig,
        n_games: int = 100,
        max_steps: int = 300,
        seed: int = 0,
    ) -> MatchResult:
        """Play *a* vs *b* for *n_games* with colour swap."""
        result = MatchResult(agent_a=a.name, agent_b=b.name, total_games=n_games)
        half = n_games // 2

        for g in range(n_games):
            if g < half:
                red_cfg, black_cfg = a, b
            else:
                red_cfg, black_cfg = b, a

            env = JieqiEnv(max_steps=max_steps)
            env.reset(seed=seed + g)
            red = self._make_agent(red_cfg, seed=seed + g * 2)
            black = self._make_agent(black_cfg, seed=seed + g * 2 + 1)

            steps = 0
            done = False
            while not done:
                agent = red if env.current_player() == 0 else black
                action = agent.select_action(env)
                if action not in env.legal_actions():
                    result.illegal_actions += 1
                    action = env.legal_actions()[0]
                _obs, reward, terminated, truncated, _info = env.step(action)
                steps += 1
                if terminated:
                    if reward > 0:
                        if env.current_player() == 1:  # Red moved → Red won
                            if g < half:
                                result.a_wins += 1
                                result.a_as_red_wins += 1
                            else:
                                result.b_wins += 1
                                result.b_as_red_wins += 1
                        else:
                            if g < half:
                                result.b_wins += 1
                                result.b_as_black_wins += 1
                            else:
                                result.a_wins += 1
                                result.a_as_black_wins += 1
                    else:
                        result.draws += 1
                    done = True
                elif truncated:
                    result.draws += 1
                    done = True
            result.avg_steps += steps

        result.avg_steps /= max(n_games, 1)
        return result

    # ---- round-robin ------------------------------------------------------

    def run_round_robin(
        self,
        n_games: int = 100,
        max_steps: int = 300,
        seed: int = 0,
    ) -> list[MatchResult]:
        """Play every pair of agents and update Elo ratings (single process)."""
        self._results = []
        n = len(self.agents)
        for i in range(n):
            for j in range(i + 1, n):
                mr = self.run_match(
                    self.agents[i], self.agents[j],
                    n_games=n_games, max_steps=max_steps, seed=seed,
                )
                self._results.append(mr)
                self._update_elo(mr)
                seed += 1
        return self._results

    def run_round_robin_mp(
        self,
        n_games: int = 100,
        max_steps: int = 300,
        seed: int = 0,
        workers: int = 4,
    ) -> list[MatchResult]:
        """Play every pair of agents using game-level multiprocessing.

        Each *game* is a separate task, load-balanced across *workers*
        processes.  Scales well even when matches have different speeds.
        """
        n = len(self.agents)
        half = n_games // 2

        # Build game-level tasks
        tasks: list[tuple] = []
        for i in range(n):
            for j in range(i + 1, n):
                a_cfg = _dataclass_to_dict(self.agents[i])
                b_cfg = _dataclass_to_dict(self.agents[j])
                for g in range(n_games):
                    tasks.append((a_cfg, b_cfg, g, half, n_games, max_steps, seed))
                seed += 1

        total_games = len(tasks)
        print(f"  Running {total_games} games ({len(self.agents)} agents × {n_games} games) "
              f"with {workers} workers ...")

        with mp.Pool(processes=workers) as pool:
            raw_results: list[GameResult] = pool.map(_mp_run_game, tasks)

        # Aggregate game results into match results
        self._results = []
        for i in range(n):
            for j in range(i + 1, n):
                a_name = self.agents[i].name
                b_name = self.agents[j].name
                mr = MatchResult(agent_a=a_name, agent_b=b_name, total_games=n_games)

                for gr in raw_results:
                    ga_name, gb_name, g, a_is_red, winner, steps = gr
                    if {ga_name, gb_name} != {a_name, b_name}:
                        continue
                    mr.avg_steps += steps
                    if winner == "red":
                        if a_is_red:
                            mr.a_wins += 1; mr.a_as_red_wins += 1
                        else:
                            mr.b_wins += 1; mr.b_as_red_wins += 1
                    elif winner == "black":
                        if a_is_red:
                            mr.b_wins += 1; mr.b_as_black_wins += 1
                        else:
                            mr.a_wins += 1; mr.a_as_black_wins += 1
                    else:
                        mr.draws += 1

                mr.avg_steps /= max(n_games, 1)
                self._results.append(mr)
                self._update_elo(mr)

        return self._results

    def _update_elo(self, mr: MatchResult) -> None:
        score_a = mr.a_win_rate + 0.5 * mr.draw_rate
        old_a = self._ratings[mr.agent_a]
        old_b = self._ratings[mr.agent_b]
        new_a, new_b = update_elo(old_a, old_b, score_a)
        self._ratings[mr.agent_a] = new_a
        self._ratings[mr.agent_b] = new_b

    # ---- ratings ----------------------------------------------------------

    def rating_table(self) -> list[dict[str, Any]]:
        """Elo ratings sorted highest-first."""
        items = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        return [{"name": name, "elo": round(elo, 1)} for name, elo in items]

    # ---- output -----------------------------------------------------------

    def summary_markdown(self) -> str:
        lines = [
            "| Rank | Agent | Elo | W | L | D | WR% |",
            "|------|-------|-----|---|---|---|-----|",
        ]
        for rank, item in enumerate(self.rating_table(), 1):
            name = item["name"]
            elo = item["elo"]
            w = sum(
                r.a_wins if r.agent_a == name else r.b_wins
                for r in self._results
                if r.agent_a == name or r.agent_b == name
            )
            l = sum(
                r.b_wins if r.agent_a == name else r.a_wins
                for r in self._results
                if r.agent_a == name or r.agent_b == name
            )
            d = sum(
                r.draws for r in self._results
                if r.agent_a == name or r.agent_b == name
            )
            total = w + l + d
            wr = w / max(total, 1) * 100
            lines.append(f"| {rank} | {name} | {elo:.0f} | {w} | {l} | {d} | {wr:.1f} |")

        # Per-match details
        lines.append("\n## Match Details\n")
        for mr in self._results:
            lines.append(
                f"- **{mr.agent_a}** vs **{mr.agent_b}**: "
                f"{mr.a_wins}-{mr.b_wins}-{mr.draws} "
                f"({mr.a_win_rate:.0%} / {mr.draw_rate:.0%} / {mr.b_win_rate:.0%}) "
                f"avg {mr.avg_steps:.0f} steps"
            )
        return "\n".join(lines)

    def summary_json(self) -> dict[str, Any]:
        return {
            "ratings": self.rating_table(),
            "matches": [
                {
                    "agent_a": mr.agent_a, "agent_b": mr.agent_b,
                    "a_wins": mr.a_wins, "b_wins": mr.b_wins, "draws": mr.draws,
                    "a_win_rate": round(mr.a_win_rate, 3),
                    "b_win_rate": round(mr.b_win_rate, 3),
                    "draw_rate": round(mr.draw_rate, 3),
                    "avg_steps": round(mr.avg_steps, 1),
                    "illegal_actions": mr.illegal_actions,
                }
                for mr in self._results
            ],
        }
