#!/usr/bin/env python3
"""Evaluate agent A vs agent B for N games."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.rollout_agent import RolloutAgent
from jieqi.env import JieqiEnv

AGENT_REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "rollout": RolloutAgent,
}


def _make_agent(name: str, seed: int) -> Any:
    cls = AGENT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent '{name}'. Options: {list(AGENT_REGISTRY)}")
    return cls(seed=seed)


def run_eval(
    red_name: str = "random",
    black_name: str = "random",
    n_games: int = 100,
    max_steps: int = 500,
    seed: int = 0,
) -> dict:
    red_wins = 0
    black_wins = 0
    draws = 0
    total_steps = 0
    errors = 0

    for g in range(n_games):
        env = JieqiEnv(max_steps=max_steps)
        env.reset(seed=(seed + g) if seed is not None else None)
        red = _make_agent(red_name, seed=(seed + g * 2) if seed is not None else None)
        black = _make_agent(black_name, seed=(seed + g * 2 + 1) if seed is not None else None)

        try:
            done = False
            steps = 0
            while not done:
                if env.current_player() == 0:
                    action = red.select_action(env)
                else:
                    action = black.select_action(env)
                _obs, reward, terminated, truncated, _info = env.step(action)
                steps += 1
                if terminated:
                    if reward > 0:
                        if env.current_player() == 1:
                            red_wins += 1
                        else:
                            black_wins += 1
                    else:
                        draws += 1
                    done = True
                elif truncated:
                    draws += 1
                    done = True
            total_steps += steps
        except Exception as exc:
            errors += 1
            print(f"[game {g}] error: {exc}")

    return {
        "red_wins": red_wins,
        "black_wins": black_wins,
        "draws": draws,
        "avg_steps": total_steps / max(n_games, 1),
        "errors": errors,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Agent vs Agent evaluation")
    p.add_argument("--red", default="random", help="Red agent name")
    p.add_argument("--black", default="random", help="Black agent name")
    p.add_argument("--games", type=int, default=100, help="Number of games")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per game")
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = p.parse_args()

    print(f"Red: {args.red}  vs  Black: {args.black}  ({args.games} games)")
    result = run_eval(
        red_name=args.red,
        black_name=args.black,
        n_games=args.games,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print(f"\nResults ({args.games} games):")
    print(f"  Red ({args.red}) wins:   {result['red_wins']}  ({result['red_wins'] / args.games * 100:.1f}%)")
    print(f"  Black ({args.black}) wins: {result['black_wins']}  ({result['black_wins'] / args.games * 100:.1f}%)")
    print(f"  Draws:               {result['draws']}  ({result['draws'] / args.games * 100:.1f}%)")
    print(f"  Avg steps:           {result['avg_steps']:.1f}")
    if result["errors"]:
        print(f"  Errors:              {result['errors']}")


if __name__ == "__main__":
    main()
