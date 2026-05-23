#!/usr/bin/env python3
"""Run RandomAgent vs RandomAgent self-play for N games."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from jieqi.env import JieqiEnv


def run_selfplay(n_games: int = 100, max_steps: int = 500, seed: int = 0) -> dict:
    """Run *n_games* of RandomAgent vs RandomAgent.

    Returns a dict with keys: red_wins, black_wins, draws, avg_steps, errors.
    """
    red_wins = 0
    black_wins = 0
    draws = 0
    total_steps = 0
    errors = 0

    for g in range(n_games):
        env = JieqiEnv(max_steps=max_steps)
        env.reset(seed=(seed + g) if seed is not None else None)
        agent_red = RandomAgent(seed=(seed + g * 2) if seed is not None else None)
        agent_black = RandomAgent(seed=(seed + g * 2 + 1) if seed is not None else None)

        try:
            done = False
            steps = 0
            while not done:
                if env.current_player() == 0:
                    action = agent_red.select_action(env)
                else:
                    action = agent_black.select_action(env)
                _obs, reward, terminated, truncated, _info = env.step(action)
                steps += 1
                if terminated:
                    if reward > 0:
                        if env.current_player() == 1:  # turn was swapped
                            red_wins += 1
                        else:
                            black_wins += 1
                    elif reward < 0:
                        if env.current_player() == 1:
                            black_wins += 1
                        else:
                            red_wins += 1
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
    p = argparse.ArgumentParser(description="RandomAgent self-play")
    p.add_argument("--games", type=int, default=100, help="Number of games")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per game")
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = p.parse_args()

    print(f"Running {args.games} RandomAgent vs RandomAgent games ...")
    result = run_selfplay(n_games=args.games, max_steps=args.max_steps, seed=args.seed)

    print(f"\nResults ({args.games} games):")
    print(f"  Red wins:   {result['red_wins']}  ({result['red_wins'] / args.games * 100:.1f}%)")
    print(f"  Black wins: {result['black_wins']}  ({result['black_wins'] / args.games * 100:.1f}%)")
    print(f"  Draws:      {result['draws']}  ({result['draws'] / args.games * 100:.1f}%)")
    print(f"  Avg steps:  {result['avg_steps']:.1f}")
    if result["errors"]:
        print(f"  Errors:     {result['errors']}")


if __name__ == "__main__":
    main()
