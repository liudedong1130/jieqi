#!/usr/bin/env python3
"""Round-robin agent tournament with Elo ratings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.arena import AgentConfig, Arena


def main() -> None:
    p = argparse.ArgumentParser(description="Arena agent tournament")
    p.add_argument("--agents", type=str, default="random,greedy",
                   help="Comma-separated agent list (or 'all' for all five)")
    p.add_argument("--games", type=int, default=50, help="Games per match")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Save JSON to file")
    args = p.parse_args()

    if args.agents == "all":
        names = ["random", "greedy", "rollout", "belief_mcts"]
    else:
        names = [n.strip() for n in args.agents.split(",")]

    configs = [
        AgentConfig(name=n, type=n) for n in names
    ]

    print(f"Agents: {names}")
    print(f"Games per match: {args.games}")
    print()

    arena = Arena(configs)
    results = arena.run_round_robin(n_games=args.games, max_steps=args.max_steps, seed=args.seed)

    print(arena.summary_markdown())

    if args.output:
        summary = arena.summary_json()
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON saved to {args.output}")


if __name__ == "__main__":
    main()
