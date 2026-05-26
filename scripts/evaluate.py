#!/usr/bin/env python3
"""Evaluate two agents against each other with fair colour swap."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.belief_mcts_agent import BeliefMCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.musesfish_agent import MusesfishAgent
from agents.musesfish_cpp_agent import MusesfishCppAgent
from agents.policy_agent import PolicyAgent
from agents.random_agent import RandomAgent
from agents.rollout_agent import RolloutAgent
from rl.ismcts import ISMCTSAgent
from jieqi.env import JieqiEnv

# Agents that don't need a checkpoint
_SIMPLE_REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "rollout": RolloutAgent,
    "belief_mcts": BeliefMCTSAgent,
    "ismcts": ISMCTSAgent,
    "musesfish": MusesfishAgent,
    "musesfish_cpp": MusesfishCppAgent,
}


def _make_agent(
    name: str,
    seed: int,
    checkpoint: str | None = None,
    deterministic: bool = False,
) -> Any:
    if name == "policy":
        if checkpoint is None:
            raise ValueError("--checkpoint required for policy agent")
        return PolicyAgent(checkpoint, deterministic=deterministic, seed=seed)
    cls = _SIMPLE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent '{name}'. Options: {list(_SIMPLE_REGISTRY)} + policy")
    return cls(seed=seed)


def run_single_game(
    env: JieqiEnv,
    red: Any,
    black: Any,
    seed: int,
) -> dict:
    """Run one game. Returns dict with steps, winner, illegal_count, error."""
    env.reset(seed=seed)
    steps = 0
    illegal_count = 0

    try:
        done = False
        while not done:
            if env.current_player() == 0:
                action = red.select_action(env)
            else:
                action = black.select_action(env)

            # Validate action legality
            if action not in env.legal_actions():
                illegal_count += 1
                action = env.legal_actions()[0]  # fallback

            _obs, reward, terminated, truncated, _info = env.step(action)
            steps += 1

            if terminated:
                if reward > 0:
                    winner = "red" if env.current_player() == 1 else "black"
                else:
                    winner = "draw"
                return {"steps": steps, "winner": winner, "illegal": illegal_count, "error": None}
            elif truncated:
                return {"steps": steps, "winner": "draw", "illegal": illegal_count, "error": None}
    except Exception as exc:
        return {"steps": steps, "winner": "draw", "illegal": illegal_count, "error": str(exc)}

    return {"steps": steps, "winner": "draw", "illegal": illegal_count, "error": None}


def _run_eval_game_task(args: tuple) -> dict:
    (
        g,
        agent_a_name,
        agent_b_name,
        max_steps,
        seed,
        checkpoint_a,
        checkpoint_b,
        deterministic,
        swap,
        half,
    ) = args
    if swap and g < half:
        red_name, black_name = agent_a_name, agent_b_name
        red_ckpt, black_ckpt = checkpoint_a, checkpoint_b
        a_is_red = True
    elif swap:
        red_name, black_name = agent_b_name, agent_a_name
        red_ckpt, black_ckpt = checkpoint_b, checkpoint_a
        a_is_red = False
    else:
        red_name, black_name = agent_a_name, agent_b_name
        red_ckpt, black_ckpt = checkpoint_a, checkpoint_b
        a_is_red = True

    env = JieqiEnv(max_steps=max_steps)
    red = _make_agent(red_name, seed=seed + g * 2, checkpoint=red_ckpt, deterministic=deterministic)
    black = _make_agent(black_name, seed=seed + g * 2 + 1, checkpoint=black_ckpt, deterministic=deterministic)
    try:
        game = run_single_game(env, red, black, seed=seed + g)
    finally:
        for agent in (red, black):
            if hasattr(agent, "close"):
                agent.close()
    game["game_index"] = g
    game["a_is_red"] = a_is_red
    return game


def _accumulate_game_stats(result: dict, game: dict) -> None:
    result["avg_steps"] += game["steps"]
    result["illegal_actions"] += game["illegal"]
    if game["error"]:
        result["errors"] += 1

    a_is_red = game["a_is_red"]
    if game["winner"] == "red":
        if a_is_red:
            result["agent_a"]["wins"] += 1
            result["agent_a"]["as_red_wins"] += 1
        else:
            result["agent_b"]["wins"] += 1
            result["agent_b"]["as_red_wins"] += 1
    elif game["winner"] == "black":
        if a_is_red:
            result["agent_b"]["wins"] += 1
            result["agent_b"]["as_black_wins"] += 1
        else:
            result["agent_a"]["wins"] += 1
            result["agent_a"]["as_black_wins"] += 1
    else:
        result["draws"] += 1


def run_eval(
    agent_a_name: str,
    agent_b_name: str,
    n_games: int = 100,
    max_steps: int = 300,
    seed: int = 0,
    checkpoint_a: str | None = None,
    checkpoint_b: str | None = None,
    deterministic: bool = False,
    swap: bool = True,
    workers: int = 1,
    progress_interval: int = 0,
) -> dict:
    """Run *n_games* between agent A and agent B, optionally swapping colours."""

    half = n_games // 2 if swap else n_games
    result = {
        "agent_a": {
            "name": agent_a_name,
            "wins": 0,
            "as_red_wins": 0,
            "as_black_wins": 0,
            "as_red_total": half if swap else n_games,
            "as_black_total": n_games - half if swap else 0,
        },
        "agent_b": {
            "name": agent_b_name,
            "wins": 0,
            "as_red_wins": 0,
            "as_black_wins": 0,
            "as_red_total": n_games - half if swap else 0,
            "as_black_total": half if swap else n_games,
        },
        "draws": 0,
        "games": n_games,
        "avg_steps": 0.0,
        "illegal_actions": 0,
        "errors": 0,
    }
    tasks = [
        (
            g,
            agent_a_name,
            agent_b_name,
            max_steps,
            seed,
            checkpoint_a,
            checkpoint_b,
            deterministic,
            swap,
            half,
        )
        for g in range(n_games)
    ]

    if workers <= 1:
        for idx, task in enumerate(tasks, 1):
            _accumulate_game_stats(result, _run_eval_game_task(task))
            if progress_interval > 0 and (idx % progress_interval == 0 or idx == n_games):
                print(f"  progress: {idx}/{n_games} games", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_eval_game_task, task) for task in tasks]
            completed = 0
            for fut in as_completed(futures):
                _accumulate_game_stats(result, fut.result())
                completed += 1
                if progress_interval > 0 and (completed % progress_interval == 0 or completed == n_games):
                    print(f"  progress: {completed}/{n_games} games", flush=True)

    result["avg_steps"] /= max(n_games, 1)
    return result


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  -"
    return f"{n / total * 100:5.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="Agent vs Agent evaluation")
    p.add_argument("--agent-a", default="random", help="Agent A name")
    p.add_argument("--agent-b", default="random", help="Agent B name")
    p.add_argument("--checkpoint-a", default=None, help="Checkpoint for agent A (policy)")
    p.add_argument("--checkpoint-b", default=None, help="Checkpoint for agent B (policy)")
    p.add_argument("--games", type=int, default=100, help="Number of games")
    p.add_argument("--max-steps", type=int, default=300, help="Max steps per game")
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    p.add_argument("--deterministic", action="store_true", help="Policy agent deterministic mode")
    p.add_argument("--no-swap", action="store_true", help="Disable colour swap")
    p.add_argument("--workers", type=int, default=1, help="Parallel game workers")
    p.add_argument("--progress-interval", type=int, default=0, help="Print progress every N completed games")
    p.add_argument("--output", default=None, help="Save JSON result to file")
    args = p.parse_args()

    swap = not args.no_swap
    print(f"Agent A ({args.agent_a})  vs  Agent B ({args.agent_b})  [{args.games} games]")
    if swap:
        print(f"  (colour swap enabled: {args.games // 2} games each as Red)")
    if args.workers > 1:
        print(f"  (parallel workers: {args.workers})")
    print()

    result = run_eval(
        agent_a_name=args.agent_a,
        agent_b_name=args.agent_b,
        n_games=args.games,
        max_steps=args.max_steps,
        seed=args.seed,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        deterministic=args.deterministic,
        swap=swap,
        workers=args.workers,
        progress_interval=args.progress_interval,
    )

    a = result["agent_a"]
    b = result["agent_b"]

    print("=" * 60)
    print(f"  {a['name']:>10} wins: {a['wins']:4d}  ({_pct(a['wins'], result['games'])})")
    if a["as_red_total"] > 0:
        print(f"           as Red: {a['as_red_wins']:4d}  ({_pct(a['as_red_wins'], a['as_red_total'])})")
    if a["as_black_total"] > 0:
        print(f"         as Black: {a['as_black_wins']:4d}  ({_pct(a['as_black_wins'], a['as_black_total'])})")
    print(f"  {b['name']:>10} wins: {b['wins']:4d}  ({_pct(b['wins'], result['games'])})")
    if b["as_red_total"] > 0:
        print(f"           as Red: {b['as_red_wins']:4d}  ({_pct(b['as_red_wins'], b['as_red_total'])})")
    if b["as_black_total"] > 0:
        print(f"         as Black: {b['as_black_wins']:4d}  ({_pct(b['as_black_wins'], b['as_black_total'])})")
    print(f"  {'Draws':>10}: {result['draws']:4d}  ({_pct(result['draws'], result['games'])})")
    print(f"  {'Avg steps':>10}: {result['avg_steps']:.1f}")
    if result["illegal_actions"]:
        print(f"  {'Illegal':>10}: {result['illegal_actions']}")
    if result["errors"]:
        print(f"  {'Errors':>10}: {result['errors']}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON saved to {args.output}")


if __name__ == "__main__":
    main()
