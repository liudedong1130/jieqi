#!/usr/bin/env python3
"""Benchmark agent inference latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agents.belief_mcts_agent import BeliefMCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.policy_agent import PolicyAgent
from agents.random_agent import RandomAgent
from jieqi.env import JieqiEnv
from rl.ismcts import ISMCTSAgent


def percentile(arr: list[float], p: float) -> float:
    return float(np.percentile(arr, p)) if arr else 0.0


def benchmark_agent(
    name: str, agent, env: JieqiEnv, num_moves: int = 50,
) -> dict:
    """Measure move latency for *agent*."""
    env.reset(seed=42)
    latencies: list[float] = []

    for _ in range(num_moves):
        actions = env.legal_actions()
        if not actions:
            break
        t0 = time.perf_counter()
        action = agent.select_action(env)
        dt = time.perf_counter() - t0
        latencies.append(dt * 1000)  # ms
        env.step(action)

    arr = np.array(latencies)
    return {
        "agent": name,
        "moves": len(latencies),
        "avg_ms": round(float(arr.mean()), 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p90_ms": round(percentile(latencies, 90), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--moves", type=int, default=30)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    results = []

    # Random / Greedy
    env = JieqiEnv(max_steps=200)
    for name, cls in [("random", RandomAgent), ("greedy", GreedyAgent)]:
        print(f"Benchmarking {name} ...")
        r = benchmark_agent(name, cls(seed=1), env, args.moves)
        results.append(r)

    # BeliefMCTS (10 / 30 samples)
    for ns in [10, 30]:
        name = f"belief_mcts({ns})"
        print(f"Benchmarking {name} ...")
        r = benchmark_agent(name, BeliefMCTSAgent(num_samples=ns, seed=1), env, min(args.moves, 15))
        results.append(r)

    # ISMCTS (50 / 100 simulations)
    for ns in [50, 100]:
        name = f"ismcts({ns})"
        print(f"Benchmarking {name} ...")
        r = benchmark_agent(
            name, ISMCTSAgent(num_simulations=ns, max_depth=5, temperature=1.0, seed=1),
            env, min(args.moves, 10),
        )
        results.append(r)

    # Policy agent (if checkpoint provided)
    if args.checkpoint:
        print(f"Benchmarking policy ...")
        pa = PolicyAgent(args.checkpoint, deterministic=True, seed=1)
        # Measure model forward time separately
        env2 = JieqiEnv(max_steps=200)
        env2.reset(seed=42)
        obs = env2.observation()
        import torch
        t = torch.from_numpy(obs).unsqueeze(0).to(pa.device)
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                pa.model(t)
        # Measure
        fwd_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            with torch.no_grad():
                pa.model(t)
            fwd_times.append((time.perf_counter() - t0) * 1000)
        fwd_arr = np.array(fwd_times)
        print(f"  model forward: {fwd_arr.mean():.2f}ms avg")

        r = benchmark_agent("policy", pa, env, args.moves)
        results.append(r)

    # Print table
    print("\n" + "=" * 75)
    print(f"{'Agent':<22} {'Moves':>5} {'Avg(ms)':>8} {'p50(ms)':>8} {'p90(ms)':>8} {'p99(ms)':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['agent']:<22} {r['moves']:>5} {r['avg_ms']:>8.1f} {r['p50_ms']:>8.1f} {r['p90_ms']:>8.1f} {r['p99_ms']:>8.1f}")
    print("=" * 75)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
