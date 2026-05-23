#!/usr/bin/env python3
"""Compare PPO training: random init vs pretrained init."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import deque

from evaluation.arena import AgentConfig, Arena
from jieqi.env import JieqiEnv
from rl.trainer import PPOTrainer


def quick_eval(ckpt_path: str, opponent: str, n_games: int, max_steps: int) -> dict:
    policy_cfg = AgentConfig("p", "policy", checkpoint=ckpt_path, deterministic=True)
    opp_cfg = AgentConfig(opponent, opponent)
    arena = Arena([policy_cfg, opp_cfg])
    mr = arena.run_match(policy_cfg, opp_cfg, n_games=n_games, max_steps=max_steps, seed=99999)
    return {"win_rate": round(mr.a_win_rate, 3), "draw_rate": round(mr.draw_rate, 3)}


def train_one(
    label: str,
    ckpt_dir: str,
    episodes: int,
    max_steps: int,
    eval_interval: int,
    seed: int,
    model_type: str,
    model_kwargs: dict,
    init_checkpoint: str | None,
) -> str:
    """Train and return path to CSV metrics file."""
    PPOTrainer.set_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "avg_return", "avg_len", "eval_vs_random", "eval_vs_greedy"])

    env = JieqiEnv(max_steps=max_steps)
    trainer = PPOTrainer(
        env, episodes_per_update=8,
        model_type=model_type, model_kwargs=model_kwargs,
    )

    if init_checkpoint:
        trainer.load_model_only(init_checkpoint)

    ep_returns = deque(maxlen=50)
    ep_lengths = deque(maxlen=50)

    for ep in range(1, episodes + 1):
        stats = trainer.collect_episode(seed=seed + ep)
        ep_returns.append(stats["return"])
        ep_lengths.append(stats["steps"])

        if ep % 8 == 0:
            trainer.update()

        if eval_interval > 0 and ep % eval_interval == 0:
            tmp = os.path.join(ckpt_dir, "_tmp.pt")
            trainer.save(tmp)
            er = quick_eval(tmp, "random", 5, 150)
            eg = quick_eval(tmp, "greedy", 5, 150)
            os.remove(tmp)

            avg_r = sum(ep_returns) / max(len(ep_returns), 1)
            avg_l = sum(ep_lengths) / max(len(ep_lengths), 1)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ep, round(avg_r, 3), round(avg_l, 1),
                            round(er["win_rate"], 3), round(eg["win_rate"], 3)])
            print(f"  [{label}] ep {ep:4d} | ret {avg_r:+.2f} | len {avg_l:.0f} | "
                  f"vs_r {er['win_rate']:.1%} | vs_g {eg['win_rate']:.1%}")

    return csv_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-ckpt", type=str, required=True, help="Pretrained checkpoint path")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--model", type=str, default="simple_cnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="/tmp/ppo_compare")
    args = p.parse_args()

    model_kwargs = {}
    if args.model == "resnet":
        model_kwargs = {"channels": 64, "num_blocks": 2}

    t0 = time()

    print("=== Scratch (random init) ===")
    scratch_dir = os.path.join(args.output_dir, "scratch")
    scratch_csv = train_one(
        "scratch", scratch_dir, args.episodes, args.max_steps,
        args.eval_interval, args.seed, args.model, model_kwargs, None,
    )

    print("\n=== Pretrained init ===")
    pretrain_dir = os.path.join(args.output_dir, "pretrained")
    pretrain_csv = train_one(
        "pretrained", pretrain_dir, args.episodes, args.max_steps,
        args.eval_interval, args.seed + 1000, args.model, model_kwargs,
        args.pretrained_ckpt,
    )

    print(f"\nDone in {time() - t0:.0f}s")
    print(f"Scratch CSV:     {scratch_csv}")
    print(f"Pretrained CSV:  {pretrain_csv}")


if __name__ == "__main__":
    main()
