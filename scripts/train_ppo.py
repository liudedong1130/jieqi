#!/usr/bin/env python3
"""Minimal PPO self-play training script for Jieqi."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jieqi.env import JieqiEnv
from rl.trainer import PPOTrainer


def main() -> None:
    p = argparse.ArgumentParser(description="PPO self-play training for Jieqi")
    p.add_argument("--episodes", type=int, default=1000, help="Total training episodes")
    p.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    p.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    p.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    p.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    p.add_argument("--episodes-per-update", type=int, default=8, help="Episodes per PPO update")
    p.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N episodes")
    p.add_argument("--log-interval", type=int, default=10, help="Log every N episodes")
    p.add_argument("--checkpoint", type=str, default="ckpt", help="Checkpoint directory")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    args = p.parse_args()

    os.makedirs(args.checkpoint, exist_ok=True)

    env = JieqiEnv(max_steps=args.max_steps)
    trainer = PPOTrainer(
        env,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        episodes_per_update=args.episodes_per_update,
    )

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} (episode {trainer.episode_count})")

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    t_start = time()

    for ep in range(1, args.episodes + 1):
        stats = trainer.collect_episode(seed=ep)
        ep_returns.append(stats["reward"])
        ep_lengths.append(stats["steps"])

        # Update when enough episodes collected
        if ep % args.episodes_per_update == 0:
            loss_stats = trainer.update()
        else:
            loss_stats = {}

        # Logging
        if ep % args.log_interval == 0:
            avg_r = sum(ep_returns[-args.log_interval:]) / len(ep_returns[-args.log_interval:])
            avg_l = sum(ep_lengths[-args.log_interval:]) / len(ep_lengths[-args.log_interval:])
            elapsed = time() - t_start
            print(
                f"ep {trainer.episode_count:5d} | "
                f"ret {avg_r:+.3f} | "
                f"len {avg_l:5.0f} | "
                f"p_loss {loss_stats.get('policy_loss', 0):.4f} | "
                f"v_loss {loss_stats.get('value_loss', 0):.4f} | "
                f"ent {loss_stats.get('entropy', 0):.4f} | "
                f"time {elapsed:.0f}s"
            )

        # Checkpoint
        if ep % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint, f"ppo_ep{ep}.pt")
            trainer.save(ckpt_path)
            print(f"  saved {ckpt_path}")

    # Final save
    final_path = os.path.join(args.checkpoint, "ppo_final.pt")
    trainer.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
