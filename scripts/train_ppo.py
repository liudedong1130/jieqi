#!/usr/bin/env python3
"""PPO self-play training for Jieqi with CSV logging and periodic eval."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import deque

from evaluation.arena import AgentConfig, Arena
from jieqi.env import JieqiEnv
from rl.opponent_pool import OpponentPool
from rl.trainer import PPOTrainer

# =============================================================================
#  CSV Logger
# =============================================================================


class CSVLogger:
    def __init__(self, path: str, headers: list[str]) -> None:
        self._path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(headers)
        self._file.flush()

    def log(self, row: list) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# =============================================================================
#  Quick eval
# =============================================================================


def quick_eval(checkpoint_path: str, opponent: str, n_games: int = 10, max_steps: int = 200) -> dict:
    policy_cfg = AgentConfig("policy", "policy", checkpoint=checkpoint_path, deterministic=True)
    opp_cfg = AgentConfig(opponent, opponent)
    arena = Arena([policy_cfg, opp_cfg])
    mr = arena.run_match(policy_cfg, opp_cfg, n_games=n_games, max_steps=max_steps, seed=9999)
    return {
        "opponent": opponent,
        "win_rate": round(mr.a_win_rate, 3),
        "draw_rate": round(mr.draw_rate, 3),
        "loss_rate": round(mr.b_win_rate, 3),
        "avg_steps": round(mr.avg_steps, 1),
    }


# =============================================================================
#  Main
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description="PPO self-play training for Jieqi")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="ckpt")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet"])
    p.add_argument("--channels", type=int, default=128, help="ResNet channels")
    p.add_argument("--blocks", type=int, default=3, help="ResNet residual blocks")
    p.add_argument("--init-checkpoint", type=str, default=None, help="Init model from pretrained checkpoint")
    p.add_argument("--opponent-pool", action="store_true", help="Enable opponent pool training")
    p.add_argument("--add-checkpoint-interval", type=int, default=100, help="Add self to pool every N episodes")
    p.add_argument("--self-play-prob", type=float, default=0.5, help="Probability of self-play vs opponent")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    PPOTrainer.set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # CSV logger
    csv_headers = [
        "episode", "total_steps", "avg_return", "avg_len",
        "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_frac", "explained_var",
        "eval_vs_random", "eval_vs_greedy", "eval_vs_belief_mcts",
    ]
    csv_path = os.path.join(args.checkpoint_dir, "metrics.csv")
    logger = CSVLogger(csv_path, csv_headers)

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
        device=args.device,
        model_type=args.model,
        model_kwargs={"channels": args.channels, "num_blocks": args.blocks} if args.model == "resnet" else {},
    )

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} (ep {trainer.episode_count})")
    elif args.init_checkpoint:
        cfg = trainer.load_model_only(args.init_checkpoint)
        print(f"Initialized model from {args.init_checkpoint}")
        if cfg:
            print(f"  model_config: {cfg}")

    ep_returns: deque = deque(maxlen=100)
    ep_lengths: deque = deque(maxlen=100)
    t_start = time()
    best_vs_random = -1.0
    best_vs_greedy = -1.0
    best_vs_belief_mcts = -1.0

    # Opponent pool setup
    pool: OpponentPool | None = None
    if args.opponent_pool:
        pool = OpponentPool()
        print(f"Opponent pool: {len(pool)} agents (random, greedy)")
        if args.init_checkpoint:
            pool.add_policy(args.init_checkpoint, "init_pretrained")

    for ep in range(1, args.episodes + 1):
        if pool is not None:
            opp_cfg = pool.sample(random.Random(args.seed + ep * 3))
            opponent = OpponentPool.make_agent(opp_cfg, seed=args.seed + ep * 3 + 1)
            stats = trainer.collect_episode_with_opponent(
                seed=args.seed + ep, opponent=opponent,
                self_play_prob=args.self_play_prob,
            )
        else:
            stats = trainer.collect_episode(seed=args.seed + ep)
        ep_returns.append(stats["return"])
        ep_lengths.append(stats["steps"])

        if ep % args.episodes_per_update == 0:
            loss_stats = trainer.update()
            if loss_stats.get("nan_detected", 0) > 0:
                print(f"\n!!! NaN detected at episode {trainer.episode_count} !!!")
                dbg_path = os.path.join(args.checkpoint_dir, f"nan_debug_ep{trainer.episode_count}.pt")
                trainer.save(dbg_path)
                print(f"Debug checkpoint saved to {dbg_path}")
                csv_path2 = os.path.join(args.checkpoint_dir, "metrics.csv")
                logger.close()
                sys.exit(1)
        else:
            loss_stats = {}

        # Periodic eval
        eval_vs_random = None
        eval_vs_greedy = None
        eval_vs_belief_mcts = None
        if args.eval_interval > 0 and ep % args.eval_interval == 0:
            tmp_ckpt = os.path.join(args.checkpoint_dir, "_eval_tmp.pt")
            trainer.save(tmp_ckpt)
            er = quick_eval(tmp_ckpt, "random", n_games=args.eval_games, max_steps=min(200, args.max_steps))
            eg = quick_eval(tmp_ckpt, "greedy", n_games=args.eval_games, max_steps=min(200, args.max_steps))
            eb = quick_eval(tmp_ckpt, "belief_mcts", n_games=min(args.eval_games, 5), max_steps=min(150, args.max_steps))
            eval_vs_random = er["win_rate"]
            eval_vs_greedy = eg["win_rate"]
            eval_vs_belief_mcts = eb["win_rate"]
            if er["win_rate"] > best_vs_random:
                best_vs_random = er["win_rate"]
                trainer.save(os.path.join(args.checkpoint_dir, "best_vs_random.pt"))
            if eg["win_rate"] > best_vs_greedy:
                best_vs_greedy = eg["win_rate"]
                trainer.save(os.path.join(args.checkpoint_dir, "best_vs_greedy.pt"))
            if eb["win_rate"] > best_vs_belief_mcts:
                best_vs_belief_mcts = eb["win_rate"]
                trainer.save(os.path.join(args.checkpoint_dir, "best_vs_belief_mcts.pt"))
            os.remove(tmp_ckpt)
            print(f"  eval: vs random {er['win_rate']:.1%}  vs greedy {eg['win_rate']:.1%}  vs bmcts {eb['win_rate']:.1%}")

        # Add current model to opponent pool
        if pool is not None and args.add_checkpoint_interval > 0 and ep % args.add_checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"pool_ep{ep}.pt")
            trainer.save(ckpt_path)
            pool.add_policy(ckpt_path, f"self_ep{ep}")
            print(f"  pool: added self_ep{ep} ({len(pool)} total)")

        # Logging
        if ep % args.log_interval == 0:
            avg_r = sum(ep_returns) / max(len(ep_returns), 1)
            avg_l = sum(ep_lengths) / max(len(ep_lengths), 1)
            elapsed = time() - t_start
            row = [
                trainer.episode_count,
                sum(ep_lengths),
                round(avg_r, 4),
                round(avg_l, 1),
                round(loss_stats.get("policy_loss", 0.0), 4),
                round(loss_stats.get("value_loss", 0.0), 4),
                round(loss_stats.get("entropy", 0.0), 4),
                round(loss_stats.get("approx_kl", 0.0), 6),
                round(loss_stats.get("clip_frac", 0.0), 4),
                round(loss_stats.get("explained_var", 0.0), 4),
                round(eval_vs_random, 4) if eval_vs_random is not None else "",
                round(eval_vs_greedy, 4) if eval_vs_greedy is not None else "",
                round(eval_vs_belief_mcts, 4) if eval_vs_belief_mcts is not None else "",
            ]
            logger.log(row)
            print(
                f"ep {trainer.episode_count:5d} | "
                f"ret {avg_r:+.2f} | len {avg_l:5.0f} | "
                f"p_loss {loss_stats.get('policy_loss', 0):.3f} | "
                f"v_loss {loss_stats.get('value_loss', 0):.3f} | "
                f"ent {loss_stats.get('entropy', 0):.3f} | "
                f"kl {loss_stats.get('approx_kl', 0):.5f} | "
                f"ev {loss_stats.get('explained_var', 0):.2f} | "
                f"{elapsed:.0f}s"
            )

        # Checkpoint
        if ep % args.save_interval == 0:
            trainer.save(os.path.join(args.checkpoint_dir, f"snapshot_ep{trainer.episode_count}.pt"))

        # Always save latest
        trainer.save(os.path.join(args.checkpoint_dir, "latest.pt"))

    logger.close()
    print(f"\nDone. Metrics → {csv_path}")


if __name__ == "__main__":
    main()
