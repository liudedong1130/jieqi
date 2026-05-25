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

from agents.musesfish_agent import MusesfishAgent
from agents.musesfish_cpp_agent import MusesfishCppAgent
from evaluation.arena import AgentConfig, Arena
from jieqi.env import JieqiEnv
from rl.opponent_pool import OpponentPool
from rl.trainer import PPOTrainer
from utils.run_manager import RunManager

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
    p.add_argument("--eval-opponents", type=str, default="random,greedy,belief_mcts",
                   help="Comma-separated eval opponents: random,greedy,belief_mcts,musesfish,musesfish_cpp")
    p.add_argument("--checkpoint-dir", type=str, default="ckpt")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet"])
    p.add_argument("--channels", type=int, default=128, help="ResNet channels")
    p.add_argument("--blocks", type=int, default=3, help="ResNet residual blocks")
    p.add_argument("--init-checkpoint", type=str, default=None, help="Init model from pretrained checkpoint")
    p.add_argument("--opponent-pool", action="store_true", help="Enable opponent pool training")
    p.add_argument("--musesfish-opponent", action="store_true", help="Add Musesfish rule agent to opponent pool")
    p.add_argument("--opponents", type=str, default=None,
                   help="Comma-separated built-in opponents for pool: random,greedy,belief_mcts,musesfish,musesfish_cpp")
    p.add_argument("--imitation-agent", type=str, default=None, choices=["musesfish", "musesfish_cpp"],
                   help="Optional per-state imitation teacher for PPO auxiliary loss")
    p.add_argument("--imitation-coef", type=float, default=0.0,
                   help="Weight for imitation cross-entropy loss; 0 disables it")
    p.add_argument("--musesfish-cpp-min-depth", type=int, default=3, help="C++ Musesfish min depth for PPO imitation")
    p.add_argument("--musesfish-cpp-max-depth", type=int, default=4, help="C++ Musesfish max depth for PPO imitation")
    p.add_argument("--musesfish-cpp-timeout", type=float, default=2.0, help="C++ Musesfish timeout per PPO imitation move")
    p.add_argument("--add-checkpoint-interval", type=int, default=100, help="Add self to pool every N episodes")
    p.add_argument("--self-play-prob", type=float, default=0.5, help="Probability of self-play vs opponent")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    PPOTrainer.set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Run manager
    run = RunManager(
        algo="ppo", seed=args.seed,
        config={"algo": "ppo", "model": args.model, "episodes": args.episodes,
                 "lr": args.lr, "gamma": args.gamma, "seed": args.seed,
                 "init_checkpoint": args.init_checkpoint or "none",
                 "imitation_agent": args.imitation_agent or "none",
                 "imitation_coef": args.imitation_coef},
    )

    # CSV logger
    csv_headers = [
        "episode", "total_steps", "avg_return", "avg_len",
        "policy_loss", "value_loss", "entropy", "imitation_loss",
        "approx_kl", "clip_frac", "explained_var",
        "eval_vs_random", "eval_vs_greedy", "eval_vs_belief_mcts", "eval_vs_musesfish", "eval_vs_musesfish_cpp",
    ]
    csv_path = os.path.join(args.checkpoint_dir, "metrics.csv")
    logger = CSVLogger(csv_path, csv_headers)

    env = JieqiEnv(max_steps=args.max_steps)
    imitation_agent = None
    if args.imitation_agent and args.imitation_coef > 0:
        if args.imitation_agent == "musesfish_cpp":
            imitation_agent = MusesfishCppAgent(
                seed=args.seed + 99991,
                timeout=args.musesfish_cpp_timeout,
                min_depth=args.musesfish_cpp_min_depth,
                max_depth=args.musesfish_cpp_max_depth,
                fallback=False,
            )
        else:
            imitation_agent = MusesfishAgent(seed=args.seed + 99991)
        print(f"Imitation teacher: {args.imitation_agent} (coef={args.imitation_coef:g})")
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
        imitation_agent=imitation_agent,
        imitation_coef=args.imitation_coef,
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
    best_vs_musesfish = -1.0
    best_by_opponent = {
        "random": best_vs_random,
        "greedy": best_vs_greedy,
        "belief_mcts": best_vs_belief_mcts,
        "musesfish": best_vs_musesfish,
        "musesfish_cpp": -1.0,
    }
    eval_opponents = [o.strip() for o in args.eval_opponents.split(",") if o.strip()]

    # Opponent pool setup
    pool: OpponentPool | None = None
    eval_musesfish = args.musesfish_opponent
    if args.opponent_pool:
        configured_opponents = None
        if args.opponents:
            configured_opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
            eval_musesfish = "musesfish" in configured_opponents or "musesfish_cpp" in configured_opponents
        pool = OpponentPool(
            include_musesfish=args.musesfish_opponent,
            opponents=configured_opponents,
        )
        if configured_opponents is not None:
            names = ", ".join(configured_opponents)
        else:
            names = "random, greedy" + (", musesfish" if args.musesfish_opponent else "")
        print(f"Opponent pool: {len(pool)} agents ({names})")
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
        eval_vs_musesfish = None
        eval_vs_musesfish_cpp = None
        if args.eval_interval > 0 and ep % args.eval_interval == 0:
            tmp_ckpt = os.path.join(args.checkpoint_dir, "_eval_tmp.pt")
            trainer.save(tmp_ckpt)
            eval_results: dict[str, dict] = {}
            opponents_to_eval = list(eval_opponents)
            if eval_musesfish and "musesfish" not in opponents_to_eval and "musesfish_cpp" not in opponents_to_eval:
                opponents_to_eval.append("musesfish")
            for opponent_name in opponents_to_eval:
                n_games = args.eval_games
                max_steps = min(200, args.max_steps)
                if opponent_name == "belief_mcts":
                    n_games = min(args.eval_games, 5)
                    max_steps = min(150, args.max_steps)
                result = quick_eval(tmp_ckpt, opponent_name, n_games=n_games, max_steps=max_steps)
                eval_results[opponent_name] = result
                if result["win_rate"] > best_by_opponent.get(opponent_name, -1.0):
                    best_by_opponent[opponent_name] = result["win_rate"]
                    trainer.save(os.path.join(args.checkpoint_dir, f"best_vs_{opponent_name}.pt"))
            eval_vs_random = eval_results.get("random", {}).get("win_rate")
            eval_vs_greedy = eval_results.get("greedy", {}).get("win_rate")
            eval_vs_belief_mcts = eval_results.get("belief_mcts", {}).get("win_rate")
            eval_vs_musesfish = eval_results.get("musesfish", {}).get("win_rate")
            eval_vs_musesfish_cpp = eval_results.get("musesfish_cpp", {}).get("win_rate")
            os.remove(tmp_ckpt)
            msg = "  eval:"
            for opponent_name, result in eval_results.items():
                label = "bmcts" if opponent_name == "belief_mcts" else opponent_name
                msg += f" vs {label} {result['win_rate']:.1%}"
            print(msg)
            arena_result = {
                "episode": trainer.episode_count,
            }
            for opponent_name, result in eval_results.items():
                arena_result[f"vs_{opponent_name}"] = result["win_rate"]
            run.log_arena_result(arena_result)

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
                round(loss_stats.get("imitation_loss", 0.0), 4),
                round(loss_stats.get("approx_kl", 0.0), 6),
                round(loss_stats.get("clip_frac", 0.0), 4),
                round(loss_stats.get("explained_var", 0.0), 4),
                round(eval_vs_random, 4) if eval_vs_random is not None else "",
                round(eval_vs_greedy, 4) if eval_vs_greedy is not None else "",
                round(eval_vs_belief_mcts, 4) if eval_vs_belief_mcts is not None else "",
                round(eval_vs_musesfish, 4) if eval_vs_musesfish is not None else "",
                round(eval_vs_musesfish_cpp, 4) if eval_vs_musesfish_cpp is not None else "",
            ]
            logger.log(row)
            # Also log to run manager
            metric_dict = dict(zip(csv_headers[1:], row[1:])) if len(row) == len(csv_headers) else {}
            metric_dict["episode"] = row[0] if row else 0
            run.log_metrics({k: v for k, v in metric_dict.items() if v != ""})
            print(
                f"ep {trainer.episode_count:5d} | "
                f"ret {avg_r:+.2f} | len {avg_l:5.0f} | "
                f"p_loss {loss_stats.get('policy_loss', 0):.3f} | "
                f"v_loss {loss_stats.get('value_loss', 0):.3f} | "
                f"ent {loss_stats.get('entropy', 0):.3f} | "
                f"imit {loss_stats.get('imitation_loss', 0):.3f} | "
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
    run.close()
    print(f"\nDone. Run → {run.run_dir}")


if __name__ == "__main__":
    main()
