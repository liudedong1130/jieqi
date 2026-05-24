#!/usr/bin/env python3
"""AlphaZero-style self-play training for Jieqi."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from evaluation.arena import AgentConfig, Arena
from rl.az_selfplay import generate_az_selfplay_data
from rl.az_data import AZDataset
from rl.az_train import TrainStats, train_policy_value
from rl.ismcts import ISMCTSAgent
from rl.model import create_model


def _model_config(model_type: str, model_kwargs: dict) -> dict:
    return {"type": model_type, **model_kwargs}


def _save_checkpoint(path: str, model, model_type: str, model_kwargs: dict) -> None:
    torch.save({"model": model.state_dict(), "model_config": _model_config(model_type, model_kwargs)}, path)


def quick_eval(ckpt: str, opp: str, n_games: int, max_steps: int) -> dict:
    n_games = max(2, n_games)
    if n_games % 2 == 1:
        n_games += 1
    cfg_a = AgentConfig("az", "policy", checkpoint=ckpt, deterministic=True)
    if opp == "ismcts_material":
        cfg_b = AgentConfig("ismcts_material", "ismcts", params={"num_simulations": 50, "max_depth": 5, "temperature": 0})
    else:
        cfg_b = AgentConfig(opp, opp)
    arena = Arena([cfg_a, cfg_b])
    mr = arena.run_match(cfg_a, cfg_b, n_games=n_games, max_steps=max_steps, seed=99999)
    return {
        "win_rate": mr.a_win_rate,
        "draw_rate": mr.draw_rate,
        "loss_rate": mr.b_win_rate,
        "avg_steps": mr.avg_steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--selfplay-games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", type=str, default="resnet")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=2)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--replay-window", type=int, default=50000)
    p.add_argument("--search-max-depth", type=int, default=5)
    p.add_argument("--progress-interval", type=int, default=10, help="Print self-play progress every N moves")
    p.add_argument("--checkpoint-dir", type=str, default="az_ckpt")
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    model_kwargs = {}
    if args.model == "resnet":
        model_kwargs = {"channels": args.channels, "num_blocks": args.blocks}
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("model_config", {"type": args.model, **model_kwargs})
        args.model = cfg.get("type", args.model)
        model_kwargs = {k: v for k, v in cfg.items() if k != "type"}
        model = create_model(args.model, **model_kwargs).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"Initialized from {args.init_checkpoint}")
    else:
        model = create_model(args.model, **model_kwargs).to(device)

    metrics_path = os.path.join(args.checkpoint_dir, "metrics.csv")
    metrics_file = open(metrics_path, "w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow([
        "iteration", "samples", "policy_loss", "value_loss", "total_loss",
        "eval_vs_random", "eval_vs_greedy", "eval_vs_belief_mcts", "eval_vs_ismcts_material",
    ])
    metrics_file.flush()

    best_vs_greedy = -1.0
    best_vs_belief_mcts = -1.0
    replay_buffer = AZDataset()

    for it in range(1, args.iterations + 1):
        print(f"\n=== Iteration {it}/{args.iterations} ===")

        # Save current model
        search_ckpt = os.path.join(args.checkpoint_dir, f"iter_{it}_search.pt")
        _save_checkpoint(search_ckpt, model, args.model, model_kwargs)

        # Self-play
        print(f"  Self-play: {args.selfplay_games} games, {args.simulations} sims ...")
        use_policy_value = args.init_checkpoint is not None or it > 1
        search_kwargs = {
            "num_simulations": args.simulations,
            "max_depth": args.search_max_depth,
            "temperature": args.temperature,
            "seed": args.seed + it,
        }
        if use_policy_value:
            search_kwargs.update({"evaluator": "policy_value", "policy_checkpoint": search_ckpt})
        else:
            search_kwargs.update({"evaluator": "material"})
        ismcts = ISMCTSAgent(**search_kwargs)
        replay_buffer = generate_az_selfplay_data(
            ismcts,
            args.selfplay_games,
            args.max_steps,
            args.seed + it * 1000,
            replay_buffer,
            max_buffer_samples=args.replay_window,
            progress_interval=args.progress_interval,
        )
        print(f"  Replay buffer: {len(replay_buffer)} samples")

        # Train
        print(f"  Training {args.epochs} epochs ...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        stats: TrainStats = train_policy_value(
            replay_buffer,
            model,
            optimizer,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            log_prefix="    ",
        )

        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
        iter_path = os.path.join(args.checkpoint_dir, f"iter_{it}.pt")
        _save_checkpoint(latest_path, model, args.model, model_kwargs)
        _save_checkpoint(iter_path, model, args.model, model_kwargs)

        # Eval
        evals = {"random": None, "greedy": None, "belief_mcts": None, "ismcts_material": None}
        if args.eval_interval > 0 and it % args.eval_interval == 0:
            print(f"  Evaluation ({args.eval_games} games each) ...")
            eval_games = max(args.eval_games, 1)
            evals["random"] = quick_eval(latest_path, "random", eval_games, min(150, args.max_steps))
            evals["greedy"] = quick_eval(latest_path, "greedy", eval_games, min(150, args.max_steps))
            evals["belief_mcts"] = quick_eval(latest_path, "belief_mcts", min(eval_games, 5), min(150, args.max_steps))
            evals["ismcts_material"] = quick_eval(latest_path, "ismcts_material", min(eval_games, 4), min(120, args.max_steps))
            print(
                f"    vs random: {evals['random']['win_rate']:.1%}  "
                f"vs greedy: {evals['greedy']['win_rate']:.1%}  "
                f"vs bmcts: {evals['belief_mcts']['win_rate']:.1%}  "
                f"vs ismcts: {evals['ismcts_material']['win_rate']:.1%}"
            )
            if evals["greedy"]["win_rate"] > best_vs_greedy:
                best_vs_greedy = evals["greedy"]["win_rate"]
                _save_checkpoint(os.path.join(args.checkpoint_dir, "best_vs_greedy.pt"), model, args.model, model_kwargs)
            if evals["belief_mcts"]["win_rate"] > best_vs_belief_mcts:
                best_vs_belief_mcts = evals["belief_mcts"]["win_rate"]
                _save_checkpoint(os.path.join(args.checkpoint_dir, "best_vs_belief_mcts.pt"), model, args.model, model_kwargs)

        metrics.writerow([
            it,
            len(replay_buffer),
            round(stats.policy_loss, 6),
            round(stats.value_loss, 6),
            round(stats.total_loss, 6),
            round(evals["random"]["win_rate"], 4) if evals["random"] else "",
            round(evals["greedy"]["win_rate"], 4) if evals["greedy"] else "",
            round(evals["belief_mcts"]["win_rate"], 4) if evals["belief_mcts"] else "",
            round(evals["ismcts_material"]["win_rate"], 4) if evals["ismcts_material"] else "",
        ])
        metrics_file.flush()

    metrics_file.close()
    print(f"\nDone. Checkpoints → {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
