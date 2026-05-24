#!/usr/bin/env python3
"""AlphaZero-style self-play training for Jieqi."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.arena import AgentConfig, Arena
from jieqi.env import JieqiEnv
from rl.az_selfplay import generate_az_selfplay_data
from rl.az_data import AZDataset
from rl.ismcts import ISMCTSAgent
from rl.model import create_model


def train_one_iteration(dataset: AZDataset, model, optimizer, device, epochs, batch_size):
    obs_t, _, policy_t, value_t = dataset.to_tensors(device)
    ds = TensorDataset(obs_t, policy_t, value_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_p, total_v, n = 0.0, 0.0, 0
        for bo, bp, bv in loader:
            logits, values = model(bo)
            p_loss = nn.functional.cross_entropy(logits, bp)
            v_loss = nn.functional.mse_loss(values.squeeze(-1), bv)
            loss = p_loss + 0.5 * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_p += p_loss.item(); total_v += v_loss.item(); n += 1
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"    epoch {epoch:3d} | p_loss {total_p/max(n,1):.4f} | v_loss {total_v/max(n,1):.4f}")


def quick_eval(ckpt: str, opp: str, n_games: int, max_steps: int) -> float:
    cfg_a = AgentConfig("az", "policy", checkpoint=ckpt, deterministic=True)
    cfg_b = AgentConfig(opp, opp)
    arena = Arena([cfg_a, cfg_b])
    mr = arena.run_match(cfg_a, cfg_b, n_games=n_games, max_steps=max_steps, seed=99999)
    return mr.a_win_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--selfplay-games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", type=str, default="simple_cnn")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=2)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="az_ckpt")
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model_kwargs = {}
    if args.model == "resnet":
        model_kwargs = {"channels": args.channels, "num_blocks": args.blocks}
    model = create_model(args.model, **model_kwargs).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Initialized from {args.init_checkpoint}")

    best_winrate = -1.0
    replay_buffer: AZDataset | None = None

    for it in range(1, args.iterations + 1):
        print(f"\n=== Iteration {it}/{args.iterations} ===")

        # Save current model
        ckpt_path = os.path.join(args.checkpoint_dir, f"iter_{it}.pt")
        torch.save({"model": model.state_dict(), "model_config": {"type": args.model, **model_kwargs}}, ckpt_path)

        # Self-play
        print(f"  Self-play: {args.selfplay_games} games, {args.simulations} sims ...")
        ismcts = ISMCTSAgent(
            num_simulations=args.simulations, temperature=args.temperature,
            evaluator="material", seed=args.seed + it,
        )
        dataset = generate_az_selfplay_data(
            ismcts, args.selfplay_games, args.max_steps, args.seed + it * 1000, replay_buffer,
        )
        replay_buffer = dataset
        print(f"  Dataset: {len(dataset)} samples")

        # Train
        print(f"  Training {args.epochs} epochs ...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        train_one_iteration(dataset, model, optimizer, device, args.epochs, args.batch_size)

        # Save best
        torch.save({"model": model.state_dict(), "model_config": {"type": args.model, **model_kwargs}},
                   os.path.join(args.checkpoint_dir, "latest.pt"))

        # Eval
        if it % args.eval_interval == 0:
            print(f"  Evaluation ({args.eval_games} games each) ...")
            wr = quick_eval(ckpt_path, "random", args.eval_games, min(150, args.max_steps))
            wg = quick_eval(ckpt_path, "greedy", args.eval_games, min(150, args.max_steps))
            print(f"    vs random: {wr:.1%}  vs greedy: {wg:.1%}")
            if wr > best_winrate:
                best_winrate = wr
                torch.save({"model": model.state_dict(), "model_config": {"type": args.model, **model_kwargs}},
                           os.path.join(args.checkpoint_dir, "best.pt"))

    print(f"\nDone. Checkpoints → {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
