#!/usr/bin/env python3
"""Supervised pretraining pipeline using BeliefMCTS-generated data."""

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

from agents.belief_mcts_agent import BeliefMCTSAgent
from evaluation.arena import AgentConfig, Arena
from jieqi.env import JieqiEnv
from rl.az_data import AZSample, AZDataset
from rl.model import create_model


# ---------------------------------------------------------------------------
#  Data generation
# ---------------------------------------------------------------------------


def generate_data(
    games: int, max_steps: int, seed: int, num_samples: int = 30
) -> AZDataset:
    """Generate supervised data using BeliefMCTS self-play."""
    dataset = AZDataset()

    for g in range(games):
        env = JieqiEnv(max_steps=max_steps)
        env.reset(seed=seed + g)
        agent = BeliefMCTSAgent(num_samples=num_samples, seed=seed + g)

        moves_info: list[dict] = []
        done = False
        while not done:
            player = env.current_player()
            obs = env.observation().copy()
            mask = env.legal_action_mask().copy()
            action = agent.select_action(env)
            _obs, reward, terminated, truncated, _info = env.step(action)
            moves_info.append({
                "action": action, "player": player, "obs": obs, "mask": mask,
                "reward": reward, "terminated": terminated,
            })
            done = terminated or truncated

        winner = None
        for mi in reversed(moves_info):
            if mi["terminated"] and mi["reward"] > 0:
                winner = mi["player"]
                break

        for i, mi in enumerate(moves_info):
            policy = np.zeros(8100, dtype=np.float32)
            policy[mi["action"]] = 1.0
            value = 1.0 if (winner is not None and mi["player"] == winner) else (-1.0 if (winner is not None) else 0.0)

            dataset.add(AZSample(
                observation=mi["obs"], legal_mask=mi["mask"],
                policy_target=policy, value_target=value,
                player=mi["player"], game_id=f"bmcts_{g}", move_index=i,
            ))

    return dataset


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------


def train_supervised(
    dataset: AZDataset,
    model_type: str, model_kwargs: dict,
    epochs: int, lr: float, batch_size: int, device: str,
) -> nn.Module:
    model = create_model(model_type, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    obs_t, mask_t, policy_t, value_t = dataset.to_tensors(device)

    ds = TensorDataset(obs_t, policy_t, value_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_p_loss, total_v_loss, n_batches = 0.0, 0.0, 0
        for bo, bp, bv in loader:
            logits, values = model(bo)
            p_loss = nn.functional.cross_entropy(logits, bp)
            v_loss = nn.functional.mse_loss(values.squeeze(-1), bv)
            loss = p_loss + 0.5 * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:3d} | p_loss {total_p_loss/max(n_batches,1):.4f} | v_loss {total_v_loss/max(n_batches,1):.4f}")

    return model


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def quick_eval(checkpoint_path: str, opponent: str, n_games: int, max_steps: int) -> dict:
    policy_cfg = AgentConfig("policy", "policy", checkpoint=checkpoint_path, deterministic=True)
    opp_cfg = AgentConfig(opponent, opponent)
    arena = Arena([policy_cfg, opp_cfg])
    mr = arena.run_match(policy_cfg, opp_cfg, n_games=n_games, max_steps=max_steps, seed=9999)
    return {"win_rate": round(mr.a_win_rate, 3), "draw_rate": round(mr.draw_rate, 3)}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="BeliefMCTS supervised pretraining")
    p.add_argument("--games", type=int, default=100, help="Games to generate")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--data", type=str, default=None, help="Use cached .npz dataset")
    p.add_argument("--cache-data", type=str, default=None, help="Save generated data")
    p.add_argument("--model", type=str, default="simple_cnn")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--checkpoint-out", type=str, default="pretrained.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Step 1: Data
    if args.data:
        print(f"Loading cached data from {args.data} ...")
        dataset = AZDataset()
        dataset.load(args.data)
    else:
        print(f"Generating {args.games} games with BeliefMCTS ...")
        dataset = generate_data(args.games, args.max_steps, args.seed)
        if args.cache_data:
            dataset.save(args.cache_data)
            print(f"  Cached to {args.cache_data}")
    print(f"  Dataset: {len(dataset)} samples")

    # Step 2: Train
    print(f"Training {args.model} for {args.epochs} epochs ...")
    model_kwargs = {}
    if args.model == "resnet":
        model_kwargs = {"channels": args.channels, "num_blocks": args.blocks}
    model = train_supervised(dataset, args.model, model_kwargs, args.epochs, args.lr, args.batch_size, device)

    ckpt = {"model": model.state_dict(), "model_config": {"type": args.model, **model_kwargs}}
    torch.save(ckpt, args.checkpoint_out)
    print(f"  Saved {args.checkpoint_out}")

    # Step 3: Eval
    print(f"\nEvaluation ({args.eval_games} games each):")
    er = quick_eval(args.checkpoint_out, "random", args.eval_games, args.max_steps)
    eg = quick_eval(args.checkpoint_out, "greedy", args.eval_games, args.max_steps)
    print(f"  vs random:  {er['win_rate']:.1%}  (draw {er['draw_rate']:.1%})")
    print(f"  vs greedy:  {eg['win_rate']:.1%}  (draw {eg['draw_rate']:.1%})")


if __name__ == "__main__":
    main()
