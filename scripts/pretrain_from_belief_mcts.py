#!/usr/bin/env python3
"""Supervised pretraining pipeline using search-generated policy targets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from agents.belief_mcts_agent import BeliefMCTSAgent
from evaluation.arena import AgentConfig, Arena
from jieqi.env import JieqiEnv
from rl.az_data import AZSample, AZDataset
from rl.az_train import train_policy_value
from rl.ismcts import ISMCTSAgent
from rl.model import create_model


# ---------------------------------------------------------------------------
#  Data generation
# ---------------------------------------------------------------------------


def generate_data(
    games: int,
    max_steps: int,
    seed: int,
    *,
    teacher: str = "ismcts",
    simulations: int = 100,
    max_depth: int = 5,
    num_samples: int = 30,
    progress_interval: int = 1,
) -> AZDataset:
    """Generate supervised data using search self-play."""
    dataset = AZDataset()

    for g in range(games):
        env = JieqiEnv(max_steps=max_steps)
        env.reset(seed=seed + g)
        if teacher == "ismcts":
            agent = ISMCTSAgent(
                num_simulations=simulations,
                max_depth=max_depth,
                temperature=1.0,
                evaluator="material",
                seed=seed + g,
            )
        elif teacher == "belief_mcts":
            agent = BeliefMCTSAgent(num_samples=num_samples, seed=seed + g)
        else:
            raise ValueError(f"Unknown teacher '{teacher}'")

        moves_info: list[dict] = []
        done = False
        print(f"  game {g + 1}/{games} ...", flush=True)
        while not done:
            player = env.current_player()
            obs = env.observation().copy()
            mask = env.legal_action_mask().copy()
            if teacher == "ismcts":
                policy, action = agent.get_policy(env)
            else:
                action = agent.select_action(env)
                policy = np.zeros(8100, dtype=np.float32)
                policy[action] = 1.0
            if action not in env.legal_actions():
                action = env.legal_actions()[0]
                policy = np.zeros(8100, dtype=np.float32)
                policy[action] = 1.0
            _obs, reward, terminated, truncated, _info = env.step(action)
            moves_info.append({
                "action": action, "policy": policy, "player": player, "obs": obs, "mask": mask,
                "reward": reward, "terminated": terminated,
            })
            done = terminated or truncated
            if progress_interval > 0 and len(moves_info) % progress_interval == 0:
                print(
                    f"    game {g + 1}/{games} | step {len(moves_info)} | "
                    f"samples {len(dataset)}",
                    flush=True,
                )

        winner = None
        for mi in reversed(moves_info):
            if mi["terminated"] and mi["reward"] > 0:
                winner = mi["player"]
                break

        for i, mi in enumerate(moves_info):
            value = 1.0 if (winner is not None and mi["player"] == winner) else (-1.0 if (winner is not None) else 0.0)

            dataset.add(AZSample(
                observation=mi["obs"], legal_mask=mi["mask"],
                policy_target=mi["policy"], value_target=value,
                player=mi["player"], game_id=f"bmcts_{g}", move_index=i,
            ))
        print(
            f"  game {g + 1}/{games} done | steps {len(moves_info)} | "
            f"dataset {len(dataset)}",
            flush=True,
        )

    return dataset


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------


def train_supervised(
    dataset: AZDataset,
    model_type: str, model_kwargs: dict,
    epochs: int, lr: float, batch_size: int, device: str,
) -> torch.nn.Module:
    model = create_model(model_type, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_policy_value(dataset, model, optimizer, device, epochs=epochs, batch_size=batch_size)
    return model


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def quick_eval(checkpoint_path: str, opponent: str, n_games: int, max_steps: int) -> dict:
    n_games = max(2, n_games)
    if n_games % 2 == 1:
        n_games += 1
    policy_cfg = AgentConfig("policy", "policy", checkpoint=checkpoint_path, deterministic=True)
    opp_cfg = AgentConfig(opponent, opponent)
    arena = Arena([policy_cfg, opp_cfg])
    mr = arena.run_match(policy_cfg, opp_cfg, n_games=n_games, max_steps=max_steps, seed=9999)
    return {"win_rate": round(mr.a_win_rate, 3), "draw_rate": round(mr.draw_rate, 3)}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Search supervised pretraining")
    p.add_argument("--games", type=int, default=100, help="Games to generate")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--data", type=str, default=None, help="Use cached .npz dataset")
    p.add_argument("--cache-data", type=str, default=None, help="Save generated data")
    p.add_argument("--teacher", type=str, default="ismcts", choices=["ismcts", "belief_mcts"])
    p.add_argument("--simulations", type=int, default=100, help="ISMCTS simulations per move")
    p.add_argument("--max-depth", type=int, default=5, help="ISMCTS max search depth")
    p.add_argument("--teacher-samples", type=int, default=30, help="BeliefMCTS samples per move")
    p.add_argument("--progress-interval", type=int, default=10, help="Print progress every N moves during data generation")
    p.add_argument("--model", type=str, default="resnet")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--checkpoint-out", type=str, default="pretrained.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))

    # Step 1: Data
    if args.data:
        print(f"Loading cached data from {args.data} ...")
        dataset = AZDataset()
        dataset.load(args.data)
    else:
        print(f"Generating {args.games} games with {args.teacher} ...")
        dataset = generate_data(
            args.games,
            args.max_steps,
            args.seed,
            teacher=args.teacher,
            simulations=args.simulations,
            max_depth=args.max_depth,
            num_samples=args.teacher_samples,
            progress_interval=args.progress_interval,
        )
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
