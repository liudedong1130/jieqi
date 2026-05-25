#!/usr/bin/env python3
"""Supervised pretraining pipeline using search-generated policy targets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from agents.belief_mcts_agent import BeliefMCTSAgent
from agents.musesfish_agent import MusesfishAgent
from agents.musesfish_cpp_agent import MusesfishCppAgent
from evaluation.arena import AgentConfig, Arena
from jieqi.env import JieqiEnv
from rl.az_data import AZSample, AZDataset
from rl.az_train import train_policy_value
from rl.ismcts import ISMCTSAgent
from rl.model import create_model


# ---------------------------------------------------------------------------
#  Data generation
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[{time():.0f}] {msg}", flush=True)


def _generate_one_game(args_tuple: tuple) -> tuple[int, list[AZSample], int, str]:
    """Generate data for a single game (module-level for multiprocessing)."""
    (
        game_idx,
        max_steps,
        seed,
        teacher,
        simulations,
        max_depth,
        num_samples,
        musesfish_cpp_min_depth,
        musesfish_cpp_max_depth,
        musesfish_cpp_timeout,
    ) = args_tuple

    env = JieqiEnv(max_steps=max_steps)
    env.reset(seed=seed)
    if teacher == "ismcts":
        agent = ISMCTSAgent(
            num_simulations=simulations,
            max_depth=max_depth,
            temperature=1.0,
            evaluator="material",
            seed=seed,
        )
    elif teacher == "belief_mcts":
        agent = BeliefMCTSAgent(num_samples=num_samples, seed=seed)
    elif teacher == "musesfish":
        agent = MusesfishAgent(seed=seed)
    elif teacher == "musesfish_cpp":
        agent = MusesfishCppAgent(
            seed=seed,
            timeout=musesfish_cpp_timeout,
            min_depth=musesfish_cpp_min_depth,
            max_depth=musesfish_cpp_max_depth,
            fallback=False,
        )
    else:
        raise ValueError(f"Unknown teacher '{teacher}'")

    moves_info: list[dict] = []
    done = False
    while not done:
        player = env.current_player()
        obs = env.observation().copy()
        mask = env.legal_action_mask().copy()
        if hasattr(agent, "get_policy"):
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

    winner = None
    for mi in reversed(moves_info):
        if mi["terminated"] and mi["reward"] > 0:
            winner = mi["player"]
            break
    outcome = "draw" if winner is None else ("red" if winner == 0 else "black")

    samples: list[AZSample] = []
    for i, mi in enumerate(moves_info):
        value = 1.0 if (winner is not None and mi["player"] == winner) else (-1.0 if (winner is not None) else 0.0)
        samples.append(AZSample(
            observation=mi["obs"], legal_mask=mi["mask"],
            policy_target=mi["policy"], value_target=value,
            player=mi["player"], game_id=f"{teacher}_{game_idx}", move_index=i,
        ))
    return game_idx, samples, len(moves_info), outcome


def _dataset_stats(dataset: AZDataset) -> dict[str, float]:
    if len(dataset) == 0:
        return {"samples": 0, "value_mean": 0.0, "value_pos": 0.0, "value_neg": 0.0, "policy_nnz": 0.0}
    values = np.array([s.value_target for s in dataset.samples], dtype=np.float32)
    nnz = np.array([s.policy_nnz for s in dataset.samples], dtype=np.float32)
    return {
        "samples": float(len(dataset)),
        "value_mean": float(values.mean()),
        "value_pos": float((values > 0).mean()),
        "value_neg": float((values < 0).mean()),
        "policy_nnz": float(nnz.mean()),
    }


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
    workers: int = 1,
    musesfish_cpp_min_depth: int = 5,
    musesfish_cpp_max_depth: int = 6,
    musesfish_cpp_timeout: float = 5.0,
) -> AZDataset:
    """Generate supervised data using search self-play (parallel via multiprocessing)."""
    dataset = AZDataset()

    if workers <= 1:
        # --- Sequential path (original behaviour) ---
        t0 = time()
        for g in range(games):
            game_idx, samples, steps, outcome = _generate_one_game(
                (
                    g, max_steps, seed + g, teacher, simulations, max_depth, num_samples,
                    musesfish_cpp_min_depth, musesfish_cpp_max_depth, musesfish_cpp_timeout,
                ),
            )
            for s in samples:
                dataset.add(s)
            if progress_interval > 0 and ((g + 1) % progress_interval == 0 or g + 1 == games):
                stats = _dataset_stats(dataset)
                _log(
                    f"data game {g + 1}/{games} | steps={steps} outcome={outcome} "
                    f"samples={len(dataset)} value_mean={stats['value_mean']:+.3f} "
                    f"policy_nnz={stats['policy_nnz']:.1f} elapsed={time() - t0:.1f}s"
                )
        return dataset

    # --- Parallel path ---
    _log(f"data using workers={workers} games={games}")
    t0 = time()
    task_args = [
        (
            g, max_steps, seed + g, teacher, simulations, max_depth, num_samples,
            musesfish_cpp_min_depth, musesfish_cpp_max_depth, musesfish_cpp_timeout,
        )
        for g in range(games)
    ]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_generate_one_game, ta): ta[0] for ta in task_args}
        completed = 0
        for fut in as_completed(futures):
            game_idx, samples, steps, outcome = fut.result()
            for s in samples:
                dataset.add(s)
            completed += 1
            if progress_interval > 0 and completed % progress_interval == 0:
                stats = _dataset_stats(dataset)
                _log(
                    f"data completed {completed}/{games} | last_game={game_idx + 1} "
                    f"steps={steps} outcome={outcome} samples={len(dataset)} "
                    f"value_mean={stats['value_mean']:+.3f} policy_nnz={stats['policy_nnz']:.1f} "
                    f"elapsed={time() - t0:.1f}s"
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
    _log(
        f"train start | model={model_type} kwargs={model_kwargs} epochs={epochs} "
        f"batch_size={batch_size} lr={lr} device={device}"
    )
    model = create_model(model_type, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_policy_value(dataset, model, optimizer, device, epochs=epochs, batch_size=batch_size)
    _log("train done")
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
    t0 = time()
    mr = arena.run_match(policy_cfg, opp_cfg, n_games=n_games, max_steps=max_steps, seed=9999)
    return {
        "win_rate": round(mr.a_win_rate, 3),
        "draw_rate": round(mr.draw_rate, 3),
        "seconds": round(time() - t0, 2),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Search supervised pretraining")
    p.add_argument("--games", type=int, default=100, help="Games to generate")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--data", type=str, default=None, help="Use cached .npz dataset")
    p.add_argument("--cache-data", type=str, default=None, help="Save generated data")
    p.add_argument("--teacher", type=str, default="ismcts", choices=["ismcts", "belief_mcts", "musesfish", "musesfish_cpp"])
    p.add_argument("--simulations", type=int, default=100, help="ISMCTS simulations per move")
    p.add_argument("--max-depth", type=int, default=5, help="ISMCTS max search depth")
    p.add_argument("--teacher-samples", type=int, default=30, help="BeliefMCTS samples per move")
    p.add_argument("--musesfish-cpp-min-depth", type=int, default=5, help="C++ Musesfish minimum search depth")
    p.add_argument("--musesfish-cpp-max-depth", type=int, default=6, help="C++ Musesfish maximum search depth")
    p.add_argument("--musesfish-cpp-timeout", type=float, default=5.0, help="C++ Musesfish subprocess timeout per move")
    p.add_argument("--progress-interval", type=int, default=10, help="Print progress every N moves during data generation")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel workers for data generation")
    p.add_argument("--model", type=str, default="resnet")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-opponents", type=str, default="random,greedy",
                   help="Comma-separated eval opponents, e.g. random,greedy,musesfish,musesfish_cpp")
    p.add_argument("--checkpoint-out", type=str, default="pretrained.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    _log(
        f"pretrain start | teacher={args.teacher} games={args.games} max_steps={args.max_steps} "
        f"seed={args.seed} workers={args.workers} cache_data={args.cache_data or 'none'} "
        f"data={args.data or 'none'} checkpoint_out={args.checkpoint_out}"
    )
    if args.teacher == "musesfish_cpp":
        _log(
            f"musesfish_cpp config | min_depth={args.musesfish_cpp_min_depth} "
            f"max_depth={args.musesfish_cpp_max_depth} timeout={args.musesfish_cpp_timeout}"
        )
    _log(f"device resolved | device={device}")

    # Step 1: Data
    if args.data:
        _log(f"data load | path={args.data}")
        dataset = AZDataset()
        dataset.load(args.data)
    else:
        _log(f"data generation start | teacher={args.teacher}")
        dataset = generate_data(
            args.games,
            args.max_steps,
            args.seed,
            teacher=args.teacher,
            simulations=args.simulations,
            max_depth=args.max_depth,
            num_samples=args.teacher_samples,
            progress_interval=args.progress_interval,
            workers=args.workers,
            musesfish_cpp_min_depth=args.musesfish_cpp_min_depth,
            musesfish_cpp_max_depth=args.musesfish_cpp_max_depth,
            musesfish_cpp_timeout=args.musesfish_cpp_timeout,
        )
        if args.cache_data:
            dataset.save(args.cache_data)
            _log(f"data cached | path={args.cache_data}")
    stats = _dataset_stats(dataset)
    _log(
        f"dataset ready | samples={len(dataset)} value_mean={stats['value_mean']:+.3f} "
        f"value_pos={stats['value_pos']:.1%} value_neg={stats['value_neg']:.1%} "
        f"policy_nnz={stats['policy_nnz']:.1f}"
    )

    # Step 2: Train
    model_kwargs = {}
    if args.model == "resnet":
        model_kwargs = {"channels": args.channels, "num_blocks": args.blocks}
    model = train_supervised(dataset, args.model, model_kwargs, args.epochs, args.lr, args.batch_size, device)

    ckpt = {"model": model.state_dict(), "model_config": {"type": args.model, **model_kwargs}}
    torch.save(ckpt, args.checkpoint_out)
    _log(f"checkpoint saved | path={args.checkpoint_out}")

    # Step 3: Eval
    opponents = [o.strip() for o in args.eval_opponents.split(",") if o.strip()]
    _log(f"eval start | games={args.eval_games} opponents={opponents}")
    for opponent in opponents:
        result = quick_eval(args.checkpoint_out, opponent, args.eval_games, args.max_steps)
        _log(
            f"eval vs {opponent} | win={result['win_rate']:.1%} "
            f"draw={result['draw_rate']:.1%} seconds={result['seconds']:.2f}"
        )
    _log("pretrain done")


if __name__ == "__main__":
    main()
