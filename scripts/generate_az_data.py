#!/usr/bin/env python3
"""Generate AlphaZero-style training data from agent self-play."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agents.belief_mcts_agent import BeliefMCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from jieqi.env import JieqiEnv
from rl.az_data import AZSample, AZDataset

AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "belief_mcts": BeliefMCTSAgent,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate AZ training data")
    p.add_argument("--agent", type=str, default="random", help="Agent type for both sides")
    p.add_argument("--agent-red", type=str, default=None, help="Red agent (overrides --agent)")
    p.add_argument("--agent-black", type=str, default=None, help="Black agent (overrides --agent)")
    p.add_argument("--games", type=int, default=100, help="Number of games")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="az_data.npz")
    args = p.parse_args()

    red_name = args.agent_red or args.agent
    black_name = args.agent_black or args.agent

    dataset = AZDataset()
    total_moves = 0

    for g in range(args.games):
        env = JieqiEnv(max_steps=args.max_steps)
        env.reset(seed=args.seed + g)
        red = AGENTS[red_name](seed=args.seed + g * 2)
        black = AGENTS[black_name](seed=args.seed + g * 2 + 1)

        moves_info: list[dict] = []  # [(action, player, obs, mask)]
        done = False
        while not done:
            player = env.current_player()
            obs = env.observation().copy()
            mask = env.legal_action_mask().copy()

            agent = red if player == 0 else black
            action = agent.select_action(env)
            _obs, reward, terminated, truncated, _info = env.step(action)

            moves_info.append({
                "action": action, "player": player, "obs": obs, "mask": mask,
                "reward": reward, "terminated": terminated,
            })
            done = terminated or truncated

        # Determine game outcome from each player's perspective
        final_reward = 0.0
        for mi in reversed(moves_info):
            if mi["terminated"] and mi["reward"] > 0:
                final_reward = 1.0
                break

        for i, mi in enumerate(moves_info):
            policy = np.zeros(8100, dtype=np.float32)
            policy[mi["action"]] = 1.0

            # Value target from this player's perspective
            if final_reward > 0:
                # Winner's last move gives +1. Winner = mi["player"] if that player made the last winning move
                winner = moves_info[-1]["player"]
                value = 1.0 if mi["player"] == winner else -1.0
            else:
                value = 0.0

            dataset.add(AZSample(
                observation=mi["obs"],
                legal_mask=mi["mask"],
                policy_target=policy,
                value_target=value,
                player=mi["player"],
                game_id=f"game_{g}",
                move_index=i,
            ))
            total_moves += 1

    dataset.save(args.output)
    print(f"Saved {len(dataset)} samples ({args.games} games, {total_moves} moves) → {args.output}")


if __name__ == "__main__":
    main()
