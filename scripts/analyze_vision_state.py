#!/usr/bin/env python3
"""Analyze a vision state JSON and output legal moves + recommendations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.belief_mcts_agent import BeliefMCTSAgent
from agents.greedy_agent import GreedyAgent
from rl.ismcts import ISMCTSAgent
from vision.adapter import VisionBoardState, validate_vision_state, vision_state_to_game_state
from jieqi.env import JieqiEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="vision_state.json file")
    p.add_argument("--agent", type=str, default="ismcts", help="Agent for recommendations")
    p.add_argument("--top-k", type=int, default=3, help="Top K recommendations")
    p.add_argument("--sims", type=int, default=100)
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    state = VisionBoardState.from_dict(data)
    errs = validate_vision_state(state)
    if errs:
        print("ERRORS:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)

    env = JieqiEnv(max_steps=200)
    env.reset()
    vision_state_to_game_state(state, env)

    # Legal moves
    legal = env.legal_actions()
    print(f"Player: {'RED' if state.current_player == 0 else 'BLACK'}")
    print(f"Legal moves: {len(legal)}")

    # Top-K recommendations
    if args.agent == "ismcts":
        agent = ISMCTSAgent(num_simulations=args.sims, max_depth=6, temperature=1.0, seed=0)
    elif args.agent == "belief_mcts":
        agent = BeliefMCTSAgent(num_samples=20, seed=0)
    else:
        agent = GreedyAgent(seed=0)

    if hasattr(agent, "get_policy"):
        policy, _ = agent.get_policy(env)
        scored = [(a, float(policy[a])) for a in legal]
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        # Fallback: just run select_action and list top legal moves
        scored = [(a, 0.0) for a in legal[:args.top_k]]

    print(f"\nTop {args.top_k} recommendations ({args.agent}):")
    for rank, (action, score) in enumerate(scored[:args.top_k], 1):
        fpos, tpos = action // 90, action % 90
        print(f"  {rank}. {fpos//9},{fpos%9}→{tpos//9},{tpos%9}  score={score:.4f}  action={action}")


if __name__ == "__main__":
    main()
