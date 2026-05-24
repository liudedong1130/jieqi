#!/usr/bin/env python3
"""Generate top-k move recommendations with explanations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.recommendation import (
    generate_recommendations,
    recommendations_to_json,
    recommendations_to_text,
)
from jieqi.env import JieqiEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fen", type=str, default=None)
    p.add_argument("--vision", type=str, default=None)
    p.add_argument("--agent", type=str, default="ismcts", choices=["ismcts", "belief_mcts", "greedy"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = JieqiEnv(max_steps=200)

    if args.fen:
        from engine.jieqi_fen import parse_jieqi_fen
        state = parse_jieqi_fen(args.fen)
        from vision.adapter import VisionBoardState, vision_state_to_game_state
        vs = VisionBoardState(
            cells=[{"row": p["pos"]//9, "col": p["pos"]%9,
                    "state": ("red_open" if p["revealed"] else "red_hidden") if p["color"]==0
                    else ("black_open" if p["revealed"] else "black_hidden"),
                    "piece_type": p["type"]}
                   for p in state["pieces"]],
            current_player=state["current_player"],
        )
        vision_state_to_game_state(vs, env)
    elif args.vision:
        with open(args.vision) as f:
            data = json.load(f)
        from vision.adapter import VisionBoardState, vision_state_to_game_state
        vs = VisionBoardState.from_dict(data)
        vision_state_to_game_state(vs, env)
    else:
        env.reset(seed=args.seed)

    recs = generate_recommendations(env, agent_type=args.agent, top_k=args.top_k)

    print(recommendations_to_text(recs))

    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(recommendations_to_json(recs))
        print(f"\nJSON → {args.output_json}")


if __name__ == "__main__":
    main()
