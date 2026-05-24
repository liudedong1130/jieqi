#!/usr/bin/env python3
"""Simple UCCI-like CLI engine for Jieqi."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class JieqiEngine:
    """Minimal engine supporting ucci / isready / position / go / quit."""

    def __init__(self, agent_type: str = "ismcts", checkpoint: str | None = None) -> None:
        self._agent_type = agent_type
        self._checkpoint = checkpoint
        self._env = None
        self._agent = None
        print("JieqiEngine v0.1")

    def _ensure_env(self) -> None:
        from jieqi.env import JieqiEnv
        if self._env is None:
            self._env = JieqiEnv()
            self._env.reset()

    def _make_agent(self) -> Any:
        if self._agent_type == "random":
            from agents.random_agent import RandomAgent
            return RandomAgent(seed=0)
        elif self._agent_type == "greedy":
            from agents.greedy_agent import GreedyAgent
            return GreedyAgent(seed=0)
        elif self._agent_type == "ismcts":
            from rl.ismcts import ISMCTSAgent
            return ISMCTSAgent(num_simulations=100, max_depth=8, temperature=0.1, seed=0)
        elif self._agent_type == "policy":
            from agents.policy_agent import PolicyAgent
            if self._checkpoint is None:
                raise ValueError("checkpoint required for policy agent")
            return PolicyAgent(self._checkpoint, deterministic=True)
        else:
            raise ValueError(f"Unknown agent: {self._agent_type}")

    def handle(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None

        parts = line.split()
        cmd = parts[0]

        if cmd == "ucci":
            return "id name JieqiEngine\nid author JieqiRL\nucciok"
        elif cmd == "isready":
            return "readyok"
        elif cmd == "position":
            return self._cmd_position(parts[1:])
        elif cmd == "go":
            return self._cmd_go(parts[1:])
        elif cmd == "quit":
            sys.exit(0)
        elif cmd == "fen":
            self._ensure_env()
            from engine.jieqi_fen import export_jieqi_fen
            return export_jieqi_fen(self._env.board, self._env)
        else:
            return None

    def _cmd_position(self, args: list[str]) -> str | None:
        self._ensure_env()
        if args and args[0] == "startpos":
            self._env.reset()
            return None
        if args and args[0] == "fen":
            fen = " ".join(args[1:])
            from engine.jieqi_fen import parse_jieqi_fen
            state = parse_jieqi_fen(fen)
            self._env.board._cells = [None] * 90
            from jieqi import Piece, Color, PieceType
            for p in state["pieces"]:
                pt = PieceType(p["type"])
                color = Color(p["color"])
                piece = Piece(color, pt, pt, p["revealed"])
                self._env.board.set_cell(p["pos"], piece)
            self._env.board._turn = Color(state["current_player"])
            return None
        return None

    def _cmd_go(self, args: list[str]) -> str | None:
        self._ensure_env()
        movetime_ms = 1000
        for i, a in enumerate(args):
            if a == "movetime" and i + 1 < len(args):
                movetime_ms = int(args[i + 1])

        if self._agent is None:
            self._agent = self._make_agent()

        deadline = time.perf_counter() + movetime_ms / 1000.0
        action = self._agent.select_action(self._env)
        elapsed = (time.perf_counter() - (deadline - movetime_ms / 1000.0)) * 1000

        from_pos = action // 90
        to_pos = action % 90
        fr, fc = from_pos // 9, from_pos % 9
        tr, tc = to_pos // 9, to_pos % 9

        self._env.step(action)
        print(f"info elapsed {elapsed:.0f}ms", file=sys.stderr)
        return f"bestmove {from_pos}{to_pos:04d}  # ({fr},{fc})→({tr},{tc})"


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agent", type=str, default="ismcts")
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()

    engine = JieqiEngine(agent_type=args.agent, checkpoint=args.checkpoint)

    for line in sys.stdin:
        resp = engine.handle(line)
        if resp:
            print(resp, flush=True)


if __name__ == "__main__":
    main()
