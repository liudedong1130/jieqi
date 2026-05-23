#!/usr/bin/env python3
"""Replay a saved Jieqi game record step by step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jieqi.record import load_from_file, replay_game


def main() -> None:
    p = argparse.ArgumentParser(description="Replay a Jieqi game record")
    p.add_argument("record", help="Path to record JSON file")
    p.add_argument("--step-by-step", action="store_true", help="Wait for Enter between moves")
    p.add_argument("--speed", type=float, default=0.5, help="Delay in seconds between moves")
    args = p.parse_args()

    record = load_from_file(args.record)
    print(f"Seed: {record['seed']}")
    print(f"Moves: {record['total_steps']}")
    print(f"Winner: {record.get('winner', 'draw')}")

    replay_game(record, step_by_step=args.step_by_step, speed=args.speed)

    print("\nReplay complete.")


if __name__ == "__main__":
    main()
