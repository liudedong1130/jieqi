from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from jieqi.env import JieqiEnv
from jieqi.render import render as _render


# ---------------------------------------------------------------------------
#  Hash
# ---------------------------------------------------------------------------


def _hash_hidden(env: JieqiEnv) -> str:
    """SHA-256 of ``(pos, color, origin_type, true_type)`` for all hidden pieces."""
    h = hashlib.sha256()
    for pos, p in enumerate(env.board.cells):
        if p is None or p.revealed:
            continue
        h.update(f"{pos},{int(p.color)},{int(p.origin_type)},{int(p.true_type)}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
#  Save
# ---------------------------------------------------------------------------


def build_record(
    env: JieqiEnv,
    seed: int,
    moves_info: list[dict[str, Any]],
    debug: bool = False,
) -> dict[str, Any]:
    """Build a game-record dict from an environment and its move history.

    Parameters
    ----------
    env : JieqiEnv
        The environment *after* the game has ended.
    seed : int
        Seed used for ``env.reset()``.
    moves_info : list[dict]
        Each dict should have keys ``action``, ``player``, ``reward``,
        ``terminated``, ``truncated``.
    debug : bool
        If True, also store the initial hidden-piece ``(origin_type, true_type)``
        mapping — useful for local debugging but **not** for public records.
    """
    record: dict[str, Any] = {
        "seed": seed,
        "max_steps": env._max_steps,
        "initial_hash": _hash_hidden(env),
        "moves": [],
        "total_steps": len(moves_info),
    }

    # Determine winner
    for i, mi in enumerate(moves_info):
        record["moves"].append({
            "step": i,
            "action": mi["action"],
            "player": mi["player"],
            "reward": mi["reward"],
            "terminated": mi["terminated"],
            "truncated": mi["truncated"],
        })

    # Winner detection: last terminated move's player won
    winner = None
    for mi in reversed(moves_info):
        if mi.get("terminated"):
            winner = mi["player"]
            break
    record["winner"] = winner

    if debug:
        record["debug"] = {
            "initial_hidden": [
                {
                    "pos": pos,
                    "color": int(p.color),
                    "origin_type": int(p.origin_type),
                    "true_type": int(p.true_type),
                }
                for pos, p in enumerate(env.board.cells)
                if p is not None and not p.revealed
            ]
        }

    return record


def save_to_file(record: dict[str, Any], path: str) -> None:
    """Write *record* as JSON to *path*."""
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def load_from_file(path: str) -> dict[str, Any]:
    """Load a game record from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
#  Replay
# ---------------------------------------------------------------------------


def replay_game(
    record: dict[str, Any],
    step_by_step: bool = False,
    speed: float = 0.0,
) -> JieqiEnv:
    """Replay a game record step by step.

    Parameters
    ----------
    record : dict
        Game record (from ``build_record`` or ``load_from_file``).
    step_by_step : bool
        If True, print the board after each move and wait for Enter.
    speed : float
        Delay in seconds between moves (used when *step_by_step* is False).

    Returns
    -------
    JieqiEnv
        The environment after replaying all moves.
    """
    env = JieqiEnv(max_steps=record["max_steps"])
    env.reset(seed=record["seed"])

    for mi in record["moves"]:
        action = mi["action"]
        player_before = env.current_player()

        if player_before != mi["player"]:
            print(f"  [WARNING] step {mi['step']}: player mismatch "
                  f"(record={mi['player']}, env={player_before})")

        print(f"\n--- Step {mi['step']} | Player {player_before} | action={action} "
              f"({action // 90},{action % 90}) ---")

        obs, reward, terminated, truncated, _info = env.step(action)
        print(_render(env.board))

        if step_by_step:
            input("  Press Enter for next move...")
        elif speed > 0:
            time.sleep(speed)

    return env
