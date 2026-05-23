#!/usr/bin/env python3
"""Environment consistency stress-test for Jieqi.

Runs many RandomAgent-vs-RandomAgent games and calls
``assert_state_consistency`` after every move to catch hidden-state bugs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from jieqi import BOARD_SIZE, Color, PieceType, encode_action, decode_action
from jieqi.env import JieqiEnv
from jieqi.move import Move
from jieqi.render import render


# =============================================================================
#  Consistency checks
# =============================================================================


def assert_state_consistency(env: JieqiEnv) -> list[str]:
    """Run all consistency checks; return a list of failure descriptions."""
    failures: list[str] = []
    b = env.board
    turn = env.current_player()

    # 1. Board size
    if len(b.cells) != BOARD_SIZE:
        failures.append(f"board.cells has {len(b.cells)} elements, expected {BOARD_SIZE}")

    # 2. One piece per cell
    non_empty = sum(1 for p in b.cells if p is not None)
    if non_empty > 32:
        failures.append(f"more than 32 pieces on board: {non_empty}")

    # 3. hidden_pieces all have revealed=False
    for pos, p in b.hidden_pieces():
        if p.revealed:
            failures.append(f"hidden_piece at {pos} has revealed=True")

    # 4. revealed_pieces all have revealed=True
    for pos, p in b.revealed_pieces():
        if not p.revealed:
            failures.append(f"revealed_piece at {pos} has revealed=False")

    # 5. Captured pieces not on board
    captured_ids = {id(p) for p in b.captured}
    for pos, p in enumerate(b.cells):
        if p is not None and id(p) in captured_ids:
            failures.append(f"captured piece still on board at {pos}")

    # 6. Piece counts per side
    for color in (Color.RED, Color.BLACK):
        cnt = len(b.pieces_of(color))
        if cnt > 16:
            failures.append(f"{color.name} has {cnt} pieces (>16)")

    # 7. Total piece accounting
    on_board = sum(1 for p in b.cells if p is not None)
    total = on_board + len(b.captured)
    if total != 32:
        failures.append(f"total pieces={total} (on_board={on_board} + captured={len(b.captured)}), expected 32")

    # 8. current_player valid
    if turn not in (0, 1):
        failures.append(f"current_player={turn}, expected 0 or 1")

    # 9. Kings always revealed
    for color in (Color.RED, Color.BLACK):
        try:
            kp = b.king_pos(color)
            king = b[kp]
            if king is None or not king.revealed:
                failures.append(f"{color.name} king at {kp} is not revealed")
        except ValueError:
            failures.append(f"{color.name} king not found on board")

    # 10. King is never hidden
    for pos, p in enumerate(b.cells):
        if p is not None and p.is_king and not p.revealed:
            failures.append(f"hidden king at {pos}")

    # 11. legal_actions encode/decode roundtrip
    for a in env.legal_actions():
        f, t = decode_action(a)
        rt = encode_action(f, t)
        if rt != a:
            failures.append(f"action roundtrip mismatch: {a} → ({f},{t}) → {rt}")

    # 12. legal_actions from_pos belongs to current player
    cur_color = Color(turn)
    for a in env.legal_actions():
        f, t = decode_action(a)
        piece = b[f]
        if piece is None:
            failures.append(f"legal action from empty cell: from={f} to={t}")
        elif piece.color != cur_color:
            failures.append(f"legal action from opponent piece: from={f} color={piece.color.name} expected {cur_color.name}")

    return failures


# =============================================================================
#  Stress runner
# =============================================================================


def run_stress_test(
    n_games: int = 100,
    max_steps: int = 300,
    seed: int = 0,
    verbose: bool = False,
    save_records: bool = True,
    record_dir: str = "stress_records",
) -> dict[str, Any]:
    """Run *n_games* random self-play games with consistency checks.

    Returns a dict with keys: passed, failed, failures_detail, total_steps.
    """
    passed = 0
    failed = 0
    failures_detail: list[dict] = []
    total_steps = 0

    if save_records:
        os.makedirs(record_dir, exist_ok=True)

    for g in range(n_games):
        game_seed = seed + g
        env = JieqiEnv(max_steps=max_steps)
        env.reset(seed=game_seed)
        agent_red = RandomAgent(seed=seed + g)
        agent_black = RandomAgent(seed=seed + g + 10000)
        move_history: list[int] = []
        moves_info: list[dict] = []

        game_failed = False
        steps = 0

        while not game_failed:
            # Pre-move consistency
            errs = assert_state_consistency(env)
            if errs:
                failed += 1
                detail = {
                    "game": g,
                    "seed": game_seed,
                    "step": steps,
                    "errors": errs,
                    "board": render(env.board),
                    "move_history": list(move_history),
                    "captured_count": len(env.board.captured),
                }
                failures_detail.append(detail)
                _save_failure_record(env, game_seed, moves_info, record_dir, g, save_records)
                game_failed = True
                break

            actions = env.legal_actions()
            if not actions:
                break

            player = env.current_player()
            if env.current_player() == 0:
                action = agent_red.select_action(env)
            else:
                action = agent_black.select_action(env)

            # Verify step doesn't crash
            try:
                obs, reward, terminated, truncated, _info = env.step(action)
            except Exception as exc:
                failed += 1
                detail = {
                    "game": g,
                    "seed": game_seed,
                    "step": steps,
                    "error": str(exc),
                    "action": action,
                    "board": render(env.board),
                    "move_history": list(move_history),
                }
                failures_detail.append(detail)
                _save_failure_record(env, game_seed, moves_info, record_dir, g, save_records)
                game_failed = True
                break

            moves_info.append({
                "action": action, "player": player,
                "reward": float(reward), "terminated": terminated, "truncated": truncated,
            })
            move_history.append(action)
            steps += 1

            if steps >= max_steps:
                break

        if not game_failed:
            passed += 1
        total_steps += steps

        if verbose and (g + 1) % 100 == 0:
            print(f"  [{g + 1}/{n_games}] passed={passed} failed={failed}")

    return {
        "passed": passed,
        "failed": failed,
        "failures_detail": failures_detail,
        "total_steps": total_steps,
    }


def _save_failure_record(
    env: JieqiEnv,
    seed: int,
    moves_info: list[dict],
    record_dir: str,
    game_idx: int,
    save: bool,
) -> None:
    if not save:
        return
    from jieqi.record import build_record, save_to_file

    record = build_record(env, seed, moves_info, debug=True)
    path = os.path.join(record_dir, f"failure_game_{game_idx}_seed_{seed}.json")
    save_to_file(record, path)
    print(f"  Failure record saved to {path}")


# =============================================================================
#  CLI
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description="Jieqi env stress test")
    p.add_argument("--games", type=int, default=1000, help="Number of games")
    p.add_argument("--max-steps", type=int, default=300, help="Max steps per game")
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    p.add_argument("--verbose", action="store_true", help="Progress output")
    args = p.parse_args()

    print(f"Running {args.games} games (max {args.max_steps} steps each) ...")
    result = run_stress_test(
        n_games=args.games,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(f"\nResults:")
    print(f"  Passed: {result['passed']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Total steps: {result['total_steps']}")

    if result["failed"] > 0:
        print(f"\nFailure details:")
        for i, fd in enumerate(result["failures_detail"][:5]):
            print(f"\n  --- Failure {i + 1} ---")
            print(f"  Game: {fd.get('game')}, Seed: {fd.get('seed')}, Step: {fd.get('step')}")
            if "action" in fd:
                print(f"  Action: {fd['action']}")
            if "errors" in fd:
                for e in fd["errors"]:
                    print(f"  Error: {e}")
            if "error" in fd:
                print(f"  Exception: {fd['error']}")
            print(f"  Captured: {fd.get('captured_count', 'N/A')}")
            print(f"  Board:\n{fd.get('board', 'N/A')}")
            print(f"  Move history: {fd.get('move_history', [])[:20]}...")


if __name__ == "__main__":
    main()
