from __future__ import annotations

import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np

from agents.musesfish_agent import MusesfishAgent
from jieqi.constants import Color, PieceType
from jieqi.env import JieqiEnv
from jieqi.move import encode_action, pos_to_rc, rc_to_pos

# C++ Musesfish wrapper.  Source vendored from miaosiSari/Jieqi (GPL v3).

_REVEALED_CHAR = {
    PieceType.KING: "K",
    PieceType.ADVISOR: "A",
    PieceType.ELEPHANT: "B",
    PieceType.HORSE: "N",
    PieceType.ROOK: "R",
    PieceType.CANNON: "C",
    PieceType.PAWN: "P",
}

_HIDDEN_CHAR = {
    PieceType.ROOK: "D",
    PieceType.HORSE: "E",
    PieceType.ELEPHANT: "F",
    PieceType.ADVISOR: "G",
    PieceType.CANNON: "H",
    PieceType.PAWN: "I",
}

_POOL_KEY = {
    PieceType.ROOK: "R",
    PieceType.HORSE: "N",
    PieceType.ELEPHANT: "B",
    PieceType.ADVISOR: "A",
    PieceType.CANNON: "C",
    PieceType.PAWN: "P",
}


def _engine_index(pos: int) -> int:
    row, col = pos_to_rc(pos)
    return (row + 3) * 16 + (col + 3)


def _remaining_pool(env: JieqiEnv, color: Color) -> dict[str, int]:
    counts = {"R": 2, "N": 2, "B": 2, "A": 2, "C": 2, "P": 5}
    for piece in env.board.captured:
        if piece.color != color or not piece.revealed or piece.true_type == PieceType.KING:
            continue
        key = _POOL_KEY[piece.true_type]
        counts[key] = max(0, counts[key] - 1)
    for piece in env.board.cells:
        if piece is None or piece.color != color or not piece.revealed or piece.true_type == PieceType.KING:
            continue
        key = _POOL_KEY[piece.true_type]
        counts[key] = max(0, counts[key] - 1)
    return counts


def _state_rows(env: JieqiEnv) -> list[str]:
    chars = [" "] * 256
    for row in range(16):
        for col in range(16):
            idx = row * 16 + col
            if 3 <= row <= 12 and 3 <= col <= 11:
                chars[idx] = "."
            else:
                chars[idx] = " "
    for pos, piece in enumerate(env.board.cells):
        if piece is None:
            continue
        ch = _REVEALED_CHAR[piece.true_type] if piece.revealed else _HIDDEN_CHAR[piece.origin_type]
        if piece.color == Color.BLACK:
            ch = ch.lower()
        chars[_engine_index(pos)] = ch
    return ["".join(chars[row * 16:(row + 1) * 16]) for row in range(16)]


def _ucci_to_action(ucci: str, player: int = int(Color.RED)) -> int | None:
    if len(ucci) < 4:
        return None
    try:
        fc = ord(ucci[0]) - ord("a")
        fr_rank = int(ucci[1])
        tc = ord(ucci[2]) - ord("a")
        tr_rank = int(ucci[3])
    except ValueError:
        return None
    if not (0 <= fr_rank < 10 and 0 <= fc < 9 and 0 <= tr_rank < 10 and 0 <= tc < 9):
        return None
    if player == int(Color.BLACK):
        fr, fc = fr_rank, 8 - fc
        tr, tc = tr_rank, 8 - tc
    else:
        fr, fc = 9 - fr_rank, fc
        tr, tc = 9 - tr_rank, tc
    return encode_action(rc_to_pos(fr, fc), rc_to_pos(tr, tc))


class MusesfishCppAgent:
    """Agent backed by the vendored C++ Musesfish query binary."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        binary_path: str | None = None,
        timeout: float = 3.0,
        min_depth: int = 5,
        max_depth: int = 6,
        persistent: bool = True,
        fallback: bool = True,
    ) -> None:
        self.timeout = timeout
        self.min_depth = min_depth
        self.max_depth = max(min_depth, max_depth)
        self.persistent = persistent
        root = Path(__file__).resolve().parent.parent
        self.binary_path = Path(binary_path) if binary_path else (
            root / "agents" / "vendor" / "musesfish_cpp" / "build" / "musesfish_query"
        )
        self.score_file = root / "agents" / "vendor" / "musesfish_cpp" / "score.conf"
        self._fallback = MusesfishAgent(seed=seed, think_time=min(timeout, 1.0)) if fallback else None
        self._worker: subprocess.Popen[bytes] | None = None
        self._stdout_buffer = b""

    def select_action(self, env: JieqiEnv) -> int:
        action = self._select_action_cpp(env)
        if action in env.legal_actions():
            return int(action)
        if self._fallback is not None:
            return self._fallback.select_action(env)
        legal = env.legal_actions()
        return legal[0] if legal else 0

    def get_policy(self, env: JieqiEnv) -> tuple[np.ndarray, int]:
        action = self.select_action(env)
        policy = np.zeros(8100, dtype=np.float32)
        if action in env.legal_actions():
            policy[action] = 1.0
        return policy, action

    def _select_action_cpp(self, env: JieqiEnv) -> int | None:
        if not self.binary_path.exists():
            return None
        player = env.current_player()
        red = _remaining_pool(env, Color.RED)
        black = _remaining_pool(env, Color.BLACK)
        header = [
            f"{1 if player == int(Color.RED) else 0} 1 0",
            " ".join(f"{red[k]} {black[k]}" for k in "RNBACP"),
        ]
        payload = "\n".join(header + _state_rows(env)) + "\n"
        if self.persistent:
            return self._select_action_persistent(payload, player)
        return self._select_action_oneshot(payload, player)

    def _command(self, *, loop: bool) -> list[str]:
        cmd = [str(self.binary_path), str(self.score_file), str(self.min_depth), str(self.max_depth)]
        if loop:
            cmd.append("--loop")
        return cmd

    def _select_action_oneshot(self, payload: str, player: int) -> int | None:
        try:
            proc = subprocess.run(
                self._command(loop=False),
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        return _ucci_to_action(lines[-1], player)

    def _select_action_persistent(self, payload: str, player: int) -> int | None:
        proc = self._ensure_worker()
        if proc is None or proc.stdin is None or proc.stdout is None:
            return None
        try:
            proc.stdin.write(payload.encode("utf-8"))
            proc.stdin.flush()
            move = self._read_worker_move(proc.stdout.fileno())
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            return None
        if move is None:
            self.close()
            return None
        return _ucci_to_action(move, player)

    def _ensure_worker(self) -> subprocess.Popen[bytes] | None:
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        if not self.binary_path.exists():
            return None
        try:
            self._worker = subprocess.Popen(
                self._command(loop=True),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._stdout_buffer = b""
        except OSError:
            self._worker = None
        return self._worker

    def _read_worker_move(self, fd: int) -> str | None:
        deadline = time.monotonic() + self.timeout
        while True:
            while b"\n" in self._stdout_buffer:
                line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith("__MOVE__ "):
                    return text[len("__MOVE__ "):].strip()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(fd, 4096)
            if not chunk:
                return None
            self._stdout_buffer += chunk

    def close(self) -> None:
        proc = self._worker
        self._worker = None
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass

    def __del__(self) -> None:
        self.close()
