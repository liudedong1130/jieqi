from __future__ import annotations

import contextlib
import io
import time

from agents.vendor.musesfish import engine
from jieqi.constants import Color, PieceType
from jieqi.env import JieqiEnv
from jieqi.move import decode_action, encode_action, pos_to_rc, rc_to_pos

# Vendored from miaosiSari/Jieqi (GPL v3), adapted to this project's Env/Move API.
# The adapter only uses public hidden-piece origin types and revealed/captured true
# types; it must not inspect hidden true_type values.

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


def _project_index(idx: int) -> int | None:
    row = idx // 16 - 3
    col = idx % 16 - 3
    if 0 <= row < 10 and 0 <= col < 9:
        return rc_to_pos(row, col)
    return None


def _empty_engine_board() -> list[str]:
    chars = [" "] * 256
    for row in range(15):
        chars[row * 16 + 15] = "\n"
    for row in range(3, 13):
        for col in range(3, 12):
            chars[row * 16 + col] = "."
    return chars


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


def _configure_engine_pools(env: JieqiEnv) -> None:
    red = _remaining_pool(env, Color.RED)
    black = {k.lower(): v for k, v in _remaining_pool(env, Color.BLACK).items()}
    engine.r = dict(red)
    engine.b = dict(black)
    engine.di = {0: {True: dict(red), False: dict(black)}}
    engine.sumall = {
        0: {
            True: sum(red.values()),
            False: sum(black.values()),
        }
    }
    engine.average = {0: {}}


def env_to_engine_position(env: JieqiEnv) -> tuple[object, bool]:
    """Convert env into a Musesfish Position and return (position, rotated).

    ``rotated`` is true when black is to move; returned engine moves must then be
    mapped back with ``254 - index``.
    """
    _configure_engine_pools(env)
    board = _empty_engine_board()
    for pos, piece in enumerate(env.board.cells):
        if piece is None:
            continue
        if piece.revealed:
            ch = _REVEALED_CHAR[piece.true_type]
        else:
            ch = _HIDDEN_CHAR[piece.origin_type]
        if piece.color == Color.BLACK:
            ch = ch.lower()
        board[_engine_index(pos)] = ch

    board_str = "".join(board)
    searcher = engine.Searcher()
    searcher.calc_average()
    red_pos = engine.Position(board_str, 0, True, 0).set()
    red_pos = engine.Position(board_str, red_pos.score_rough, True, 0).set()
    if env.current_player() == int(Color.RED):
        return red_pos, False
    return red_pos.rotate(), True


def engine_move_to_action(move: tuple[int, int], rotated: bool) -> int | None:
    src, dst = move
    if rotated:
        src, dst = 254 - src, 254 - dst
    from_pos = _project_index(src)
    to_pos = _project_index(dst)
    if from_pos is None or to_pos is None:
        return None
    return encode_action(from_pos, to_pos)


class OriginalMusesfishSearch:
    """Thin wrapper around the original Python PVS Musesfish searcher."""

    def __init__(self, *, think_time: float = 1.0, min_depth: int = 4, max_depth: int = 5) -> None:
        self.think_time = think_time
        self.min_depth = min_depth
        self.max_depth = max_depth

    def select_action(self, env: JieqiEnv) -> int | None:
        pos, rotated = env_to_engine_position(env)
        searcher = engine.Searcher()
        move = None
        start = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            old_min, old_max = engine.SEARCH_MIN_DEPTH, engine.SEARCH_MAX_DEPTH
            engine.SEARCH_MIN_DEPTH = self.min_depth
            engine.SEARCH_MAX_DEPTH = self.max_depth
            engine.generate_forbiddenmoves(pos, check_bozi=True, step=0)
            try:
                for _depth, candidate, _score in searcher.search(pos, (pos,)):
                    if candidate is not None:
                        move = candidate
                    if time.time() - start >= self.think_time:
                        break
            finally:
                engine.SEARCH_MIN_DEPTH, engine.SEARCH_MAX_DEPTH = old_min, old_max
        if move is None:
            return None
        return engine_move_to_action(move, rotated)
