from __future__ import annotations

from typing import Any

import numpy as np

from .board import Board
from .constants import NUM_ACTIONS
from .encoder import encode_observation
from .move import encode_action, decode_action, Move
from .render import render as _render
from .rules import generate_legal_moves


class JieqiEnv:
    """Gymnasium-style RL environment for Jieqi (揭棋).

    Parameters
    ----------
    max_steps : int
        Maximum moves per episode before truncation (default 500).

    Observation space
    ----------------
    ``(28, 10, 9)`` float32 tensor.  See ``jieqi.encoder`` for the
    channel layout.  Hidden pieces are represented by ``origin_type``
    only — ``true_type`` is **never** exposed.

    Action space
    ------------
    Discrete 8100: ``from_pos * 90 + to_pos`` (see ``jieqi.move``).
    """

    def __init__(self, max_steps: int = 500) -> None:
        self._board = Board()
        self._max_steps = max_steps
        self._steps = 0

    # ---- lifecycle ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the board to the starting position and return the observation tensor."""
        self._board.reset(seed=seed)
        self._steps = 0
        return self.observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute *action* and return ``(obs, reward, terminated, truncated, info)``.

        Raises
        ------
        ValueError
            If *action* is not in the current legal-action set.
        """
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(
                f"Illegal action {action} (from={action // 90}, to={action % 90})"
            )

        from_pos, to_pos = decode_action(action)
        move = Move(from_pos, to_pos)

        mover = self._board.turn  # player who is about to move
        self._board.apply_move(move)
        self._steps += 1

        # --- terminal checks (from mover's perspective) ---
        terminated = False
        truncated = False
        reward = 0.0

        opponent = self._board.turn  # turn was already swapped
        opp_legal = generate_legal_moves(self._board, opponent)

        # King missing or opponent has no legal moves → mover wins
        try:
            self._board.king_pos(opponent)
        except ValueError:
            terminated = True
            reward = 1.0

        if not terminated and len(opp_legal) == 0:
            terminated = True
            reward = 1.0

        if not terminated and self._steps >= self._max_steps:
            truncated = True

        info: dict[str, Any] = {}
        return self.observation(), reward, terminated, truncated, info

    # ---- query methods -----------------------------------------------------

    def legal_actions(self) -> list[int]:
        """Return all legal actions for the current player."""
        moves = generate_legal_moves(self._board, self._board.turn)
        return [encode_action(m.from_pos, m.to_pos) for m in moves]

    def legal_action_mask(self) -> np.ndarray:
        """Return a boolean mask of shape ``(8100,)`` for legal actions."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        for a in self.legal_actions():
            mask[a] = 1
        return mask

    def current_player(self) -> int:
        """Return the player to move: 0 = RED, 1 = BLACK."""
        return int(self._board.turn)

    def observation(self) -> np.ndarray:
        """Return the (28, 10, 9) encoded observation tensor."""
        return encode_observation(self._board)

    def render(self) -> str:
        """Return an ASCII representation of the board."""
        return _render(self._board)

    # ---- accessors ---------------------------------------------------------

    @property
    def board(self) -> Board:
        """Expose the internal Board (for debugging / testing)."""
        return self._board

    @property
    def steps(self) -> int:
        return self._steps
