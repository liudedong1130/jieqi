from __future__ import annotations

from dataclasses import dataclass

from .constants import Color, PieceType


@dataclass(frozen=True)
class Piece:
    """A Jieqi piece with hidden/revealed duality.

    - `true_type`: actual piece type, known only to the environment while hidden.
    - `origin_type`: type determined by the starting position the piece occupies.
    - `effective_type`: returns `origin_type` while hidden, `true_type` once revealed.
      This is the ONLY accessor that should be used when building observations.
    """

    color: Color
    origin_type: PieceType
    true_type: PieceType
    revealed: bool = False

    @property
    def effective_type(self) -> PieceType:
        return self.true_type if self.revealed else self.origin_type

    @property
    def is_king(self) -> bool:
        return self.true_type == PieceType.KING
