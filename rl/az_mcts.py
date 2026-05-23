from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MCTSNode:
    """Skeleton node for AlphaZero MCTS."""

    __slots__ = ("visit_count", "total_value", "prior", "children", "state")

    def __init__(self, prior: float = 0.0) -> None:
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.children: dict[int, MCTSNode] = {}
        self.state: Any = None

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class AZMCTS(ABC):
    """Abstract interface for AlphaZero MCTS.

    Subclasses must implement ``search`` and ``get_policy``.
    """

    @abstractmethod
    def search(self, observation: np.ndarray, legal_mask: np.ndarray) -> None:
        """Run MCTS simulations from the given state."""
        ...

    @abstractmethod
    def get_policy(self, temperature: float = 1.0) -> np.ndarray:
        """Return the improved policy from visit counts.

        Returns a ``(8100,)`` array summing to 1.
        """
        ...
