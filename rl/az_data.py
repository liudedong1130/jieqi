from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import torch


@dataclass
class AZSample:
    """A single training sample for AlphaZero-style policy+value learning.

    Attributes
    ----------
    observation : np.ndarray
        ``(28, 10, 9)`` encoded board from the current player's perspective.
    legal_mask : np.ndarray
        ``(8100,)`` mask of legal actions at the time of the move.
    policy_target : np.ndarray
        ``(8100,)`` target policy distribution (sum = 1).
        In v1 this is a one-hot encoding of the chosen action.
    value_target : float
        Final game outcome from the current player's perspective.
        +1 = win, -1 = loss, 0 = draw.
    player : int
        0 = RED, 1 = BLACK.
    game_id : str
        Unique game identifier.
    move_index : int
        Index of this move within the game.
    """

    observation: np.ndarray
    legal_mask: np.ndarray
    policy_target: np.ndarray = field(default_factory=lambda: np.zeros(8100, dtype=np.float32))
    value_target: float = 0.0
    player: int = 0
    game_id: str = ""
    move_index: int = 0


class AZDataset:
    """Container for AlphaZero training samples with save/load support."""

    def __init__(self) -> None:
        self.samples: list[AZSample] = []

    def add(self, sample: AZSample) -> None:
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def extend(self, other: AZDataset) -> None:
        self.samples.extend(other.samples)

    def trim_to_recent(self, max_samples: int) -> None:
        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]

    def save(self, path: str) -> None:
        """Save dataset as a compressed .npz file."""
        n = len(self.samples)
        if n == 0:
            raise ValueError("Cannot save empty dataset")
        obs_shape = self.samples[0].observation.shape
        obs_arr = np.zeros((n, *obs_shape), dtype=np.float32)
        mask_arr = np.zeros((n, 8100), dtype=np.int8)
        policy_arr = np.zeros((n, 8100), dtype=np.float32)
        value_arr = np.zeros(n, dtype=np.float32)
        player_arr = np.zeros(n, dtype=np.int8)
        game_ids: list[str] = []
        move_idxs = np.zeros(n, dtype=np.int32)

        for i, s in enumerate(self.samples):
            obs_arr[i] = s.observation
            mask_arr[i] = s.legal_mask
            policy_arr[i] = s.policy_target
            value_arr[i] = s.value_target
            player_arr[i] = s.player
            game_ids.append(s.game_id)
            move_idxs[i] = s.move_index

        np.savez_compressed(
            path,
            obs=obs_arr, mask=mask_arr, policy=policy_arr,
            value=value_arr, player=player_arr,
            game_ids=np.array(game_ids), move_idx=move_idxs,
        )

    def load(self, path: str) -> None:
        """Load dataset from a .npz file."""
        data = np.load(path, allow_pickle=True)
        obs_arr = data["obs"]
        mask_arr = data["mask"]
        policy_arr = data["policy"]
        value_arr = data["value"]
        player_arr = data["player"]
        game_ids = data["game_ids"]
        move_idxs = data["move_idx"]

        self.samples = []
        for i in range(len(obs_arr)):
            self.samples.append(AZSample(
                observation=obs_arr[i].astype(np.float32),
                legal_mask=mask_arr[i].astype(np.int8),
                policy_target=policy_arr[i].astype(np.float32),
                value_target=float(value_arr[i]),
                player=int(player_arr[i]),
                game_id=str(game_ids[i]),
                move_index=int(move_idxs[i]),
            ))

    def to_tensors(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        """Return (obs, mask, policy, value) as torch tensors."""
        n = len(self)
        obs_t = torch.from_numpy(np.stack([s.observation for s in self.samples])).to(device)
        mask_t = torch.from_numpy(np.stack([s.legal_mask for s in self.samples])).to(device)
        policy_t = torch.from_numpy(np.stack([s.policy_target for s in self.samples])).to(device)
        value_t = torch.tensor([s.value_target for s in self.samples], dtype=torch.float32, device=device)
        return obs_t, mask_t, policy_t, value_t
