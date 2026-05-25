from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import torch


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

    __slots__ = (
        "observation", "legal_mask", "_policy_indices", "_policy_probs",
        "value_target", "player", "game_id", "move_index",
    )

    def __init__(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray,
        policy_target: np.ndarray | None = None,
        value_target: float = 0.0,
        player: int = 0,
        game_id: str = "",
        move_index: int = 0,
        policy_indices: np.ndarray | None = None,
        policy_probs: np.ndarray | None = None,
    ) -> None:
        self.observation = observation.astype(np.float32, copy=False)
        self.legal_mask = legal_mask.astype(np.int8, copy=False)
        if policy_indices is not None and policy_probs is not None:
            self._policy_indices = np.asarray(policy_indices, dtype=np.int32)
            self._policy_probs = np.asarray(policy_probs, dtype=np.float32)
        else:
            self.policy_target = (
                np.zeros(8100, dtype=np.float32)
                if policy_target is None else policy_target
            )
        self.value_target = float(value_target)
        self.player = int(player)
        self.game_id = str(game_id)
        self.move_index = int(move_index)

    @property
    def policy_indices(self) -> np.ndarray:
        return self._policy_indices

    @property
    def policy_probs(self) -> np.ndarray:
        return self._policy_probs

    @property
    def policy_nnz(self) -> int:
        return int(len(self._policy_indices))

    @property
    def policy_target(self) -> np.ndarray:
        policy = np.zeros(8100, dtype=np.float32)
        if len(self._policy_indices) > 0:
            policy[self._policy_indices] = self._policy_probs
        return policy

    @policy_target.setter
    def policy_target(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        idx = np.flatnonzero(arr > 1e-8).astype(np.int32)
        self._policy_indices = idx
        self._policy_probs = arr[idx].astype(np.float32, copy=True)


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
        max_policy_nnz = max(s.policy_nnz for s in self.samples)
        policy_idx_arr = np.full((n, max_policy_nnz), -1, dtype=np.int32)
        policy_prob_arr = np.zeros((n, max_policy_nnz), dtype=np.float32)
        policy_len_arr = np.zeros(n, dtype=np.int16)
        value_arr = np.zeros(n, dtype=np.float32)
        player_arr = np.zeros(n, dtype=np.int8)
        game_ids: list[str] = []
        move_idxs = np.zeros(n, dtype=np.int32)

        for i, s in enumerate(self.samples):
            obs_arr[i] = s.observation
            mask_arr[i] = s.legal_mask
            k = s.policy_nnz
            policy_len_arr[i] = k
            if k > 0:
                policy_idx_arr[i, :k] = s.policy_indices
                policy_prob_arr[i, :k] = s.policy_probs
            value_arr[i] = s.value_target
            player_arr[i] = s.player
            game_ids.append(s.game_id)
            move_idxs[i] = s.move_index

        np.savez_compressed(
            path,
            obs=obs_arr, mask=mask_arr,
            policy_idx=policy_idx_arr, policy_prob=policy_prob_arr, policy_len=policy_len_arr,
            value=value_arr, player=player_arr,
            game_ids=np.array(game_ids), move_idx=move_idxs,
        )

    def load(self, path: str) -> None:
        """Load dataset from a .npz file."""
        data = np.load(path, allow_pickle=True)
        obs_arr = data["obs"]
        mask_arr = data["mask"]
        dense_policy_arr = data["policy"] if "policy" in data else None
        policy_idx_arr = data["policy_idx"] if "policy_idx" in data else None
        policy_prob_arr = data["policy_prob"] if "policy_prob" in data else None
        policy_len_arr = data["policy_len"] if "policy_len" in data else None
        value_arr = data["value"]
        player_arr = data["player"]
        game_ids = data["game_ids"]
        move_idxs = data["move_idx"]

        self.samples = []
        for i in range(len(obs_arr)):
            kwargs: dict[str, Any] = {}
            if dense_policy_arr is not None:
                kwargs["policy_target"] = dense_policy_arr[i].astype(np.float32)
            else:
                k = int(policy_len_arr[i])
                kwargs["policy_indices"] = policy_idx_arr[i, :k].astype(np.int32)
                kwargs["policy_probs"] = policy_prob_arr[i, :k].astype(np.float32)
            self.samples.append(AZSample(
                observation=obs_arr[i].astype(np.float32),
                legal_mask=mask_arr[i].astype(np.int8),
                value_target=float(value_arr[i]),
                player=int(player_arr[i]),
                game_id=str(game_ids[i]),
                move_index=int(move_idxs[i]),
                **kwargs,
            ))

    def to_tensors(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        """Return (obs, mask, policy, value) as torch tensors."""
        n = len(self)
        obs_t = torch.from_numpy(np.stack([s.observation for s in self.samples])).to(device)
        mask_t = torch.from_numpy(np.stack([s.legal_mask for s in self.samples])).to(device)
        policy_t = torch.from_numpy(np.stack([s.policy_target for s in self.samples])).to(device)
        value_t = torch.tensor([s.value_target for s in self.samples], dtype=torch.float32, device=device)
        return obs_t, mask_t, policy_t, value_t
