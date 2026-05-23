from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Categorical

from rl.model import PolicyValueNet

if TYPE_CHECKING:
    from jieqi.env import JieqiEnv


class PolicyAgent:
    """Agent that selects actions using a trained PPO policy network.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pt`` checkpoint saved by ``PPOTrainer.save()``.
    deterministic : bool
        If True, pick the action with the highest logit (argmax).
        If False, sample from the categorical distribution.
    """

    def __init__(
        self,
        checkpoint_path: str,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self._deterministic = deterministic
        self._rng = random.Random(seed)

    def select_action(self, env: JieqiEnv) -> int:
        obs = env.observation()
        mask = env.legal_action_mask()

        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _value = self.model(t)
        logits = logits[0]
        mask_t = torch.from_numpy(mask).to(self.device)
        logits = logits.masked_fill(mask_t == 0, -1e9)

        # NaN fallback → random legal
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            actions = env.legal_actions()
            return self._rng.choice(actions)

        if self._deterministic:
            action = int(logits.argmax().item())
        else:
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())

        return action
