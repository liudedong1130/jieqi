from __future__ import annotations

import torch
import torch.nn as nn

NUM_CHANNELS = 28
ACTION_SPACE = 8100


class PolicyValueNet(nn.Module):
    """CNN policy + value network for Jieqi.

    Input:  ``(batch, 28, 10, 9)`` observation tensor.
    Output: ``(policy_logits, value)`` where
            ``policy_logits`` has shape ``(batch, 8100)`` and
            ``value`` has shape ``(batch, 1)`` in ``[-1, 1]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 64 * 10 * 9  # 5760

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Linear(256, ACTION_SPACE)
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, value)``."""
        h = self.conv(x)
        h = self.fc(h)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value
