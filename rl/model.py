from __future__ import annotations

import torch
import torch.nn as nn

NUM_CHANNELS = 28
ACTION_SPACE = 8100


# =============================================================================
#  Simple CNN
# =============================================================================


class PolicyValueNet(nn.Module):
    """Simple CNN policy + value network for Jieqi."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 64 * 10 * 9

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
        h = self.conv(x)
        h = self.fc(h)
        return self.policy_head(h), self.value_head(h)


# =============================================================================
#  Residual CNN
# =============================================================================


class ResidualBlock(nn.Module):
    """Pre-activation residual block with two conv layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResidualPolicyValueNet(nn.Module):
    """Residual CNN policy + value network for Jieqi.

    Parameters
    ----------
    channels : int
        Number of filters in residual blocks (default 128).
    num_blocks : int
        Number of residual blocks (default 3).
    """

    def __init__(self, channels: int = 128, num_blocks: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks

        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head: 2-channel conv → flatten → FC
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * 10 * 9, ACTION_SPACE)

        # Value head: 1-channel conv → flatten → FC
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(1 * 10 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_conv(x)
        h = self.blocks(h)

        p = self.policy_conv(h)
        p = p.flatten(1)
        logits = self.policy_fc(p)

        v = self.value_conv(h)
        v = v.flatten(1)
        value = self.value_fc(v)

        return logits, value

    def config(self) -> dict:
        return {"type": "resnet", "channels": self.channels, "num_blocks": self.num_blocks}


# =============================================================================
#  Factory
# =============================================================================


def create_model(model_type: str = "simple_cnn", **kwargs) -> nn.Module:
    """Create a policy-value network by name.

    Parameters
    ----------
    model_type : str
        ``"simple_cnn"`` or ``"resnet"``.
    **kwargs
        Passed to the model constructor (e.g. ``channels``, ``num_blocks``).
    """
    if model_type == "simple_cnn":
        return PolicyValueNet()
    elif model_type == "resnet":
        return ResidualPolicyValueNet(
            channels=kwargs.get("channels", 128),
            num_blocks=kwargs.get("num_blocks", 3),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _model_from_config(config: dict) -> nn.Module:
    """Reconstruct a model from a saved config dict."""
    t = config.get("type", "simple_cnn")
    return create_model(t, **{k: v for k, v in config.items() if k != "type"})
