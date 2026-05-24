from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rl.az_data import AZDataset


@dataclass
class TrainStats:
    policy_loss: float
    value_loss: float
    total_loss: float


def masked_policy_loss(
    logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy against a search distribution over legal actions only."""
    mask = legal_mask.to(dtype=torch.bool, device=logits.device)
    target = policy_target.to(dtype=logits.dtype, device=logits.device)
    target = target * mask.to(dtype=target.dtype)
    target_sum = target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target = target / target_sum

    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = nn.functional.log_softmax(masked_logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def train_policy_value(
    dataset: AZDataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    epochs: int,
    batch_size: int,
    value_coef: float = 0.5,
    log_prefix: str = "  ",
) -> TrainStats:
    obs_t, mask_t, policy_t, value_t = dataset.to_tensors(device)
    ds = TensorDataset(obs_t, mask_t, policy_t, value_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    last_stats = TrainStats(0.0, 0.0, 0.0)
    for epoch in range(1, epochs + 1):
        total_p, total_v, total_loss, n = 0.0, 0.0, 0.0, 0
        for bo, bm, bp, bv in loader:
            logits, values = model(bo)
            p_loss = masked_policy_loss(logits, bp, bm)
            v_loss = nn.functional.mse_loss(values.squeeze(-1), bv)
            loss = p_loss + value_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_p += p_loss.item()
            total_v += v_loss.item()
            total_loss += loss.item()
            n += 1

        last_stats = TrainStats(
            policy_loss=float(total_p / max(n, 1)),
            value_loss=float(total_v / max(n, 1)),
            total_loss=float(total_loss / max(n, 1)),
        )
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{log_prefix}epoch {epoch:3d} | "
                f"p_loss {last_stats.policy_loss:.4f} | "
                f"v_loss {last_stats.value_loss:.4f}"
            )

    return last_stats


def policy_mass_is_legal(dataset: AZDataset) -> bool:
    for sample in dataset.samples:
        illegal_mass = np.asarray(sample.policy_target)[np.asarray(sample.legal_mask) == 0].sum()
        if abs(float(illegal_mass)) > 1e-6:
            return False
    return True
