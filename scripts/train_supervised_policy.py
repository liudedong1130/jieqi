#!/usr/bin/env python3
"""Supervised pretraining of policy+value net on AZ-style data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rl.az_data import AZDataset
from rl.model import create_model, ACTION_SPACE


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised policy+value training")
    p.add_argument("--data", type=str, required=True, help="AZ dataset .npz file")
    p.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet"])
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load data
    dataset = AZDataset()
    dataset.load(args.data)
    print(f"Loaded {len(dataset)} samples")

    obs_t, mask_t, policy_t, value_t = dataset.to_tensors(device)

    # Create model
    model_kwargs = {}
    if args.model == "resnet":
        model_kwargs = {"channels": args.channels, "num_blocks": args.blocks}
    model = create_model(args.model, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = TensorDataset(obs_t, policy_t, value_t)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        batches = 0

        for batch_obs, batch_policy, batch_value in loader:
            logits, values = model(batch_obs)
            values = values.squeeze(-1)

            policy_loss = nn.functional.cross_entropy(logits, batch_policy)
            value_loss = nn.functional.mse_loss(values, batch_value)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d} | "
                f"p_loss {total_policy_loss / max(batches, 1):.4f} | "
                f"v_loss {total_value_loss / max(batches, 1):.4f}"
            )

    torch.save({"model": model.state_dict(), "model_config": {"type": args.model, **model_kwargs}}, "az_pretrained.pt")
    print("Saved az_pretrained.pt")


if __name__ == "__main__":
    main()
