from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from jieqi.env import JieqiEnv
from rl.model import PolicyValueNet


class PPOTrainer:
    """Minimal PPO trainer for Jieqi self-play."""

    def __init__(
        self,
        env: JieqiEnv,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        episodes_per_update: int = 8,
        device: str | None = None,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.episodes_per_update = episodes_per_update

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PolicyValueNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[int] = []
        self._logp_buf: list[float] = []
        self._val_buf: list[float] = []
        self._rew_buf: list[float] = []
        self._done_buf: list[bool] = []
        self._player_buf: list[int] = []

        self._episode_count = 0

    # ------------------------------------------------------------------
    #  Seed
    # ------------------------------------------------------------------

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    #  Trajectory collection
    # ------------------------------------------------------------------

    def collect_episode(self, seed: int | None = None) -> dict:
        obs = self.env.reset(seed=seed)
        done = False
        steps = 0
        ep_return = 0.0

        while not done:
            mover = self.env.current_player()
            action, log_prob, value = self._select_action(obs)

            next_obs, reward, terminated, truncated, _info = self.env.step(action)
            done = terminated or truncated

            self._obs_buf.append(obs)
            self._act_buf.append(action)
            self._logp_buf.append(log_prob)
            self._val_buf.append(value)
            self._rew_buf.append(reward)
            self._done_buf.append(done)
            self._player_buf.append(mover)

            obs = next_obs
            steps += 1
            if terminated:
                ep_return = 1.0 if reward > 0 else (-1.0 if reward < 0 else 0.0)
            elif truncated:
                ep_return = 0.0

        self._episode_count += 1
        return {"steps": steps, "return": ep_return}

    @torch.no_grad()
    def _select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        logits, value = self.model(t)
        logits = logits[0]

        mask = self.env.legal_action_mask()
        mask_t = torch.from_numpy(mask).to(self.device)
        logits = logits.masked_fill(mask_t == 0, -1e9)

        dist = Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        return action, log_prob, value.item()

    # ------------------------------------------------------------------
    #  GAE
    # ------------------------------------------------------------------

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        T = len(self._rew_buf)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = 0.0
            else:
                next_val = -self._val_buf[t + 1]
            mask = 1.0 - float(self._done_buf[t])
            delta = self._rew_buf[t] + self.gamma * next_val * mask - self._val_buf[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + np.array(self._val_buf, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    #  PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        if len(self._rew_buf) == 0:
            return {}

        advantages, returns = self._compute_gae()

        obs_t = torch.from_numpy(np.stack(self._obs_buf)).to(self.device)
        act_t = torch.tensor(self._act_buf, dtype=torch.long, device=self.device)
        old_logp_t = torch.tensor(self._logp_buf, dtype=torch.float32, device=self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)

        # normalize
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        stats: dict[str, list[float]] = defaultdict(list)

        for _ in range(self.update_epochs):
            logits, values = self.model(obs_t)
            values = values.squeeze(-1)

            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_t)
            with torch.no_grad():
                approx_kl = ((new_logp - old_logp_t) ** 2).mean()
                clip_frac = ((ratio - 1).abs() > self.clip_ratio).float().mean()

            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            value_loss = nn.functional.mse_loss(values, ret_t)
            entropy_loss = -entropy

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                stats["nan_detected"] = [1.0]
                self._clear_buf()
                return {k: float(np.mean(v)) for k, v in stats.items()}

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["total_loss"].append(loss.item())
            stats["approx_kl"].append(approx_kl.item())
            stats["clip_frac"].append(clip_frac.item())

        # explained variance
        with torch.no_grad():
            _, vals = self.model(obs_t)
            vals = vals.squeeze(-1).cpu().numpy()
        ev = 1.0 - np.var(ret_t.cpu().numpy() - vals) / (np.var(ret_t.cpu().numpy()) + 1e-8)
        stats["explained_var"].append(float(ev))

        self._clear_buf()
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _clear_buf(self) -> None:
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._val_buf.clear()
        self._rew_buf.clear()
        self._done_buf.clear()
        self._player_buf.clear()

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episode": self._episode_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._episode_count = ckpt.get("episode", 0)

    @property
    def episode_count(self) -> int:
        return self._episode_count
