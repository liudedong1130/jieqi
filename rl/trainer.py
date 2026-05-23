from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from jieqi.env import JieqiEnv
from rl.model import PolicyValueNet


class PPOTrainer:
    """Minimal PPO trainer for Jieqi self-play.

    Parameters
    ----------
    env : JieqiEnv
        Environment instance (will be reset internally).
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda parameter.
    clip_ratio : float
        PPO clipping epsilon.
    entropy_coef : float
        Entropy bonus coefficient.
    value_coef : float
        Value loss coefficient.
    update_epochs : int
        Number of optimisation epochs per update.
    episodes_per_update : int
        Collect this many episodes before each PPO update.
    """

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
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.episodes_per_update = episodes_per_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Buffers for trajectory collection
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[int] = []
        self._logp_buf: list[float] = []
        self._val_buf: list[float] = []
        self._rew_buf: list[float] = []
        self._done_buf: list[bool] = []
        self._player_buf: list[int] = []

        self._episode_count = 0

    # ------------------------------------------------------------------
    #  Trajectory collection
    # ------------------------------------------------------------------

    def collect_episode(self, seed: int | None = None) -> dict:
        """Run one episode of self-play, store trajectory, return stats."""
        obs = self.env.reset(seed=seed)
        done = False
        steps = 0
        ep_reward = 0.0
        illegal_count = 0

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
            ep_reward += reward

        self._episode_count += 1
        return {
            "steps": steps,
            "reward": ep_reward,
            "illegal": illegal_count,
        }

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
    #  GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute advantages and returns for the buffered trajectory.

        Handles the zero-sum perspective flip: every step the current
        player changes, so ``V(s_{t+1})`` is from the opponent's
        perspective and must be negated.
        """
        T = len(self._rew_buf)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = 0.0
            else:
                # next value is from opponent → negate for zero-sum
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
        """Run one PPO update on all buffered episodes, then clear buffer."""
        if len(self._rew_buf) == 0:
            return {}

        advantages, returns = self._compute_gae()

        obs_t = torch.from_numpy(np.stack(self._obs_buf)).to(self.device)
        act_t = torch.tensor(self._act_buf, dtype=torch.long, device=self.device)
        old_logp_t = torch.tensor(self._logp_buf, dtype=torch.float32, device=self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        stats: dict[str, list[float]] = defaultdict(list)

        for _ in range(self.update_epochs):
            logits, values = self.model(obs_t)
            values = values.squeeze(-1)

            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_t)
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            value_loss = nn.functional.mse_loss(values, ret_t)
            entropy_loss = -entropy

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["total_loss"].append(loss.item())

        # Clear buffers
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._val_buf.clear()
        self._rew_buf.clear()
        self._done_buf.clear()
        self._player_buf.clear()

        return {k: float(np.mean(v)) for k, v in stats.items()}

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
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._episode_count = ckpt.get("episode", 0)

    @property
    def episode_count(self) -> int:
        return self._episode_count
