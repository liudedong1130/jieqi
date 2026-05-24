from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from jieqi.env import JieqiEnv
from rl.model import PolicyValueNet, create_model


def _get_device(requested: str | None = None) -> torch.device:
    """Resolve device: explicit string > CUDA > MPS > CPU."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        model_type: str = "simple_cnn",
        model_kwargs: dict | None = None,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.episodes_per_update = episodes_per_update

        self.device = _get_device(device)
        self._model_type = model_type
        self._model_kwargs = model_kwargs or {}
        self.model = create_model(model_type, **self._model_kwargs).to(self.device)
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
        """Run one episode of self-play and buffer all transitions.

        The episode return is reported from the perspective of the *first*
        player (RED).  +1 = RED won, -1 = RED lost, 0 = draw/truncated.
        """
        obs = self.env.reset(seed=seed)
        done = False
        steps = 0
        first_player = self.env.current_player()

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

        # Determine episode return from first player's perspective
        if any(self._rew_buf):
            # last reward is from the winner's perspective
            last_rew = self._rew_buf[-1]
            last_player = self._player_buf[-1]
            if last_rew > 0:
                ep_return = 1.0 if last_player == first_player else -1.0
            elif last_rew < 0:
                ep_return = -1.0 if last_player == first_player else 1.0
            else:
                ep_return = 0.0
        else:
            ep_return = 0.0  # truncated

        self._episode_count += 1
        return {"steps": steps, "return": ep_return}

    def collect_episode_with_opponent(
        self, seed: int | None, opponent: Any, self_play_prob: float = 0.5,
    ) -> dict:
        """Run one episode with an opponent from the pool.

        With probability *self_play_prob*, both sides use the training
        policy (standard self-play).  Otherwise the training policy
        plays one randomly-chosen colour against *opponent*.

        Only the training policy's transitions are buffered.  Returns
        are computed via Monte-Carlo from the game outcome.
        """
        use_self_play = random.random() < self_play_prob
        if use_self_play:
            return self.collect_episode(seed=seed)

        obs = self.env.reset(seed=seed)
        done = False
        steps = 0
        train_player = random.choice([0, 1])  # 0=RED, 1=BLACK

        while not done:
            current = self.env.current_player()
            if current == train_player:
                # Training policy's move — buffer it
                action, log_prob, value = self._select_action(obs)
                next_obs, reward, terminated, truncated, _info = self.env.step(action)
                done = terminated or truncated

                self._obs_buf.append(obs)
                self._act_buf.append(action)
                self._logp_buf.append(log_prob)
                self._val_buf.append(value)
                self._rew_buf.append(reward)
                self._done_buf.append(done)
                self._player_buf.append(current)

                obs = next_obs
                steps += 1
            else:
                # Opponent's move — don't buffer
                action = opponent.select_action(self.env)
                if action not in self.env.legal_actions():
                    action = self.env.legal_actions()[0]
                next_obs, reward, terminated, truncated, _info = self.env.step(action)
                done = terminated or truncated
                obs = next_obs
                steps += 1
                if terminated and reward > 0:
                    # Opponent won → back-propagate -1 to last training move
                    for i in range(len(self._rew_buf) - 1, -1, -1):
                        if self._player_buf[i] == train_player:
                            self._rew_buf[i] = -1.0
                            break

        # MC returns: game outcome from train_player's perspective
        ep_return = 0.0
        for i in range(len(self._rew_buf) - 1, -1, -1):
            if self._rew_buf[i] > 0:
                ep_return = 1.0 if self._player_buf[i] == train_player else -1.0
                break
            elif self._done_buf[i] and self._rew_buf[i] == 0:
                ep_return = 0.0
                break

        self._episode_count += 1
        return {"steps": steps, "return": ep_return, "opponent": True}

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

    # ------------------------------------------------------------------
    #  Value function definition
    # ------------------------------------------------------------------
    #  V(s) = E[ final return | state s, current player to move ]
    #
    #  The final return is +1 if the *current player* wins, -1 if they
    #  lose, 0 for a draw.  Because the game is zero-sum, the opponent's
    #  value is the negation:  V_opp(s) = -V_own(s).
    #
    #  In self-play every step swaps the current player, so the TD
    #  target for step t uses *negated* next-value:
    #      target_t = r_t + γ · (-V_{t+1}) · (1 - done_t)
    #
    #  Likewise, when GAE accumulates multi-step TD errors the sign
    #  flips at each step because each δ_{t+k} is from a different
    #  player's perspective:
    #      A_t = δ_t - (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} - …
    # ------------------------------------------------------------------

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
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

            if t == T - 1:
                gae = delta
            else:
                # accumulated gae so far is from opponent's perspective;
                # negate it before adding to the current player's δ_t
                gae = delta + self.gamma * self.gae_lambda * mask * (-gae)

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
        ckpt: dict = {
            "model": self.model.state_dict(),
            "model_config": {"type": self._model_type, **self._model_kwargs},
            "optimizer": self.optimizer.state_dict(),
            "episode": self._episode_count,
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        """Full resume: model + optimizer + episode count."""
        from rl.model import _model_from_config

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "model_config" in ckpt:
            self._model_type = ckpt["model_config"]["type"]
            self._model_kwargs = {k: v for k, v in ckpt["model_config"].items() if k != "type"}
            self.model = _model_from_config(ckpt["model_config"]).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._episode_count = ckpt.get("episode", 0)

    def load_model_only(self, path: str) -> dict:
        """Load model weights from a pretrained checkpoint (no optimizer).

        Returns the checkpoint's model_config dict for logging.
        The current model architecture must match the checkpoint.
        Episode count and optimizer state are **not** restored.
        """
        from rl.model import _model_from_config

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        if cfg:
            ckpt_type = cfg.get("type", "?")
            if ckpt_type != self._model_type:
                self._model_type = ckpt_type
                self._model_kwargs = {k: v for k, v in cfg.items() if k != "type"}
                self.model = _model_from_config(cfg).to(self.device)
                lr = self.optimizer.param_groups[0].get("lr", 3e-4)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.load_state_dict(ckpt["model"])
        return cfg

    @property
    def episode_count(self) -> int:
        return self._episode_count
