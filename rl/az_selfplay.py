from __future__ import annotations

from typing import Any

import numpy as np

from jieqi.env import JieqiEnv
from rl.az_data import AZSample, AZDataset
from rl.ismcts import ISMCTSAgent


def generate_az_selfplay_data(
    agent: ISMCTSAgent,
    num_games: int,
    max_steps: int,
    seed: int,
    replay_buffer: AZDataset | None = None,
    max_buffer_samples: int | None = None,
    progress_interval: int = 0,
) -> AZDataset:
    """Generate AlphaZero-style self-play training data using ISMCTS.

    For each move the agent's visit-count distribution becomes the
    policy target, and the final game outcome becomes the value target
    from each player's perspective.

    Parameters
    ----------
    agent : ISMCTSAgent
        Agent used for both sides.
    num_games : int
        Number of games to play.
    max_steps : int
        Max steps per game.
    seed : int
        Base random seed.
    replay_buffer : AZDataset | None
        If provided, new samples are appended.
    max_buffer_samples : int | None
        If provided, keep only the most recent samples after appending.
    progress_interval : int
        If > 0, print progress every N moves.

    Returns
    -------
    AZDataset
    """
    if replay_buffer is None:
        dataset = AZDataset()
    else:
        dataset = replay_buffer

    for g in range(num_games):
        env = JieqiEnv(max_steps=max_steps)
        env.reset(seed=seed + g)

        moves: list[dict] = []
        done = False
        if progress_interval > 0:
            print(f"    selfplay game {g + 1}/{num_games} ...", flush=True)
        while not done:
            player = env.current_player()
            obs = env.observation().copy()
            mask = env.legal_action_mask().copy()
            policy, action = agent.get_policy(env)
            if action not in env.legal_actions():
                action = env.legal_actions()[0]
                policy = np.zeros(8100, dtype=np.float32)
                policy[action] = 1.0
            _obs, reward, terminated, truncated, _info = env.step(action)
            moves.append({
                "obs": obs, "mask": mask, "policy": policy,
                "action": action, "player": player,
                "reward": reward, "terminated": terminated,
            })
            done = terminated or truncated
            if progress_interval > 0 and len(moves) % progress_interval == 0:
                print(
                    f"      selfplay game {g + 1}/{num_games} | step {len(moves)} | "
                    f"buffer {len(dataset)}",
                    flush=True,
                )

        # Determine winner
        winner = None
        for m in reversed(moves):
            if m["terminated"] and m["reward"] > 0:
                winner = m["player"]
                break

        for i, m in enumerate(moves):
            value = 1.0 if (winner is not None and m["player"] == winner) else (-1.0 if winner is not None else 0.0)
            dataset.add(AZSample(
                observation=m["obs"], legal_mask=m["mask"],
                policy_target=m["policy"], value_target=value,
                player=m["player"], game_id=f"az_{seed}_{g}", move_index=i,
            ))
        if progress_interval > 0:
            print(
                f"    selfplay game {g + 1}/{num_games} done | steps {len(moves)} | "
                f"buffer {len(dataset)}",
                flush=True,
            )

    if max_buffer_samples is not None:
        dataset.trim_to_recent(max_buffer_samples)
    return dataset
