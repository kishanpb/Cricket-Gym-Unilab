"""Process-parallel batting with explicit, synchronized episode resets."""

from functools import partial
import time

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
import torch

from integrations.g1_dynamics.groot_batting_env import GrootBattingEnv
from unilab.tasks.manipulation.g1_cricket.articulated_learning import (
    ARTICULATED_DELIVERY_POOL,
)


def scheduled_cases(seed, rank, workers):
    rng = np.random.default_rng(seed)
    feeds = len(ARTICULATED_DELIVERY_POOL)
    index = 0
    while True:
        for case in rng.permutation(2 * feeds)[::-1]:
            if index % workers == rank:
                yield {
                    "hand": "right" if case < feeds else "left",
                    "feed_index": int(case % feeds),
                }
            index += 1


class ScheduledBatting(gym.Wrapper):
    def __init__(self, env, rank, workers, seed):
        super().__init__(env)
        self.rank, self.workers, self.schedule_seed = rank, workers, seed
        self.cases = scheduled_cases(seed, rank, workers)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.cases = scheduled_cases(self.schedule_seed, self.rank, self.workers)
        case = options["cases"][self.rank] if options is not None else next(self.cases)
        self.started = time.monotonic()
        return self.env.reset(seed=seed, options=case)

    def episode_record(self):
        return self.env.episodes[-1]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Nullable diagnostics stay intact in episode_record, outside info batching.
        info = {key: value for key, value in info.items() if key != "result"}
        return obs, reward, terminated, truncated, info

    def parity_record(self, baseline, output):
        from integrations.g1_dynamics.groot_batting_ppo import parity_result

        return parity_result(self.env, baseline, output, self.started)


class ExplicitResetVectorWrapper(GymnasiumWrapper):
    def reset(self, *, options=None):
        observation, info = self._env.reset(seed=self._seed, options=options)
        self._seed = None
        return torch.as_tensor(observation, device=self.device), info

    def close(self):
        self._env.close(timeout=5)


def make_worker(
    upstream, checkout, scene, recovery_fade, rank, workers, seed, env_options=None
):
    return ScheduledBatting(
        GrootBattingEnv(
            upstream,
            checkout,
            scene,
            recovery_fade=recovery_fade,
            **(env_options or {}),
        ),
        rank,
        workers,
        seed,
    )


def make_vector(
    upstream, checkout, scene, *, workers, seed, recovery_fade, env_options=None
):
    raw = AsyncVectorEnv(
        [
            partial(
                make_worker,
                upstream,
                checkout,
                scene,
                recovery_fade,
                rank,
                workers,
                seed,
                env_options,
            )
            for rank in range(workers)
        ],
        context="spawn",
        autoreset_mode=AutoresetMode.DISABLED,
    )
    raw.device = "cpu"
    return raw, ExplicitResetVectorWrapper(raw)
