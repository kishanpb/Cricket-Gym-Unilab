"""Serial CPU cricket adapter for uni_rl's injected environment protocol."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from uni_rl.env_contract import EnvAlgoCapabilities

@dataclass
class CricketState:
    obs: dict
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict
    final_observation: dict | None = None

class UniRlCricketEnv:
    """Serial CPU vectorization; underlying cricket rewards and physics are unchanged."""

    def __init__(self, envs, seed=0):
        self.envs = envs
        self.num_envs = len(envs)
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space
        self.obs_groups_spec = {'obs': self.observation_space.shape[0]}
        self.algo_capabilities = EnvAlgoCapabilities(action_low=self.action_space.low.copy(), action_high=self.action_space.high.copy())
        self.cfg = SimpleNamespace(max_episode_seconds=8.0, ctrl_dt=0.02)
        self.play_capabilities = SimpleNamespace(supports_physics_state_playback=False)
        self.seed = seed
        self.state = None

    def init_state(self):
        resets = [env.reset(seed=self.seed + i) for i, env in enumerate(self.envs)]
        self.state = CricketState({'obs': np.stack([obs for obs, _ in resets])}, np.zeros(self.num_envs), np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool), {'steps': np.zeros(self.num_envs, dtype=np.int64), 'reset_info': [info for _, info in resets]})
        return self.state

    def reset(self, env_indices):
        if self.state is None:
            self.init_state()
        infos = []
        for i in env_indices:
            obs, info = self.envs[i].reset()
            self.state.obs['obs'][i] = obs
            self.state.reward[i] = 0
            self.state.terminated[i] = self.state.truncated[i] = False
            self.state.info['steps'][i] = 0
            infos.append(info)
        self.state.final_observation = None
        return ({'obs': self.state.obs['obs'][env_indices].copy()}, {'reset_info': infos})

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, *self.action_space.shape) or not np.isfinite(actions).all():
            raise ValueError('Expected one finite action vector per cricket environment')
        if self.state is None:
            self.init_state()
        results = [env.step(action) for env, action in zip(self.envs, actions, strict=True)]
        obs, rewards, terminated, truncated, infos = zip(*results, strict=True)
        obs = np.stack(obs)
        terminated, truncated = (np.asarray(terminated), np.asarray(truncated))
        done = terminated | truncated
        final = {'obs': obs.copy()} if done.any() else None
        steps = self.state.info['steps'] + 1
        reset_infos = {}
        for i in np.flatnonzero(done):
            obs[i], reset_infos[int(i)] = self.envs[i].reset()
            steps[i] = 0
        self.state = CricketState({'obs': obs}, np.asarray(rewards), terminated, truncated, {'steps': steps, 'case_info': list(infos), 'reset_info': reset_infos}, final)
        return self.state

    def set_nan_guard(self, guard):
        if guard is not None:
            raise NotImplementedError('Cricket adapter does not support physics-snapshot NaN guards')

    def close(self):
        for env in self.envs:
            env.close()
