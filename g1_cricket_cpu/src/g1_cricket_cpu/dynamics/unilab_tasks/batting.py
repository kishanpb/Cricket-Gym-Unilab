"""Registered external CPU task; this is not a ManagerBasedRlEnv backend."""
from dataclasses import dataclass, field, fields
from pathlib import Path
from unilab.base import registry
from unilab.base.base import ABEnv, EnvCfg
from unilab.base.np_env import NpEnvState
from ..portable_batting import verify_bundle
from ..unirl_batting_env import batting_env_factory

@dataclass
class G1CricketResidualCfg(EnvCfg):
    runtime_bundle: str = ''
    groot_checkout: str = ''
    scene_bundle: str = ''
    seed: int = 29
    sim_dt: float = 0.000125
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 8.0
    reward_config: dict = field(default_factory=lambda: {'version': 'retained_batting_v1'})

    def validate(self):
        super().validate()
        defaults = type(self)()
        for item in fields(EnvCfg):
            if getattr(self, item.name) != getattr(defaults, item.name):
                raise ValueError(f'Retained cricket runtime does not support overriding {item.name}')
        if self.reward_config != defaults.reward_config:
            raise ValueError('Retained cricket runtime uses its fixed reward')
        for name in ('runtime_bundle', 'groot_checkout', 'scene_bundle'):
            if not getattr(self, name):
                raise ValueError(f'Configure {name} explicitly before constructing cricket')

class G1CricketResidualEnv(ABEnv):

    def __init__(self, cfg, adapter):
        self._cfg, self.adapter = (cfg, adapter)
        self._state = None

    @property
    def cfg(self):
        return self._cfg

    @property
    def num_envs(self):
        return self.adapter.num_envs

    @property
    def observation_space(self):
        return self.adapter.observation_space

    @property
    def action_space(self):
        return self.adapter.action_space

    @property
    def obs_groups_spec(self):
        return self.adapter.obs_groups_spec

    @property
    def algo_capabilities(self):
        return self.adapter.algo_capabilities

    @property
    def state(self):
        return self._state

    def _record(self, state):
        self._state = NpEnvState(state.obs, state.reward, state.terminated, state.truncated, state.info, state.final_observation)
        return self._state

    def init_state(self):
        return self._record(self.adapter.init_state())

    def reset(self, env_indices):
        result = self.adapter.reset(env_indices)
        self._record(self.adapter.state)
        return result

    def step(self, actions):
        return self._record(self.adapter.step(actions))

    def set_nan_guard(self, guard):
        self.adapter.set_nan_guard(guard)

    def close(self):
        self.adapter.close()

def make_cricket_env(cfg, *, num_envs=1, backend_type='mujoco'):
    cfg.validate()
    if backend_type != 'mujoco':
        raise ValueError('This task registers only the retained MuJoCo runtime')
    bundle, upstream, scenes = map(Path, (cfg.runtime_bundle, cfg.groot_checkout, cfg.scene_bundle))
    manifest = verify_bundle(bundle, upstream, scenes)
    options = dict(manifest['environment_options'], physics_backend='mujoco', recovery_fade=True)
    options['reference_files'] = [bundle / path for path in manifest['references']]
    factory = batting_env_factory(upstream, bundle / 'data', scenes / 'batting.xml', seed=cfg.seed, **options)
    return G1CricketResidualEnv(cfg, factory(num_envs, None))
registry.register_env_config('G1CricketResidualCpu', G1CricketResidualCfg)
registry.register_env('G1CricketResidualCpu', make_cricket_env, sim_backend='mujoco')
