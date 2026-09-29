"""Retained G1 overarm and experimental underarm tasks for CPU residual learning."""
from dataclasses import dataclass, field
from pathlib import Path
from unilab.base import registry
from ..amp_bowling_env import AmpBowlingEnv
from ..bowling_policy_bundle import load as load_policy_bundle
from ..portable_batting import verify_bundle
from ..scene_distribution import checked_file
from ..unirl_cricket_env import UniRlCricketEnv
from .batting import G1CricketResidualCfg, G1CricketResidualEnv

@dataclass
class G1CricketBowlingCfg(G1CricketResidualCfg):
    amp_checkout: str = ''
    delivery_style: str = 'overarm'
    sim_dt: float = 6.25e-05
    reward_config: dict = field(default_factory=lambda: {'version': 'retained_bowling_v1'})

    def validate(self):
        super().validate()
        if not self.amp_checkout:
            raise ValueError('Configure amp_checkout explicitly before constructing cricket')
        if self.delivery_style not in ('overarm', 'underarm'):
            raise ValueError('delivery_style must be overarm or underarm')

@dataclass
class G1CricketBowlingPolicyCfg(G1CricketBowlingCfg):
    policy_bundle: str = ''
    reward_config: dict = field(default_factory=lambda: {'version': 'checkpoint_bowling_v1'})

    def validate(self):
        super().validate()
        if not self.policy_bundle or self.delivery_style != 'overarm':
            raise ValueError('Checkpoint bowling requires policy_bundle and overarm style')

def make_bowling_env(cfg, *, num_envs=1, backend_type='mujoco'):
    cfg.validate()
    if backend_type != 'mujoco':
        raise ValueError('This task registers only the retained MuJoCo runtime')
    if num_envs < 1:
        raise ValueError('At least one bowling environment is required')
    bundle, groot, amp, scenes = map(Path, (cfg.runtime_bundle, cfg.groot_checkout, cfg.amp_checkout, cfg.scene_bundle))
    manifest = verify_bundle(bundle, groot, scenes)
    checked_file(amp, 'checkpoints/model_6200.pt', manifest['amp_checkpoint'])
    for name, expected in manifest['bowling_scenes'].items():
        checked_file(scenes, name, expected)
    options = dict(manifest['bowling'][cfg.delivery_style]['options'], groot_gather=groot, arm_reference_directory=bundle / 'data' / cfg.delivery_style)
    scene_files = {hand: scenes / f'bowling_{hand}.xml' for hand in ('right', 'left')}
    environment_options = {}
    if isinstance(cfg, G1CricketBowlingPolicyCfg):
        options, scene_files, environment_options = load_policy_bundle(Path(cfg.policy_bundle), groot, amp)
    envs = []
    try:
        for _ in range(num_envs):
            envs.append(AmpBowlingEnv(amp, bundle / 'data', bundle / 'data', rollout_options=options, scene_files=scene_files, **environment_options))
        return G1CricketResidualEnv(cfg, UniRlCricketEnv(envs, cfg.seed))
    except Exception:
        for env in envs:
            env.close()
        raise
registry.register_env_config('G1CricketBowlingCpu', G1CricketBowlingCfg)
registry.register_env('G1CricketBowlingCpu', make_bowling_env, sim_backend='mujoco')
registry.register_env_config('G1CricketBowlingPolicyCpu', G1CricketBowlingPolicyCfg)
registry.register_env('G1CricketBowlingPolicyCpu', make_bowling_env, sim_backend='mujoco')
