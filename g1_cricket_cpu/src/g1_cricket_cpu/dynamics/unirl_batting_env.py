"""CPU batting factory for uni_rl's injected environment protocol."""
from functools import partial
from .unirl_cricket_env import CricketState as BattingState
from .unirl_cricket_env import UniRlCricketEnv as UniRlBattingEnv

def make_batting_env(upstream, checkout, scene, options, seed, num_envs, env_cfg_override=None):
    from .groot_batting_env import GrootBattingEnv
    if env_cfg_override:
        raise ValueError('Configure batting through the explicit factory options')
    if num_envs < 1:
        raise ValueError('At least one batting environment is required')
    envs = []
    try:
        for _ in range(num_envs):
            envs.append(GrootBattingEnv(upstream, checkout, scene, **options))
        return UniRlBattingEnv(envs, seed)
    except Exception:
        for env in envs:
            env.close()
        raise

def batting_env_factory(upstream, checkout, scene, *, seed=0, **options):
    return partial(make_batting_env, upstream, checkout, scene, options, seed)
