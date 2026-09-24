"""Native G1 batting foundation with a declared rigid bat fixture."""

from unilab.base import registry

from .impact import G1CricketImpactCfg
from .task import G1CricketCfg, make_g1_cricket_env

registry.register_env_config("G1CricketBatting", G1CricketCfg)
registry.register_env("G1CricketBatting", make_g1_cricket_env, sim_backend="mujoco")
registry.register_env_config("G1CricketImpact", G1CricketImpactCfg)
registry.register_env("G1CricketImpact", make_g1_cricket_env, sim_backend="mujoco")

__all__ = ["G1CricketCfg", "make_g1_cricket_env"]
