"""Native G1 batting foundation with a declared rigid bat fixture."""

from unilab.base import registry

from .bimanual_contact import G1BimanualContactCfg
from .impact import G1CricketImpactCfg
from .task import G1CricketCfg, make_g1_cricket_env
from .tracking import G1BimanualTrackingCfg

registry.register_env_config("G1CricketBatting", G1CricketCfg)
registry.register_env("G1CricketBatting", make_g1_cricket_env, sim_backend="mujoco")
registry.register_env_config("G1CricketImpact", G1CricketImpactCfg)
registry.register_env("G1CricketImpact", make_g1_cricket_env, sim_backend="mujoco")
registry.register_env_config("G1CricketBimanualTracking", G1BimanualTrackingCfg)
registry.register_env("G1CricketBimanualTracking", make_g1_cricket_env, sim_backend="mujoco")

registry.register_env_config("G1CricketBimanualContact", G1BimanualContactCfg)
registry.register_env("G1CricketBimanualContact", make_g1_cricket_env, sim_backend="mujoco")

registry.register_env_config("G1CricketBimanualLearning", G1BimanualContactCfg)
registry.register_env("G1CricketBimanualLearning", make_g1_cricket_env, sim_backend="mujoco")

__all__ = ["G1CricketCfg", "make_g1_cricket_env"]
