"""G1 cricket manager terms; only reset events write root state."""

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import numpy as np

from unilab.assets.hub import ensure_robot_assets_for_paths
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    _resolve_backend_entity_contract,
)
from unilab.managers import ManagerTermBaseCfg

from .scene import (
    BALL_CONTACT_NAMES,
    CONTACT_SLOTS,
    CONTACT_WIDTH,
    SUPPORT_NAMES,
    SUPPORT_SLOTS,
    build_scene,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity


@dataclass
class G1CricketCfg(ManagerBasedRlEnvCfg):
    handedness: str = "right"


class G1CricketEnv(ManagerBasedRlEnv):
    scene_directory: TemporaryDirectory[str]

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.scene_directory.cleanup()


def make_g1_cricket_env(
    cfg: G1CricketCfg, num_envs: int = 1, backend_type: str = "mujoco"
) -> G1CricketEnv:
    cfg.validate()
    assert cfg.scene is not None
    ensure_robot_assets_for_paths([cfg.scene.model_file])
    directory = TemporaryDirectory(prefix="unilab-g1-cricket-")
    try:
        scene_file = Path(directory.name) / "cricket.xml"
        build_scene(Path(cfg.scene.model_file), scene_file, cfg.handedness)
        scene = replace(cfg.scene, model_file=str(scene_file))
        cfg = replace(cfg, scene=scene)
        base_name, body_state = _resolve_backend_entity_contract(cfg)
        kwargs = env_backend_kwargs(cfg)
        kwargs["base_name"] = base_name
        backend = create_backend(
            backend_type, scene, num_envs, cfg.sim_dt, body_state_required=body_state, **kwargs
        )
        try:
            env = G1CricketEnv(cfg, backend, num_envs)
        except Exception:
            backend.cleanup_scene_assets()
            raise
        env.scene_directory = directory
        return env
    except Exception:
        directory.cleanup()
        raise


class BallObservation:
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self.ball = cast("Entity", env.scene["ball"])
        self.robot = cast("Entity", env.scene["robot"])
        self.bat = env.scene.bind_sensor_data(("bat_center_world",))

    def __call__(self, env: ManagerBasedRlEnv) -> np.ndarray:
        return np.concatenate(
            (
                self.ball.data.root_link_pos_w - self.bat.read(),
                self.ball.data.root_link_lin_vel_w,
                self.robot.data.root_link_pos_w[:, 2:3],
            ),
            axis=1,
        )


class ContactObservation:
    """End-of-control-step snapshots, not integrated force peaks or impulses."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self.contacts = env.scene.bind_sensor_data(BALL_CONTACT_NAMES)
        self.support = env.scene.bind_sensor_data(SUPPORT_NAMES)
        self.fixture = env.scene.bind_sensor_data(("bat_fixture_force", "bat_fixture_torque"))

    def __call__(self, env: ManagerBasedRlEnv) -> np.ndarray:
        values = []
        for view, slots in ((self.contacts, CONTACT_SLOTS), (self.support, SUPPORT_SLOTS)):
            rows = view.read().reshape(env.num_envs, -1, slots, CONTACT_WIDTH)
            if np.any(rows[..., 0] > slots):
                raise RuntimeError("G1 cricket contact sensor capacity exceeded")
            force = np.where((rows[..., 0] > 0)[..., None], rows[..., 1:4], 0)
            normal = force[..., 0].sum(axis=-1)
            shear = np.linalg.norm(force[..., 1:], axis=-1).sum(axis=-1)
            tactile = np.stack((normal > 0, normal / 100, shear / 100), axis=-1)
            values.append(tactile.reshape(env.num_envs, -1))
        return np.concatenate((*values, self.fixture.read() / 100), axis=1)


class ResetBall:
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self.ball = cast("Entity", env.scene["ball"])

    def __call__(self, env: ManagerBasedRlEnv, env_ids: np.ndarray, speed: float = 4.0) -> None:
        states = np.array(self.ball.data.default_root_state[env_ids], copy=True)
        states[:, 7] = -speed
        self.ball.write_root_state_to_sim(states, env_ids=env_ids)


def fallen(env: ManagerBasedRlEnv) -> np.ndarray:
    robot = cast("Entity", env.scene["robot"])
    return (robot.data.root_link_pos_w[:, 2] < 0.48) | (
        robot.data.projected_gravity_b[:, 2] > -0.65
    )


def upright(env: ManagerBasedRlEnv) -> np.ndarray:
    robot = cast("Entity", env.scene["robot"])
    return np.clip(-robot.data.projected_gravity_b[:, 2], 0, 1)
