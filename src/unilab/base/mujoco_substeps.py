"""Fork-only held-control trajectory adapter, isolated from task code (ADR-0010)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco.rollout import Rollout
from unisim.backend.mujoco.backend import MuJoCoBackend

from unilab.base.backend_constraints import EqualityConstraintBackend
from unilab.base.backend_substeps import SubstepObservationBackend, SubstepObserver

if TYPE_CHECKING:
    from mjbatch.held_control import HeldControlRollout
    from unisim.dr.types import ResetRandomizationPayload


class SubstepMuJoCoBackend(MuJoCoBackend, SubstepObservationBackend, EqualityConstraintBackend):
    _observer: SubstepObserver | None = None
    _recorder: Rollout | HeldControlRollout | None = None

    def __init__(self, *args, substep_engine="rollout", group_identical_models=False, **kwargs):
        if substep_engine not in ("rollout", "mjbatch"):
            raise ValueError("unknown substep engine")
        self.substep_engine = substep_engine
        self.group_identical_models = group_identical_models
        super().__init__(*args, **kwargs)

    def get_dr_capabilities(self):
        capabilities = super().get_dr_capabilities()
        if self.substep_engine == "mjbatch":
            return replace(capabilities, supported_reset_terms=frozenset())
        return capabilities

    def materialize(self) -> None:
        if self._post_step_forward_sensor or self._cpu_ids is not None:
            raise ValueError("substep observation requires solved sensors and unpinned workers")
        super().materialize()
        assert self._pool is not None
        models = self._pool.get_all_models()
        self._equality_names = tuple(models[0].eq(i).name for i in range(models[0].neq))
        self._equality_default = np.array([m.eq_active0 for m in models], dtype=bool)
        self._equality_active = self._equality_default.copy()
        workspace_model = max(self._pool.get_all_models(), key=lambda model: model.nbvh)
        if self.substep_engine == "mjbatch":
            from mjbatch.held_control import HeldControlRollout

            self._recorder = HeldControlRollout(
                self._pool.get_all_models(),
                num_threads=self._n_threads,
                group_identical_models=self.group_identical_models,
            )
        else:
            self._recorder = Rollout(nthread=self._n_threads)
        self._record_data = [mujoco.MjData(workspace_model) for _ in range(self._n_threads)]

    def get_equality_names(self) -> tuple[str, ...]:
        return self._equality_names

    def get_equality_active(self) -> np.ndarray:
        return self._equality_active.copy()

    def set_equality_active(self, env_indices: np.ndarray, active: np.ndarray) -> None:
        if active.dtype != np.bool_ or active.shape != (
            len(env_indices),
            len(self._equality_names),
        ):
            raise ValueError(
                "equality activation must be a boolean (selected envs, constraints) array"
            )
        self._equality_active[env_indices] = active

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict | None:
        result = super().set_state(env_indices, qpos, qvel, randomization=randomization)
        self._equality_active[env_indices] = self._equality_default[env_indices]
        return cast(dict | None, result)

    def set_substep_observer(
        self, sensor_names: Sequence[str], root_body_name: str, observer: SubstepObserver
    ) -> None:
        if self._observer is not None:
            raise ValueError("only one substep observer may be registered")
        self.bind_sensor_data(sensor_names)
        self._observe_sensors = np.array(
            [i for name in sensor_names for i in self._sensor_indices[name]], dtype=int
        )
        layout = self.get_root_state_layout(root_body_name)
        self._observe_velocity = self._idx_qvel + np.array(layout.qvel_indices[:3])
        self._observer = observer

    def set_pre_step_control(self, fn) -> None:
        if fn is not None:
            raise NotImplementedError("substep observation supports held control only")
        super().set_pre_step_control(fn)

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        if self._observer is None and self.substep_engine == "rollout" and not self._equality_names:
            return cast(dict | None, super().step(ctrl, nsteps))
        assert self._pool is not None and self._recorder is not None
        start = time.perf_counter()
        control = np.broadcast_to(ctrl[:, None, :], (self._num_envs, nsteps, ctrl.shape[-1]))
        spec = int(mujoco.mjtState.mjSTATE_CTRL)
        pending = bool(np.any(self._pending_xfrc_applied))
        if pending:
            spec |= int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
            wrench = np.broadcast_to(
                self._pending_xfrc_applied[:, None, :],
                (self._num_envs, nsteps, self._pending_xfrc_applied.shape[-1]),
            )
            control = np.concatenate((control, wrench), axis=-1)
        if self._equality_names:
            spec |= int(mujoco.mjtState.mjSTATE_EQ_ACTIVE)
            active = np.broadcast_to(
                self._equality_active[:, None, :],
                (self._num_envs, nsteps, len(self._equality_names)),
            )
            control = np.concatenate((control, active), axis=-1)
        prepared = time.perf_counter()
        states, sensors = self._recorder.rollout(
            self._pool.get_all_models(),
            self._record_data,
            self._physics_state,
            control,
            control_spec=spec,
            nstep=nsteps,
            chunk_size=self._chunk_size,
        )
        simulated = time.perf_counter()
        if pending:
            self._pending_xfrc_applied.fill(0)
        self._physics_state[:] = states[:, -1]
        self._sensor_data[:] = sensors[:, -1]
        if self._observer is not None:
            observed_sensors = sensors[:, :, self._observe_sensors]
            observed_velocity = states[:, :, self._observe_velocity]
            observed_sensors.setflags(write=False)
            observed_velocity.setflags(write=False)
            self._observer(observed_sensors, observed_velocity)
        return {
            "timing": {
                "set_ctrl_ms": (prepared - start) * 1000,
                "physics_ms": (simulated - prepared) * 1000,
                "refresh_cache_ms": (time.perf_counter() - simulated) * 1000,
            }
        }

    def cleanup_scene_assets(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        self._observer = None
        super().cleanup_scene_assets()
