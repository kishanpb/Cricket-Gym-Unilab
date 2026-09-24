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

from unilab.base.backend_substeps import SubstepObservationBackend, SubstepObserver

if TYPE_CHECKING:
    from mjbatch.held_control import HeldControlRollout


class SubstepMuJoCoBackend(MuJoCoBackend, SubstepObservationBackend):
    _observer: SubstepObserver | None = None
    _recorder: Rollout | HeldControlRollout | None = None

    def __init__(self, *args, substep_engine="rollout", **kwargs):
        if substep_engine not in ("rollout", "mjbatch"):
            raise ValueError("unknown substep engine")
        self.substep_engine = substep_engine
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
        workspace_model = max(self._pool.get_all_models(), key=lambda model: model.nbvh)
        if self.substep_engine == "mjbatch":
            from mjbatch.held_control import HeldControlRollout

            self._recorder = HeldControlRollout(
                self._pool.get_all_models(), num_threads=self._n_threads
            )
        else:
            self._recorder = Rollout(nthread=self._n_threads)
        self._record_data = [mujoco.MjData(workspace_model) for _ in range(self._n_threads)]

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
        if self._observer is None and self.substep_engine == "rollout":
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
