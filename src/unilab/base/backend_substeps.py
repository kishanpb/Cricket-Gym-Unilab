"""Experimental read-only interval observation contract; see ADR-0010."""

from abc import abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
from unisim.backend.base import SimBackend

SubstepObserver = Callable[[np.ndarray, np.ndarray], None]


class SubstepObservationBackend(SimBackend):
    @abstractmethod
    def set_substep_observer(
        self, sensor_names: Sequence[str], root_body_name: str, observer: SubstepObserver
    ) -> None:
        """Bind one read-only observer on the cold path.

        Once per physics interval, before returning to the env, deliver
        ``(solved_sensors, integrated_root_velocity_world)`` with shapes
        ``(envs, substeps, channels)`` and ``(envs, substeps, 3)``. Sensors
        retain the solver phase; root velocity follows that step's integration.
        Every completed substep is included, without an initial-state sample.
        Inputs are read-only and must be copied if retained after the callback.
        The observer must not write state or control. Reset is not an observation.
        """
