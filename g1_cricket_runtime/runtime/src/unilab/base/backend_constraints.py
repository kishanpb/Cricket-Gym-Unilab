"""Fork-only held equality-constraint capability; see ADR-0011."""

from abc import abstractmethod

import numpy as np
from unisim.backend.base import SimBackend


class EqualityConstraintBackend(SimBackend):
    @abstractmethod
    def get_equality_names(self) -> tuple[str, ...]:
        """Return model-order names on the cold path; unnamed constraints use ''."""

    @abstractmethod
    def get_equality_active(self) -> np.ndarray:
        """Copy the persistent (envs, constraints) boolean activation state.

        FULLPHYSICS does not include this state. Retained replay inputs must
        record it separately. This does not report constraint force or touch.
        """

    @abstractmethod
    def set_equality_active(self, env_indices: np.ndarray, active: np.ndarray) -> None:
        """Set booleans for selected envs, held until changed or set_state reset.

        Shape is (selected envs, all constraints). Only activation changes;
        positions, velocities and current sensor caches remain untouched.
        set_state resets selected rows to the model's default activation.
        Task logic owns one-way release latches and physical regrasp rules.
        """
