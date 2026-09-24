"""Smooth bounded arm residuals; the frozen prior and physical limits are unchanged."""

from dataclasses import dataclass

import numpy as np

from .residual import FrozenPriorResidual, FrozenPriorResidualCfg


@dataclass(kw_only=True)
class TanhPriorResidualCfg(FrozenPriorResidualCfg):
    def build(self, env):
        return TanhPriorResidual(self, env)


class TanhPriorResidual(FrozenPriorResidual):
    def process_actions(self, actions):
        super().process_actions(np.tanh(actions))
        self._raw[:] = actions
