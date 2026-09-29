"""Native-limit swing intervention; class exported from the full-cohort experiment."""

import numpy as np

from .groot_batting_env import GrootBattingEnv
from .maximum_effort import servo_effort_target


class MaximumEffortBattingEnv(GrootBattingEnv):
    def __init__(self, *args, maximum_arm_effort=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.maximum_arm_effort = maximum_arm_effort

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        controller = self.controller
        self.effort_ids = controller.actuators[15:29]
        self.effort_count = 0
        self.effort_peak = np.zeros((2, 14))
        self.effort_interval = []
        moving = np.max(np.abs(np.diff(controller.poses[:, controller.q[15:29]], axis=0)), axis=1) > 1e-6
        original_apply = controller.apply

        def apply(data):
            original_apply(data)
            index = np.searchsorted(controller.times, data.time, side="right") - 1
            if not self.maximum_arm_effort or not 0 <= index < len(moving) or not moving[index]:
                return
            target, requested, capped = servo_effort_target(self.model, data, self.effort_ids)
            data.ctrl[self.effort_ids] = target
            self.effort_count += 1
            self.effort_peak = np.maximum(self.effort_peak, np.abs([requested, capped]))
            if not self.effort_interval:
                self.effort_interval = [float(data.time), float(data.time)]
            self.effort_interval[-1] = float(data.time)

        controller.apply = apply
        return observation, info

    def step(self, action):
        result = super().step(action)
        if result[2]:
            self.result["maximum_arm_effort"] = dict(
                enabled=self.maximum_arm_effort, controller_calls=self.effort_count,
                active_interval_s=self.effort_interval,
                peak_original_request_nm=self.effort_peak[0].tolist(),
                peak_capped_request_nm=self.effort_peak[1].tolist(),
                actuator_names=[self.model.actuator(int(i)).name for i in self.effort_ids],
                native_limits_nm=self.model.actuator_forcerange[self.effort_ids].tolist(),
                scope="Requested effort; physical force limits and contact audits remain separate",
            )
        return result
