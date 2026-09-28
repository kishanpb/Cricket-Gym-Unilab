"""Bounded wrist-target feedback around a measured, unconstrained bat."""
import mujoco
import numpy as np
from g1_cricket_cpu.physics.shared_bat_pose import SharedBatPoseIK

class BatRelativeGripFeedback:

    def __init__(self, model, initial_pose, gain, *, include_waist=False):
        if not np.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError('Grip feedback gain must be finite and between zero and one')
        self.gain = gain
        self.ik = SharedBatPoseIK(model, joint_offset_limit=0.1, include_waist=include_waist)
        data = self.ik.data
        data.qpos[:] = initial_pose
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[self.ik.bat].reshape(3, 3)
        self.positions = (data.xpos[self.ik.wrists] - data.xpos[self.ik.bat]) @ rotation
        self.rotations = rotation.T @ data.xmat[self.ik.wrists].reshape(2, 3, 3)
        self.offset = np.zeros(len(self.ik.q))
        self.last_tick = -1
        self.rows = []

    def update(self, state):
        tick = int(np.floor((state.time + 1e-10) / 0.02))
        if tick == self.last_tick:
            return self.offset
        self.last_tick = tick
        geometry = self.ik.data
        geometry.qpos[:] = state.qpos
        mujoco.mj_kinematics(self.ik.model, geometry)
        rotation = geometry.xmat[self.ik.bat].reshape(3, 3).copy()
        positions = geometry.xpos[self.ik.bat] + self.positions @ rotation.T
        rotations = rotation @ self.rotations
        delta, diagnostic = self.ik.solve_targets(state.qpos, positions, rotations)
        requested = np.clip(self.gain * delta, -0.05, 0.05)
        self.offset += np.clip(requested - self.offset, -0.005, 0.005)
        self.rows.append({'time_s': float(state.time), 'offset_rad': self.offset.tolist(), **diagnostic})
        return self.offset
