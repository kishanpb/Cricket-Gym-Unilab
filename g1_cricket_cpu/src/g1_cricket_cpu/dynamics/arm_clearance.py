"""Bounded arm target projection away from the native torso and legs."""
import mujoco
import numpy as np
from scipy.optimize import least_squares

class ArmClearance:

    def __init__(self, model, joints, *, margin=0.012, limit=0.35):
        self.model = model
        self.data = mujoco.MjData(model)
        self.q = model.jnt_qposadr[joints]
        self.bounds = model.jnt_range[joints]
        self.margin, self.limit = (margin, limit)
        self.pairs = [(model.geom(f'{side}_{part}_collision').id, model.geom(body).id) for side in ('left', 'right') for part in ('hand', 'wrist') for body in (f'{side}_hip_collision', f'{side}_thigh_collision', 'torso_collision')]

    def distances(self):
        mujoco.mj_kinematics(self.model, self.data)
        return np.array([mujoco.mj_geomDistance(self.model, self.data, a, b, 0.1, None) for a, b in self.pairs])

    def project(self, state, target):
        self.data.qpos[:] = state.qpos
        lo = np.maximum(self.bounds[:, 0] - target, -self.limit)
        hi = np.minimum(self.bounds[:, 1] - target, self.limit)

        def residual(offset):
            self.data.qpos[self.q] = target + offset
            return np.r_[offset, 100 * np.maximum(self.margin - self.distances(), 0)]
        solved = least_squares(residual, np.zeros(len(target)), bounds=(lo, hi), max_nfev=25)
        residual(solved.x)
        diagnostic = np.r_[solved.success, solved.nfev, self.distances().min(), solved.x]
        return (target + solved.x, diagnostic)
