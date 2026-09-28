"""Cold, regularized support allocation for the coordinated batting reference."""
from dataclasses import dataclass
import mujoco
import numpy as np
from scipy.optimize import nnls
from .articulated_swing import transfer_bat_wrench
from .prior import SDK_JOINTS

@dataclass
class SupportSolution:
    torque: np.ndarray
    loads: np.ndarray
    residual: np.ndarray
    foot_force: np.ndarray
    foot_clearance: np.ndarray
    optimality_error: float

class SupportAllocator:
    """Assumed planted feet, not measured contact forces or runtime root assistance."""

    def __init__(self, model, *, regularization=0.0001, length_scale=0.5):
        if regularization <= 0 or length_scale <= 0:
            raise ValueError('support regularization and length scale must be positive')
        self.model = model
        self.data = mujoco.MjData(model)
        self.regularization = regularization
        self.weights = np.r_[np.ones(3), np.full(3, 1 / length_scale)]
        self.dofs = model.jnt_dofadr[[model.joint(name).id for name in SDK_JOINTS]]

    def prepare(self, position):
        model, data = (self.model, self.data)
        data.qpos[:] = position
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        basis, directions, clearances = ([], [], [])
        pitch = model.geom('pitch').id
        for side in ('left', 'right'):
            for i in range(1, 8):
                geom = model.geom(f'{side}_foot{i}_collision')
                friction = min(model.geom_friction[geom.id, 0], model.geom_friction[pitch, 0])
                rays = np.array([[friction, 0, 1], [-friction, 0, 1], [0, friction, 1], [0, -friction, 1]])
                axis = data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
                for sign in (-1, 1):
                    point = data.geom_xpos[geom.id] + sign * geom.size[1] * axis
                    clearances.append(point[2] - geom.size[0])
                    point[2] = 0
                    jacobian = np.empty((3, model.nv))
                    mujoco.mj_jac(model, data, jacobian, None, point, int(geom.bodyid[0]))
                    basis.extend(rays @ jacobian)
                    directions.extend(rays)
        self.basis = np.asarray(basis)
        self.directions = np.asarray(directions).reshape(2, -1, 3)
        self.clearances = np.asarray(clearances).reshape(2, -1)
        self.matrix = self.weights[:, None] * self.basis[:, :6].T
        self.augmented = np.vstack((self.matrix, np.sqrt(self.regularization) * np.eye(len(self.basis))))

    def solve(self, velocity, acceleration):
        model, data = (self.model, self.data)
        data.qvel[:] = velocity
        mujoco.mj_forward(model, data)
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        required = transfer_bat_wrench(model, data, mass @ acceleration + data.qfrc_bias - data.qfrc_passive)
        rhs = self.weights * required[:6]
        loads, _ = nnls(self.augmented, np.r_[rhs, np.zeros(len(self.basis))], maxiter=2000)
        remaining = required - self.basis.T @ loads
        gradient = self.matrix.T @ (self.matrix @ loads - rhs) + self.regularization * loads
        kkt = np.where(loads > 1e-08, np.abs(gradient), np.maximum(-gradient, 0))
        return SupportSolution(torque=remaining[self.dofs], loads=loads, residual=remaining[:6], foot_force=np.sum(self.directions * loads.reshape(2, -1, 1), axis=1), foot_clearance=self.clearances.copy(), optimality_error=float(kkt.max()))
