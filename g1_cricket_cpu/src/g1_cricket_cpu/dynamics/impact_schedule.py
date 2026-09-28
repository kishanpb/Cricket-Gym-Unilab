"""Quantized local refinement for native ball/blade impacts."""
import mujoco
import numpy as np
COARSE_DT = 0.000125
FINE_DT = COARSE_DT / 8
POLICY_TICKS = round(0.02 / FINE_DT)

def blade_gap(model, data):
    ball, blade = (model.geom('ball_geom').id, model.geom('bat_blade').id)
    local = data.geom_xmat[blade].reshape(3, 3).T @ (data.geom_xpos[ball] - data.geom_xpos[blade])
    outside = np.maximum(np.abs(local) - model.geom_size[blade], 0)
    return float(np.linalg.norm(outside) - model.geom_size[ball, 0])

class ImpactSchedule:

    def __init__(self, coarse_substeps=1, coarse_integrator='RK4'):
        if coarse_substeps not in (1, 2):
            raise ValueError('Coarse substeps must be one or two')
        self.coarse_stride = 8 // coarse_substeps
        if coarse_integrator not in ('RK4', 'implicitfast'):
            raise ValueError('Coarse integrator must be RK4 or implicitfast')
        self.coarse_integrator = coarse_integrator
        self.fine_steps = self.coarse_steps = 0
        self.alignment_steps = 0
        self.fine_seconds = 0.0

    def __call__(self, model, data, duration):
        end = round(duration / FINE_DT)
        if not np.isclose(end * FINE_DT, duration, rtol=0, atol=1e-12):
            raise ValueError('Episode duration must align with the fine clock')
        tick = 0
        while tick < end:
            near = blade_gap(model, data) < 0.15
            stride = 1 if near else self.coarse_stride
            stride = min(stride, end - tick, POLICY_TICKS - tick % POLICY_TICKS)
            model.opt.timestep = stride * FINE_DT
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4 if near or self.coarse_integrator == 'RK4' else mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            tick += stride
            if stride == self.coarse_stride:
                self.coarse_steps += 1
            elif stride == 1:
                self.fine_steps += 1
                self.fine_seconds += model.opt.timestep
            else:
                self.alignment_steps += 1
            yield (tick % POLICY_TICKS == 0)

    def report(self):
        return {'fine_steps': self.fine_steps, 'coarse_steps': self.coarse_steps, 'alignment_steps': self.alignment_steps, 'fine_seconds': self.fine_seconds, 'fine_dt_s': FINE_DT, 'coarse_dt_s': self.coarse_stride * FINE_DT, 'coarse_integrator': self.coarse_integrator, 'fine_integrator': 'RK4', 'refinement_gap_m': 0.15, 'scope': 'Refine near the blade only; pitch and other contacts retain the coarse timestep.'}
