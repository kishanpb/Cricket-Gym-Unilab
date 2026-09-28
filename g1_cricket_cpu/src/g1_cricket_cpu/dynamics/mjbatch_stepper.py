"""Native mjbatch stepping for controllers that inspect MjData each substep."""
from pathlib import Path
import mujoco

class MjBatchStepper:
    """Transfer live integration state, never reference poses, across API buffers.

    The caller retains its existing post-step mj_forward/sensor convention.
    One simulation is used here; this adapter makes no batching speedup claim.
    """

    def __init__(self, model):
        from mjbatch import Batch
        self.model = model
        self.batch = Batch(model, 1, num_threads=1, forward=False)
        self.state = self.batch.bind('state')[0]
        self.warning = self.batch.bind('warning')[0]
        self.timestep = self.batch.expand('timestep')
        self.integrator = self.batch.expand('integrator')

    def __call__(self, data):
        self.timestep[0] = self.model.opt.timestep
        self.integrator[0] = self.model.opt.integrator
        mujoco.mj_getState(self.model, data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        self.warning[:, 0] = data.warning.lastinfo
        self.warning[:, 1] = data.warning.number
        self.batch.step()
        mujoco.mj_setState(self.model, data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        data.warning.lastinfo[:] = self.warning[:, 0]
        data.warning.number[:] = self.warning[:, 1]

def backend_inputs():
    import mjbatch
    from mjbatch import _bindings
    package = Path(mjbatch.__file__).resolve().parent
    return [Path(mjbatch.__file__).resolve(), Path(_bindings.__file__).resolve(), *sorted((package / 'csrc').glob('*.h')), *sorted((package / 'csrc').glob('*.cpp'))]
