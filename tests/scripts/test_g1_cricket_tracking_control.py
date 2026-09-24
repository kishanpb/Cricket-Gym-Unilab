"""Position-actuator encoding must reproduce the declared PD velocity term."""

import mujoco
import numpy as np
from g1_cricket_tracking_control_audit import position_velocity_control


def test_velocity_reference_cancels_damping_at_desired_motion():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody><body>
      <joint name="joint"/><geom type="sphere" size=".1"/>
    </body></worldbody><actuator><position joint="joint" kp="40" kv="10"/></actuator></mujoco>
    """)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = 0.4, 2.0
    data.ctrl[:] = position_velocity_control(model, data.qpos, data.qvel)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 0, atol=1e-12)
    np.testing.assert_array_equal(
        position_velocity_control(model, data.qpos, np.zeros(1)), data.qpos
    )
    data.qvel[:] = 1.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 10, atol=1e-12)
