import mujoco
import numpy as np

from integrations.g1_dynamics.maximum_effort import maximum_effort, servo_effort_target


def test_request_sign_and_asymmetric_native_limits():
    request = np.array([-0.01, 0., 0.01, -1000., 1000.])
    limits = np.tile([-3., 5.], (5, 1))
    np.testing.assert_array_equal(maximum_effort(request, limits), [-3, 0, 5, -3, 5])
    np.testing.assert_array_equal(request, [-.01, 0, .01, -1000, 1000])


def test_servo_conversion_respects_native_force_and_control_limits():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="0 0 0"/><worldbody><body><joint name="j" type="hinge"/>
      <geom type="sphere" size=".1"/></body></worldbody>
      <actuator><position joint="j" kp="20" kv="2" forcerange="-3 5"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0], data.ctrl[0] = .2, -.1, .3
    ids = np.array([0])
    before = np.r_[data.qpos, data.qvel].copy()
    target, request, torque = servo_effort_target(model, data, ids)
    np.testing.assert_allclose(request, [2.2])
    np.testing.assert_array_equal(torque, [5])
    data.ctrl[ids] = target
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, [5])
    np.testing.assert_array_equal(np.r_[data.qpos, data.qvel], before)
    assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
    model.actuator_ctrllimited[0] = True
    model.actuator_ctrlrange[0] = [-.25, .25]
    target, _, _ = servo_effort_target(model, data, ids)
    np.testing.assert_array_equal(target, [.25])
