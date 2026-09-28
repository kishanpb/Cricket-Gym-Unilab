"""CPU transfer of released GR00T balance control to native cricket grasps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import torch
from g1_cricket_cpu.dynamics.twist2_cpu_probe import fingerprint
from g1_cricket_cpu.dynamics.grip_feedback import BatRelativeGripFeedback
from g1_cricket_cpu.scripts.evaluate_g1_articulated_swing import run_case
from g1_cricket_cpu.physics.articulated_hands import FingerController
from g1_cricket_cpu.physics.articulated_swing import ArticulatedSwingController
from g1_cricket_cpu.physics.prior import SDK_JOINTS
from g1_cricket_cpu.physics.tracking import root_position_balance
UPSTREAM_COMMIT = 'b042411fae38ee4d1af9aac82a37a1f8d14d6dd0'
CHECKPOINTS = {'Balance': 'f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18', 'Walk': '7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407'}
CONFIG = 'decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml'

def load_policy(upstream):
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('GR00T checkout must match the reviewed revision')
    for name, expected in CHECKPOINTS.items():
        path = upstream / Path(CONFIG).parent / 'policy' / f'GR00T-WholeBodyControl-{name}.onnx'
        if fingerprint(path)['sha256'] != expected:
            raise ValueError(f'Unverified {name} checkpoint (possibly a Git LFS pointer)')
    sys.path.insert(0, str(upstream.resolve()))
    from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

    class CpuPolicy(G1GearWbcPolicy):

        def load_onnx_policy(self, model_path):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
            if session.get_inputs()[0].shape != ['batch_size', 516]:
                raise ValueError('Expected six 86-value observations')

            def infer(value):
                result = session.run(None, {session.get_inputs()[0].name: value.numpy()})[0]
                return torch.from_numpy(result)
            return infer
    groups = {'body': list(range(29)), 'lower_body': list(range(15))}
    robot = SimpleNamespace(get_joint_group_indices=groups.__getitem__)
    policy = CpuPolicy(robot, str(upstream / CONFIG), 'policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx')
    policy.set_use_teleop_policy_cmd(True)
    policy.use_policy_action = True
    return policy

class NativeFingerServo(FingerController):
    """The same capped PD law, evaluated by MuJoCo inside the integrator."""

    def __init__(self, model, *, kp, kd):
        super().__init__(model, kp=kp, kd=kd)
        ids = self.actuators
        model.actuator_gaintype[ids] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[ids] = 0
        model.actuator_gainprm[ids, 0] = kp
        model.actuator_biasprm[ids] = 0
        model.actuator_biasprm[ids, 1] = -kp
        model.actuator_biasprm[ids, 2] = -kd
        model.actuator_ctrllimited[ids] = True
        model.actuator_ctrlrange[ids] = self.limits
        model.actuator_forcelimited[ids] = True
        model.actuator_forcerange[ids] = self.force_limits

    def apply(self, data, target):
        data.ctrl[self.actuators] = np.clip(target, self.limits[:, 0], self.limits[:, 1])

class GrootSwingController(ArticulatedSwingController):

    def __init__(self, model, poses, times, policy, *, reference_waist=False, root_feedback_gain=0.0, finger_preload=0.2, finger_damping=0.02, finger_stiffness=2.0, time_based_policy=False, native_finger_servo=False, grip_feedback_gain=0.0, grip_feedback_waist=False):
        if grip_feedback_waist and (not reference_waist):
            raise ValueError('Waist grip feedback requires reference waist control')
        if not np.isfinite(grip_feedback_gain) or not 0 <= grip_feedback_gain <= 1:
            raise ValueError('Grip feedback gain must be finite and between zero and one')
        if not np.isfinite(finger_preload) or not 0 <= finger_preload <= 0.6:
            raise ValueError('Finger preload must be finite and between zero and 0.6')
        super().__init__(model, poses, times, inertial=False, preload=finger_preload)
        damping_limit = 0.3 if native_finger_servo else 0.08
        if not np.isfinite(finger_damping) or not 0.02 <= finger_damping <= damping_limit:
            raise ValueError(f'Finger damping must be finite and between 0.02 and {damping_limit}')
        self.fingers.kd = finger_damping
        stiffness_limit = 6.0 if native_finger_servo else 3.0
        if not np.isfinite(finger_stiffness) or not 2 <= finger_stiffness <= stiffness_limit:
            raise ValueError(f'Finger stiffness must be finite and between 2 and {stiffness_limit}')
        self.fingers.kp = finger_stiffness
        if native_finger_servo:
            self.fingers = NativeFingerServo(model, kp=finger_stiffness, kd=finger_damping)
        self.policy = policy
        if not np.isfinite(root_feedback_gain) or not 0 <= root_feedback_gain <= 4:
            raise ValueError('Root feedback gain must be finite and between zero and four')
        self.root_feedback_gain = root_feedback_gain
        self.v = model.jnt_dofadr[[model.joint(name).id for name in SDK_JOINTS]]
        self.controlled_count = 12 if reference_waist else 15
        self.ids = self.actuators[:self.controlled_count]
        self.kp = np.asarray(policy.config['kps'])[:self.controlled_count]
        self.kd = np.asarray(policy.config['kds'])[:self.controlled_count]
        self.caps = model.actuator_forcerange[self.ids, 1].copy()
        self.native_kp = model.actuator_gainprm[self.ids, 0].copy()
        self.native_kd = -model.actuator_biasprm[self.ids, 2].copy()
        np.testing.assert_array_equal(model.actuator_biasprm[self.ids, 1], -self.native_kp)
        np.testing.assert_array_equal(model.actuator_gear[self.ids], np.tile([1, 0, 0, 0, 0, 0], (self.controlled_count, 1)))
        self.period = round(0.02 / model.opt.timestep)
        if not np.isclose(self.period * model.opt.timestep, 0.02, rtol=0, atol=1e-12):
            raise ValueError('Physics timestep must divide the 50 Hz policy period')
        reference_data = mujoco.MjData(model)
        torso = model.body('torso_link').id
        self.torso_rpy = []
        for pose in poses:
            reference_data.qpos[:] = pose
            mujoco.mj_kinematics(model, reference_data)
            root = Rotation.from_quat(pose[3:7], scalar_first=True)
            rotation = root.inv() * Rotation.from_matrix(reference_data.xmat[torso].reshape(3, 3))
            self.torso_rpy.append(rotation.as_euler('xyz'))
        self.torso_rpy = np.asarray(self.torso_rpy)
        self.step = 0
        self.time_based_policy = time_based_policy
        self.last_policy_tick = -1
        self.target = poses[0, self.q[:15]].copy()
        self.leg_target_offset = np.zeros(12)
        self.policy_rows = []
        self.grip_feedback_start = 12 if grip_feedback_waist else 15
        self.grip_feedback = BatRelativeGripFeedback(model, poses[0], grip_feedback_gain, include_waist=grip_feedback_waist) if grip_feedback_gain else None

    def apply(self, data):
        super().apply(data)
        tick = int(np.floor((data.time + 1e-10) / 0.02))
        policy_due = tick != self.last_policy_tick if self.time_based_policy else self.step % self.period == 0
        if policy_due:
            self.last_policy_tick = tick
            self.policy.set_observation({'q': data.qpos[self.q].copy(), 'dq': data.qvel[self.v].copy(), 'floating_base_pose': data.qpos[:7].copy(), 'floating_base_vel': data.qvel[:6].copy()})
            action = self.policy.get_action(time=data.time, base_height_command=float(np.interp(data.time, self.times, self.poses[:, 2])), torso_orientation_rpy=np.array([np.interp(data.time, self.times, column) for column in self.torso_rpy.T]), interpolated_navigate_cmd=np.zeros(3))['body_action'][0]
            if not np.isfinite(action).all():
                raise ValueError('Nonfinite GR00T target')
            self.target = np.clip(action, self.limits[:15, 0], self.limits[:15, 1])
            self.policy_rows.append(np.r_[data.time, self.policy.obs_buffer, self.policy.action, self.target])
        count = self.controlled_count
        target = self.target[:count].copy()
        if np.any(self.leg_target_offset):
            target[:12] = np.clip(target[:12] + self.leg_target_offset, self.limits[:12, 0], self.limits[:12, 1])
        if self.root_feedback_gain:
            index = int(np.clip(np.searchsorted(self.times, data.time, side='right') - 1, 0, len(self.times) - 2))
            duration = self.times[index + 1] - self.times[index]
            blend = np.clip((data.time - self.times[index]) / duration, 0, 1)
            before, after = (self.poses[index, :3], self.poses[index + 1, :3])
            position = (1 - blend) * before + blend * after
            velocity = (after - before) / duration
            correction = np.clip(root_position_balance(self.poses[0, 3:7], data.qpos[:3] - position, data.qvel[:3] - velocity, self.root_feedback_gain), -0.05, 0.05)
            target[[4, 10]] += correction[1]
            target[[5, 11]] += correction[0]
            target = np.clip(target, self.limits[:count, 0], self.limits[:count, 1])
        torque = self.kp * (target - data.qpos[self.q[:count]]) - self.kd * data.qvel[self.v[:count]]
        torque = np.clip(torque, -self.caps, self.caps)
        data.ctrl[self.ids] = data.qpos[self.q[:count]] + (torque + self.native_kd * data.qvel[self.v[:count]]) / self.native_kp
        if self.grip_feedback is not None:
            start = self.grip_feedback_start
            joints = self.actuators[start:]
            data.ctrl[joints] = np.clip(data.ctrl[joints] + self.grip_feedback.update(data), self.limits[start:, 0], self.limits[start:, 1])
        self.step += 1

def evaluate(upstream, checkout, output, *, reference_waist=False, root_feedback_gain=0.0, finger_preload=0.2):
    output.mkdir(parents=True, exist_ok=False)
    scene = checkout / 'g1_cricket_results/articulated_stance/scene.xml'
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.000125
    if model.nu != 43 or model.neq != 0:
        raise ValueError('Expected the free-body articulated cricket model')
    inputs = [scene, Path(__file__), Path('integrations/g1_dynamics/twist2_cpu_probe.py')]
    inputs += [checkout / f'scripts/{name}.py' for name in ('evaluate_g1_articulated_swing', 'evaluate_g1_articulated_stance')]
    inputs += sorted((checkout / 'src/unilab/tasks/manipulation/g1_cricket').glob('*.py'))
    inputs += [upstream / CONFIG]
    inputs += [upstream / f'decoupled_wbc/{path}' for path in ('control/policy/g1_gear_wbc_policy.py', 'control/policy/g1_decoupled_whole_body_policy.py', 'control/robot_model/supplemental_info/g1/g1_supplemental_info.py', 'control/base/policy.py', 'control/utils/gear_wbc_utils.py', 'sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx', 'sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx', 'sim2mujoco/resources/robots/g1/policy/NVIDIA Open Model License')]
    references = {hand: checkout / f'g1_cricket_results/articulated_swing_compact/{hand}_reference.npz' for hand in ('right', 'left')}
    inputs += list(references.values())
    sources = {str(path): fingerprint(path) for path in inputs}
    rows = []
    for hand, path in references.items():
        with np.load(path) as saved:
            poses = np.vstack([saved['qpos'], saved['qpos'][-1]])
            times = np.r_[saved['times'], 8.0]
        controller = GrootSwingController(model, poses, times, load_policy(upstream), reference_waist=reference_waist, root_feedback_gain=root_feedback_gain, finger_preload=finger_preload)
        result, states, controls, tactile = run_case(model, poses, times, inertial=False, controller=controller)
        trace = output / f'{hand}.npz'
        np.savez_compressed(trace, states=states, controls=controls, policy=controller.policy_rows)
        contact = output / f'{hand}_contacts.json'
        contact.write_text(json.dumps(tactile, allow_nan=False) + '\n')
        row = {'hand': hand, **result, 'trace': {'path': str(trace), **fingerprint(trace)}, 'contacts': {'path': str(contact), **fingerprint(contact)}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    for path, expected in sources.items():
        if fingerprint(Path(path)) != expected:
            raise ValueError(f'Source/input changed during trial: {path}')
    report = {'scope': 'Frozen released balance policy; no incoming delivery, training, promotion or bowling claim', 'cohort': 'Both hands, complete eight-second swing/recovery, no selected windows', 'control': 'Released policy legs only; native reference waist/arms/fingers; native caps and joint-target limits retained' if reference_waist else 'Released 15-action legs/waist policy, native arms/fingers; native caps and joint-target limits retained', 'reference_waist': reference_waist, 'root_feedback_gain': root_feedback_gain, 'finger_preload_rad': finger_preload, 'root_feedback_angle_limit_rad': 0.05, 'policy_trace': 'time, 516 observations, all 15 raw policy actions, all 15 bounded policy targets before optional ankle feedback; reference-waist mode applies only the first 12 targets and retains upstream raw-action history; controls trace records applied servo inputs', 'upstream_commit': UPSTREAM_COMMIT, 'physics_dt_s': model.opt.timestep, 'policy_hz': 50, 'source_input_fingerprints': sources, 'rows': rows}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-waist', action='store_true')
    parser.add_argument('--root-feedback-gain', type=float, default=0.0)
    parser.add_argument('--finger-preload', type=float, choices=(0.2, 0.4, 0.6), default=0.2)
    args = parser.parse_args()
    evaluate(args.upstream, args.checkout, args.output, reference_waist=args.reference_waist, root_feedback_gain=args.root_feedback_gain, finger_preload=args.finger_preload)
