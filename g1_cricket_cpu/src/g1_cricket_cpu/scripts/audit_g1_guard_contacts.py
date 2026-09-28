"""Read-only native substep audit of the saved loaded-guard actors, not batting."""
import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import mujoco
import numpy as np
import torch
from g1_cricket_cpu.scripts.evaluate_g1_guard_perturbations import CASES, make_owner
from g1_cricket_cpu.scripts.initialize_g1_guard_actor import ROOT, digest, rollout
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from scipy.spatial.transform import Rotation
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from g1_cricket_cpu.physics.articulated_learning import FINGER_JOINTS, ArticulatedBoundaryReward, G1ArticulatedBattingCfg
from g1_cricket_cpu.physics.prior import SDK_JOINTS
from g1_cricket_cpu.physics.scene import BALL_CONTACT_NAMES, CONTACT_SLOTS, CONTACT_WIDTH
from g1_cricket_cpu.physics.task import make_g1_cricket_env
from unilab.training import algo_config_dict
BODIES = ('pelvis', 'cricket_bat', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')
JOINTS = (*SDK_JOINTS, *FINGER_JOINTS)
FINGERS = tuple((f'{side}_{finger}' for side in ('left', 'right') for finger in ('thumb', 'index', 'middle')))
METRICS = ('root_height_m', 'root_tilt_rad', 'bat_translation_m', 'bat_rotation_rad', 'foot_translation_m', 'wrist_translation_m', 'joint_limit_excess_rad', 'handle_penetration_m', 'robot_self_penetration_m', 'bat_robot_penetration_m', 'body_torque_excess_nm', 'finger_torque_excess_nm', 'loaded_fingers', 'forbidden_contacts', 'peak_forbidden_normal_force_n')

@dataclass
class GuardAuditCfg(G1ArticulatedBattingCfg):
    audit_sensor_names: tuple[str, ...] = ()

    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        sensors = tree.getroot().find('sensor')
        names = []
        for body in BODIES:
            for kind in ('framepos', 'framequat'):
                name = f'audit_{kind}_{body}'
                ET.SubElement(sensors, kind, name=name, objtype='xbody', objname=body)
                names.append(name)
        for joint in JOINTS:
            name = f'audit_{joint}'
            ET.SubElement(sensors, 'jointpos', name=name, joint=joint)
            names.append(name)
        for name, selection in (('audit_robot_self', {'subtree1': 'pelvis', 'subtree2': 'pelvis'}), ('audit_bat_robot', {'body1': 'cricket_bat', 'subtree2': 'pelvis'})):
            ET.SubElement(sensors, 'contact', name=name, data='found force dist', num='64', reduce='none', **selection)
            names.append(name)
        self.audit_sensor_names = (*names, *(f'prior_force_{joint}' for joint in SDK_JOINTS), *(f'force_{joint}' for joint in FINGER_JOINTS), *self.grasp_sensor_names)
        tree.write(destination)
        return guards

def contact_rows(values, slots, width):
    contacts = values.reshape(len(values), slots, width)
    if np.any(contacts[..., 0] > slots):
        raise RuntimeError('guard audit contact sensor capacity exceeded')
    return (contacts, contacts[..., 0] > 0)

def penetration(contacts, active, distance_index):
    return np.maximum(0, np.where(active, -contacts[..., distance_index], 0).max(axis=1))

def checks(metrics, times):
    columns = dict(zip(METRICS, metrics.T, strict=True))
    settled = times > 0.2
    return {'upright': bool(columns['root_height_m'].min() > 0.6 and columns['root_tilt_rad'].max() < 0.35), 'retained_bat': bool(columns['bat_translation_m'].max() < 0.02 and columns['bat_rotation_rad'].max() < 0.1), 'planted_feet': bool(columns['foot_translation_m'].max() < 0.03), 'wrist_stability': bool(columns['wrist_translation_m'].max() < 0.02), 'joint_limits': bool(columns['joint_limit_excess_rad'].max() < 0.02), 'active_contact_geometry': bool(max((columns[k].max() for k in ('handle_penetration_m', 'robot_self_penetration_m', 'bat_robot_penetration_m'))) < 0.002), 'no_forbidden_contact': bool(columns['forbidden_contacts'].max() == 0), 'torque_caps': bool(max((columns[k].max() for k in ('body_torque_excess_nm', 'finger_torque_excess_nm'))) < 1e-10), 'loaded_fingers': bool(settled.any() and columns['loaded_fingers'][settled].min() >= 4)}

class GuardContactAudit(ArticulatedBoundaryReward):

    def sensor_names(self, env):
        return (*super().sensor_names(env), *env.cfg.audit_sensor_names)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.get_playback_model()
        original = super().sensor_names(env)
        self.original_width = sum((int(model.sensor(name).dim[0]) for name in original))
        self.slices = {}
        start = 0
        for name in env.cfg.audit_sensor_names:
            end = start + int(model.sensor(name).dim[0])
            self.slices[name] = slice(start, end)
            start = end
        self.dt = env.cfg.sim_dt
        self.grasp_names = env.cfg.grasp_sensor_names
        with np.load(env.cfg.reference_file) as saved:
            pose = saved['qpos'][0]
        data = mujoco.MjData(model)
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        self.positions = np.array([data.body(name).xpos for name in BODIES])
        self.quaternions = np.array([data.body(name).xquat for name in BODIES])
        self.limits = np.array([model.joint(name).range for name in JOINTS])
        action = env.action_manager.get_term('batting')
        control = action.controller
        self.caps = np.r_[model.actuator_forcerange[control.actuators, 1], control.fingers.force_limits[:, 1]]
        self.blocks = []

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.blocks.clear()

    def observe(self, sensors, integrated_velocity):
        original = sensors[..., :self.original_width]
        super().observe(original, integrated_velocity)
        if len(sensors) != 1:
            raise ValueError('guard audit retains one complete episode at a time')
        values = sensors[0, :, self.original_width:]
        if not np.isfinite(values).all():
            raise RuntimeError('nonfinite guard audit sensors')

        def read(name):
            return values[:, self.slices[name]]
        positions = np.stack([read(f'audit_framepos_{name}') for name in BODIES], axis=1)
        root_quat = read('audit_framequat_pelvis')
        reference = Rotation.from_quat(self.quaternions[0], scalar_first=True)
        tilt = (reference.inv() * Rotation.from_quat(root_quat, scalar_first=True)).as_rotvec()
        rotation = 2 * np.arccos(np.clip(np.abs(read('audit_framequat_cricket_bat') @ self.quaternions[1]), 0, 1))
        displacement = np.linalg.norm(positions - self.positions, axis=2)
        angles = np.column_stack([read(f'audit_{joint}') for joint in JOINTS])
        excess = np.maximum(0, np.maximum(self.limits[:, 0] - angles, angles - self.limits[:, 1]).max(axis=1))
        forces = np.column_stack([read(f'prior_force_{joint}') for joint in SDK_JOINTS] + [read(f'force_{joint}') for joint in FINGER_JOINTS])
        torque_excess = np.maximum(0, np.abs(forces) - self.caps)
        depths = []
        for name in ('audit_robot_self', 'audit_bat_robot'):
            contacts, active = contact_rows(read(name), 64, 5)
            depths.append(penetration(contacts, active, 4))
        normal = np.zeros((len(values), 6))
        tangent = np.zeros_like(normal)
        loaded = np.zeros_like(normal, dtype=bool)
        handle_depth = np.zeros(len(values))
        for name in self.grasp_names:
            contacts, active = contact_rows(read(name), 8, 17)
            handle_depth = np.maximum(handle_depth, penetration(contacts, active, 7))
            for index, finger in enumerate(FINGERS):
                side, digit = finger.split('_')
                if f'{side}_hand_{digit}_' in name:
                    normal[:, index] += np.where(active, contacts[..., 1], 0).sum(axis=1)
                    tangent[:, index] += np.where(active, np.linalg.norm(contacts[..., 2:4], axis=2), 0).sum(axis=1)
                    loaded[:, index] |= (active & (contacts[..., 1] > 0.1)).any(axis=1)
        base_width = len(BALL_CONTACT_NAMES) * CONTACT_SLOTS * CONTACT_WIDTH + 3
        guards = original[0, :, base_width:base_width + self.guard_width].reshape(len(values), -1, CONTACT_WIDTH)
        active = guards[..., 0] > 0
        forbidden = active.sum(axis=1)
        peak_forbidden = np.where(active, guards[..., 1], 0).max(axis=1)
        metrics = np.column_stack((positions[:, 0, 2], np.linalg.norm(tilt[:, :2], axis=1), displacement[:, 1], rotation, displacement[:, 2:4].max(axis=1), displacement[:, 4:6].max(axis=1), excess, handle_depth, *depths, torque_excess[:, :29].max(axis=1), torque_excess[:, 29:].max(axis=1), loaded.sum(axis=1), forbidden, peak_forbidden))
        self.blocks.append((metrics, normal, tangent, forces))

    def arrays(self):
        metrics, normal, tangent, forces = (np.concatenate(parts) for parts in zip(*self.blocks, strict=True))
        return dict(contact_time_s=np.arange(len(metrics)) * self.dt, metrics=metrics, finger_normal_force_n=normal, finger_tangent_force_magnitudes_n=tangent, actuator_force_nm=forces)

def make_audit_env(owner):
    OmegaConf.update(owner, 'reward.boundary.func', 'g1_cricket_cpu.scripts.audit_g1_guard_contacts.GuardContactAudit')
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override['auto_reset'] = False
    cfg = GuardAuditCfg()
    apply_cfg_overrides(cfg, override)
    return make_g1_cricket_env(cfg, num_envs=1, backend_type='mujoco')

def evaluate(output, parent, baseline):
    prior = json.loads((baseline / 'summary.json').read_text())
    fingerprints = dict(prior['source_input_sha256'])
    for name, expected in prior['artifact_sha256'].items():
        fingerprints[str((baseline / name).relative_to(ROOT))] = expected
    for path in (Path(__file__), baseline / 'summary.json', parent / 'right_actor.pt', parent / 'left_actor.pt'):
        fingerprints[str(path.relative_to(ROOT))] = digest(path)
    for path, expected in fingerprints.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'source/input changed: {path}')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for hand in ('right', 'left'):
        for label, velocity in CASES:
            owner = make_owner(hand, velocity)
            env = make_audit_env(owner)
            try:
                torch.manual_seed(1)
                wrapper = RslRlVecEnvWrapper(env, device='cpu')
                config = normalize_ppo_train_cfg(algo_config_dict(owner))
                config['logger'] = 'none'
                runner = OnPolicyRunner(wrapper, config, log_dir=None, device='cpu')
                runner.alg.actor.load_state_dict(torch.load(parent / f'{hand}_actor.pt', weights_only=True))
                result, trace = rollout(env, wrapper, runner.alg.actor)
                with np.load(baseline / f'{hand}_{label}.npz') as old:
                    for key, value in trace.items():
                        np.testing.assert_array_equal(value, old[key], err_msg=f'instrumentation changed {hand}/{label}/{key}')
                audit = env.reward_manager.get_term_cfg('boundary').func
                arrays = audit.arrays()
                np.savez_compressed(output / f'{hand}_{label}.npz', **arrays)
                gates = checks(arrays['metrics'], arrays['contact_time_s'])
                row = dict(hand=hand, case=label, result=result, parent_trace_bit_exact=True, substeps=len(arrays['metrics']), checks=gates, passed=all(gates.values()), minimum=dict(zip(METRICS, arrays['metrics'].min(axis=0).tolist(), strict=True)), maximum=dict(zip(METRICS, arrays['metrics'].max(axis=0).tolist(), strict=True)))
                rows.append(row)
                print(json.dumps(row), flush=True)
            finally:
                env.close()
    for path, expected in fingerprints.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'source/input changed during evaluation: {path}')
    report = {'scope': 'native substep active-contact and static guard audit, not batting or PPO success', 'protocol': 'all ten unchanged fixed-start cases, final guard actors, native solved contact forces, no training', 'timing': 'kinematic sensors and solved forces at preintegration contact_time_s; endpoint states separately compared bit-exact to the parent', 'contact_scope': 'force-producing contacts only; inactive raw contacts and slip velocity are not measured by this sensor audit', 'assistance': 'learned leg actor, fixed assisted waist/arm guard targets, per-substep finger PD; no live pose writes or bat weld', 'all_cases_passed': all((row['passed'] for row in rows)), 'metric_columns': METRICS, 'finger_columns': FINGERS, 'actuator_columns': JOINTS, 'source_input_sha256': fingerprints, 'rows': rows, 'artifact_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--parent', type=Path, default=ROOT / 'g1_cricket_results/guard_aggregation_v1')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'g1_cricket_results/guard_perturbations_v1')
    args = parser.parse_args()
    evaluate(args.output.resolve(), args.parent.resolve(), args.baseline.resolve())
