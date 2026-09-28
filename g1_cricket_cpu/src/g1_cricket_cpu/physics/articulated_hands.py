"""Official G1 finger mechanics; separate from the legacy palm fixtures."""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
HAND_PARTS = ('thumb_0', 'thumb_1', 'thumb_2', 'index_0', 'index_1', 'middle_0', 'middle_1')

def hand_joints(side):
    return [f'{side}_hand_{part}_joint' for part in HAND_PARTS]

def load_hand_assets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, row in manifest['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'modified hand asset: {name}')
    return ET.parse(directory / 'g1_hands.xml').getroot()

def attach_hand(root, wrist, side, source, directory):
    """Replace the rubber hand, preserving the parent wrist joint and transform."""
    official = source.find(f".//body[@name='{side}_wrist_yaw_link']")
    asset, actuator = (root.find('asset'), root.find('actuator'))
    assert official is not None and asset is not None and (actuator is not None)
    for child in list(wrist):
        if child.tag == 'inertial' or (child.tag == 'geom' and (child.get('mesh') == f'{side}_rubber_hand' or child.get('name') == f'{side}_hand_collision')):
            wrist.remove(child)
    wrist.insert(0, copy.deepcopy(official.find('inertial')))
    for child in official:
        if child.tag == 'body' or (child.tag == 'geom' and child.get('mesh') == f'{side}_hand_palm_link'):
            wrist.append(copy.deepcopy(child))
    meshes = {geom.get('mesh') for geom in wrist.iter('geom') if '_hand_' in geom.get('mesh', '')}
    for name in sorted(meshes):
        mesh = copy.deepcopy(source.find(f"asset/mesh[@name='{name}']"))
        mesh.set('file', str((Path(directory) / 'meshes' / mesh.attrib['file']).resolve()))
        asset.append(mesh)
    for body in wrist.iter('body'):
        for geom in body.findall('geom'):
            if '_hand_' in geom.get('mesh', '') or '_hand_' in body.get('name', ''):
                visual = geom.get('contype') == '0'
                geom.set('name', f"{body.get('name')}_hand_geom_{(0 if visual else 1)}")
                geom.set('group', '2' if visual else '3')
                if not visual:
                    geom.set('type', geom.get('type', 'sphere'))
                    geom.set('friction', '1 0.005 0.0001')
    for name in hand_joints(side):
        joint = wrist.find(f".//joint[@name='{name}']")
        joint.set('armature', '0')
        joint.set('frictionloss', '0')
        joint.set('damping', '0')
        ET.SubElement(actuator, 'motor', name=name, joint=name, ctrllimited='true', ctrlrange=joint.attrib['actuatorfrcrange'])

def build_grasp_bench(directory, destination, *, timestep=0.00025, contact_time_constant=0.02, batting_hand='right'):
    """Two fixed wrists and an unconstrained 0.7 kg bat: retention, not balance."""
    source = load_hand_assets(directory)
    root = ET.Element('mujoco', model='g1_articulated_grasp_bench')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep=str(timestep), integrator='implicitfast')
    ET.SubElement(root, 'asset')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'headlight', ambient='0.6 0.6 0.6', diffuse='0.8 0.8 0.8')
    ET.SubElement(root.find('asset'), 'texture', name='sky', type='skybox', builtin='gradient', rgb1='0.55 0.65 0.75', rgb2='0.8 0.86 0.9', width='512', height='3072')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(root, 'actuator')
    ET.SubElement(world, 'light', pos='0 -2 3', dir='0 1 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 0.05', rgba='0.18 0.35 0.24 1')
    if batting_hand not in {'right', 'left'}:
        raise ValueError('batting_hand must be right or left')
    left_z, right_z = (1.16, 1.05) if batting_hand == 'right' else (1.05, 1.16)
    for side, pos, quat in (('left', f'-0.135 0.03 {left_z}', '1 0 0 0'), ('right', f'0.135 0.03 {right_z}', '0 0 0 1')):
        wrist = ET.SubElement(world, 'body', name=f'{side}_wrist_yaw_link', pos=pos, quat=quat)
        attach_hand(root, wrist, side, source, directory)
    bat = ET.SubElement(world, 'body', name='cricket_bat', pos='0 0 1.2')
    ET.SubElement(bat, 'freejoint', name='bat_free')
    ET.SubElement(bat, 'geom', name='bat_handle', type='capsule', fromto='0 0 0 0 0 -0.2', size='0.014', mass='0.12', rgba='0.15 0.18 0.22 1')
    ET.SubElement(bat, 'geom', name='bat_blade', type='box', pos='0 0 -0.4', size='0.025 0.055 0.20', mass='0.58', rgba='0.85 0.71 0.43 1', friction='0.6 0.01 0.001')
    contact = ET.SubElement(root, 'contact')
    for side in ('left', 'right'):
        wrist = world.find(f"body[@name='{side}_wrist_yaw_link']")
        for child in wrist.findall('body'):
            ET.SubElement(contact, 'exclude', body1=wrist.attrib['name'], body2=child.attrib['name'])
    for geom in world.findall('.//geom'):
        if '_hand_' in geom.get('name', '') and geom.get('contype') != '0':
            ET.SubElement(contact, 'pair', geom1='bat_handle', geom2=geom.attrib['name'], condim='3', solref=f'{contact_time_constant} 1')
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(destination, encoding='unicode')

class FingerController:

    def __init__(self, model, *, kp=2.0, kd=0.02):
        names = [name for side in ('left', 'right') for name in hand_joints(side)]
        self.joints = np.array([model.joint(name).id for name in names])
        self.actuators = np.array([model.actuator(name).id for name in names])
        self.q = model.jnt_qposadr[self.joints]
        self.v = model.jnt_dofadr[self.joints]
        self.limits = model.jnt_range[self.joints]
        self.force_limits = model.actuator_ctrlrange[self.actuators]
        self.kp, self.kd = (kp, kd)

    def apply(self, data, target):
        target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        torque = self.kp * (target - data.qpos[self.q]) - self.kd * data.qvel[self.v]
        data.ctrl[self.actuators] = np.clip(torque, self.force_limits[:, 0], self.force_limits[:, 1])

def hand_contact_state(model, data, *, velocity=None):
    """Native contact-frame wrench, position and distance for each finger/handle pair."""
    handle = model.geom('bat_handle').id
    velocity = data.qvel if velocity is None else velocity
    rows = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if handle not in contact.geom:
            continue
        other = int(contact.geom[1] if contact.geom[0] == handle else contact.geom[0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        if '_hand_' not in name:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        relative_velocity = np.zeros(3)
        for sign, geom in ((-1, contact.geom[0]), (1, contact.geom[1])):
            jacobian = np.empty((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, contact.pos, model.geom_bodyid[geom])
            relative_velocity += sign * (jacobian @ velocity)
        normal = contact.frame[:3]
        tangential = relative_velocity - normal * (normal @ relative_velocity)
        rows.append({'geom': name, 'geom_order': [model.geom(int(geom)).name for geom in contact.geom], 'normal_force_n': float(wrench[0]), 'wrench_contact_frame': wrench.tolist(), 'position_m': contact.pos.tolist(), 'distance_m': float(contact.dist), 'frame': contact.frame.reshape(3, 3).tolist(), 'tangential_slip_m_s': float(np.linalg.norm(tangential))})
    return rows
