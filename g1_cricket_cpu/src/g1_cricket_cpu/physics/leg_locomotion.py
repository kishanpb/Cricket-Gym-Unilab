"""Frozen Unitree 12-leg-joint controller; upper-body cricket remains separate."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import yaml
from .prior import SDK_JOINTS
ASSET_HASHES = {'motion.pt': 'cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d', 'g1.yaml': '73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb'}

class LegLocomotionPolicy:

    def __init__(self, assets):
        assets = Path(assets)
        for name, expected in ASSET_HASHES.items():
            if hashlib.sha256((assets / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'modified locomotion asset: {name}')
        self.config = yaml.safe_load((assets / 'g1.yaml').read_text())
        self.policy = torch.jit.load(str(assets / 'motion.pt'), map_location='cpu').eval()
        self.default = np.asarray(self.config['default_angles'], dtype=np.float32)
        self.command_scale = np.asarray(self.config['cmd_scale'], dtype=np.float32)
        self.action = np.zeros(12, dtype=np.float32)
        self.reset()

    def reset(self):
        self.policy.reset_memory()
        self.action[:] = 0

    def validate_position_servos(self, model):
        ids = [model.actuator(name).id for name in SDK_JOINTS[:12]]
        joints = [model.joint(name).id for name in SDK_JOINTS[:12]]
        np.testing.assert_array_equal(model.actuator_trnid[ids, 0], joints)
        np.testing.assert_array_equal(model.actuator_gainprm[ids, 0], self.config['kps'])
        np.testing.assert_array_equal(model.actuator_biasprm[ids, 1], -np.asarray(self.config['kps']))
        np.testing.assert_array_equal(model.actuator_biasprm[ids, 2], -np.asarray(self.config['kds']))
        np.testing.assert_array_equal(model.actuator_gear[ids], np.tile([1, 0, 0, 0, 0, 0], (12, 1)))
        return (np.asarray(ids), model.jnt_qposadr[joints], model.jnt_dofadr[joints])

    def observation(self, quaternion, angular_velocity, position, velocity, command, time):
        w, x, y, z = quaternion
        gravity = [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)]
        phase = 2 * np.pi * (time % 0.8 / 0.8)
        return np.concatenate((angular_velocity * self.config['ang_vel_scale'], gravity, np.asarray(command, dtype=np.float32) * self.command_scale, (position - self.default) * self.config['dof_pos_scale'], velocity * self.config['dof_vel_scale'], self.action, [np.sin(phase), np.cos(phase)])).astype(np.float32)

    def target(self, quaternion, angular_velocity, position, velocity, command, time):
        obs = self.observation(quaternion, angular_velocity, position, velocity, command, time)
        with torch.inference_mode():
            self.action = self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze(0)
        if self.action.shape != (12,) or not np.isfinite(self.action).all():
            raise ValueError('invalid frozen leg-policy action')
        return self.default + self.config['action_scale'] * self.action
