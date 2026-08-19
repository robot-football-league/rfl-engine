"""Exact port of unitree_rl_gym's MuJoCo sim2sim deployment for the G1.

Source: third_party/unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py
Config: third_party/unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml

The observation layout, scales, PD law, and timing must not drift from that
script — it is the documented deployment path for the pretrained policy.

Addressing is resolved by joint/actuator *name* (with an optional prefix), so
the same controller drives a robot in a single-robot scene or a namespaced
robot (e.g. "r0_") in a multi-robot scene. With an empty prefix the resolved
indices equal the original hardcoded slices, so single-robot behavior is
bit-identical to the V1 port.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
import yaml

from . import paths

# Policy DOF order — matches the joint/actuator order in g1_12dof.xml and the
# kps/kds/default_angles arrays in the deploy config.
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
FREE_JOINT = "floating_base_joint"


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


class G1PolicyController:
    """Drives the pretrained velocity-tracking policy for one robot instance.

    Usage: set_command(vx, vy, wz) at any time. Single robot: call
    step(model, data) once per physics step. Multi-robot (shared mj_step):
        for c in controllers: c.apply_control(model, data)
        mujoco.mj_step(model, data)
        for c in controllers: c.advance(data)
    """

    def __init__(self, config_path=None, prefix: str = ""):
        with open(config_path or paths.DEPLOY_CFG) as f:
            cfg = yaml.safe_load(f)
        self.simulation_dt = cfg["simulation_dt"]
        self.control_decimation = cfg["control_decimation"]
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.num_actions = cfg["num_actions"]
        self.num_obs = cfg["num_obs"]

        self.prefix = prefix
        self._bound = False

        torch.set_num_threads(1)
        self.policy = torch.jit.load(str(paths.POLICY_PT))
        self.reset()

    def reset(self):
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.counter = 0

    def bind(self, m: mujoco.MjModel):
        """Resolve qpos/qvel/ctrl addresses for this robot's (prefixed) names."""
        p = self.prefix
        jids = [m.joint(p + n).id for n in JOINT_ORDER]
        self.qpos_idx = np.array([m.jnt_qposadr[j] for j in jids])
        self.qvel_idx = np.array([m.jnt_dofadr[j] for j in jids])
        fj = m.joint(p + FREE_JOINT).id
        self.base_qpos = int(m.jnt_qposadr[fj])  # 7 values: xyz + wxyz quat
        self.base_qvel = int(m.jnt_dofadr[fj])   # 6 values: world-lin + body-ang
        self.ctrl_idx = np.array([m.actuator(p + n).id for n in JOINT_ORDER])
        self._bound = True

    # ---------------------------------------------------------------- base pose

    def base_pos(self, d: mujoco.MjData) -> np.ndarray:
        return d.qpos[self.base_qpos : self.base_qpos + 3]

    def base_quat(self, d: mujoco.MjData) -> np.ndarray:
        return d.qpos[self.base_qpos + 3 : self.base_qpos + 7]

    def base_linvel(self, d: mujoco.MjData) -> np.ndarray:
        return d.qvel[self.base_qvel : self.base_qvel + 3]

    # ---------------------------------------------------------------- control

    def set_command(self, vx, vy, wz):
        self.cmd[0] = vx
        self.cmd[1] = vy
        self.cmd[2] = wz

    def apply_control(self, m: mujoco.MjModel, d: mujoco.MjData):
        """Write this robot's PD torques (call before mj_step)."""
        if not self._bound:
            self.bind(m)
        tau = pd_control(
            self.target_dof_pos, d.qpos[self.qpos_idx], self.kps,
            np.zeros_like(self.kds), d.qvel[self.qvel_idx], self.kds,
        )
        d.ctrl[self.ctrl_idx] = tau

    def advance(self, d: mujoco.MjData):
        """Advance the 50 Hz policy cadence (call after mj_step)."""
        self.counter += 1
        if self.counter % self.control_decimation == 0:
            self._policy_update(d)

    def step(self, m: mujoco.MjModel, d: mujoco.MjData):
        """Single-robot convenience: PD -> mj_step -> policy cadence."""
        self.apply_control(m, d)
        mujoco.mj_step(m, d)
        self.advance(d)

    def _policy_update(self, d: mujoco.MjData):
        na = self.num_actions
        qj = (d.qpos[self.qpos_idx] - self.default_angles) * self.dof_pos_scale
        dqj = d.qvel[self.qvel_idx] * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(self.base_quat(d))
        omega = d.qvel[self.base_qvel + 3 : self.base_qvel + 6] * self.ang_vel_scale

        period = 0.8
        count = self.counter * self.simulation_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs = self.obs
        obs[:3] = omega
        obs[3:6] = gravity_orientation
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9 : 9 + na] = qj
        obs[9 + na : 9 + 2 * na] = dqj
        obs[9 + 2 * na : 9 + 3 * na] = self.action
        obs[9 + 3 * na : 9 + 3 * na + 2] = np.array([sin_phase, cos_phase])

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.target_dof_pos = self.action * self.action_scale + self.default_angles
