"""The trained kick as a RESIDUAL on the walk, inside a STRIKE window.

Contract A of the open policy ladder (work/proposals/2026-09-07-open-policy-ladder.md):
59 observations -> 12 residual joint targets at the walk's 50 Hz cadence, for at
most STEPS control ticks, added to the walk's targets at ACTION_SCALE rad under
the engine's own PD law while the walk is told to stop. The walk keeps balancing
underneath; the residual only learns the kick.

This module is the match-side half of the seam. The training-side half is
`work/kick/engine_env.py` (station), whose `step()` this class reproduces tick
for tick — `tests/test_residual_strike.py` asserts it. Everything the policy
sees is what a robot senses (IMU, joint state, the walk's own action) plus the
ball and aim the skill layer already tracks; nothing is read from the
simulator's internals that the built-in skills do not read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .g1_policy import get_gravity_orientation

OBS = 59
ACT = 12
ACTION_SCALE = 0.75          # rad per unit residual — the artifact contract
STEPS = 40                   # the 0.8 s takeover at 50 Hz. MUST equal
                             # kick_consts.EPISODE_STEPS; the parity test
                             # asserts it. Was 100 until 2026-09-08, when
                             # the ball was measured to peak at tick 6.
SETTLE = 25                  # then 0.5 s of the walk alone, told to stop, before the skill resumes
                             # (the training env ends its episodes the same way)
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
COOLDOWN_S = 3.0             # between windows. WAS 0.5 until 2026-09-08, and
# that half second was the single biggest thing wrong with the kick. The ball
# is usually still close after a strike, so the seam re-armed and struck again
# while the robot was still recovering from the first. Measured on the pitch6
# artifact, same episodes, only the number of back-to-back strikes changed:
# one strike falls 14% of the time, TWO fall 94%, three 96%. The training env
# only ever teaches a single strike from a walking approach, so the second one
# is asked for from a state the policy has never seen. 3 s covers the settle
# plus the recovery the hand-back measures.
# THE GATE (2026-09-07 13:5x). The skill's STRIKE fires within 0.55 m of the
# STANCE, which sits 0.8 m behind the ball, so the ball can be 0.25-1.35 m
# away when it fires; the residual was trained with the ball 0.45-0.75 m
# from the robot's centre and within 0.5 rad of its heading. In the first
# fixture 6 of 8 windows ended in a fall. Until the training window is
# widened to the match's, the residual only takes over inside its own.
BALL_DIST = (0.30, 0.85)     # m, robot centre to ball centre
HEAD_ERR = 0.5               # rad


def _yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _robot_frame(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


class ResidualStrike:
    """One per robot. `begin()` when the skill layer enters STRIKE; then call
    `control()` every physics step in place of `controller.apply_control()`
    until `active` clears (STEPS ticks, or `abort()` on a fall)."""

    def __init__(self, artifact: str | Path):
        self.policy = torch.jit.load(str(artifact), map_location="cpu").eval()
        self.artifact = str(artifact)
        self.active = False
        self.aim = np.zeros(2)
        self.prev = np.zeros(ACT, dtype=np.float32)
        self.a = np.zeros(ACT, dtype=np.float32)
        self.tick = 0
        self.settle = 0
        self.ended_t = -1e9

    def can_begin(self, t: float, robot_xy=None, robot_yaw=None, ball_xy=None) -> bool:
        if self.active or t - self.ended_t < COOLDOWN_S:
            return False
        if ball_xy is None:
            return True
        dx, dy = ball_xy[0] - robot_xy[0], ball_xy[1] - robot_xy[1]
        dist = float(np.hypot(dx, dy))
        err = float(np.arctan2(np.sin(np.arctan2(dy, dx) - robot_yaw), np.cos(np.arctan2(dy, dx) - robot_yaw)))
        return BALL_DIST[0] <= dist <= BALL_DIST[1] and abs(err) <= HEAD_ERR

    def begin(self, aim_dir, t: float):
        n = float(np.hypot(aim_dir[0], aim_dir[1])) or 1.0
        self.aim = np.array([aim_dir[0] / n, aim_dir[1] / n])
        self.prev = np.zeros(ACT, dtype=np.float32)
        self.a = np.zeros(ACT, dtype=np.float32)
        self.tick = 0
        self.settle = 0
        self.active = True

    def abort(self, t: float):
        self.active = False
        self.ended_t = t

    def observe(self, c, d, ball_xy) -> np.ndarray:
        """`engine_env.EngineKickEnv._obs`, line for line: the walk action about
        to be applied, the residual applied last tick, ball and aim in the
        robot's frame, the window phase."""
        pos = d.qpos[c.base_qpos:c.base_qpos + 3]
        quat = d.qpos[c.base_qpos + 3:c.base_qpos + 7]
        yw = _yaw(quat)
        parts = [
            d.qvel[c.base_qvel + 3:c.base_qvel + 6] * ANG_VEL_SCALE,
            get_gravity_orientation(quat),
            d.qpos[c.qpos_idx] - c.default_angles,
            d.qvel[c.qvel_idx] * DOF_VEL_SCALE,
            self.prev,
            _robot_frame(np.asarray(ball_xy, dtype=float) - pos[:2], yw),
            _robot_frame(self.aim, yw),
            [self.tick / STEPS],
            c.action.astype(np.float32),
        ]
        return np.concatenate(parts).astype(np.float32)

    def control(self, c, d, ball_xy, t: float) -> bool:
        """PD torques for this physics step with the residual on the walk's
        targets. Returns False (and writes nothing) once the window is over:
        the caller then falls back to `c.apply_control`."""
        if not self.active:
            return False
        if c.counter % c.control_decimation == 0:
            # the walk has just produced this tick's action (advance()); the
            # residual answers it once per tick and holds for the substeps
            if self.tick >= STEPS:
                if self.settle >= SETTLE:
                    self.active = False
                    self.ended_t = t
                    return False
                self.settle += 1
                self.a = np.zeros(ACT, dtype=np.float32)   # the walk alone, told to stop
            else:
                obs = self.observe(c, d, ball_xy)
                with torch.no_grad():
                    self.a = self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze().astype(np.float32)
                self.a = np.clip(self.a, -1.0, 1.0)
                self.prev = self.a
                self.tick += 1
        target = c.target_dof_pos + self.a * ACTION_SCALE
        tau = (target - d.qpos[c.qpos_idx]) * c.kps - d.qvel[c.qvel_idx] * c.kds
        d.ctrl[c.ctrl_idx] = tau
        return True
