"""Small math/state helpers shared across the harness."""

from __future__ import annotations

import inspect

import numpy as np


def call_begin_episode(obj, log_dir=None):
    """Call obj.begin_episode at kickoff, passing log_dir only if the hook takes it.

    The contract (docs/RFL_RULES.md) is begin_episode(log_dir=None), and the
    hook is optional. On 2026-09-05 a club's round-2 wrapper declared
    begin_episode(self); the keyword raised TypeError at football.py's
    kickoff, m12 could not render, and a slot was hours from airing nothing.
    Scrutineering cannot see a signature and the rule's last-good-commit
    fallback is not implemented, so the engine tolerates the bare form here:
    a club that ignores log_dir loses only its own per-episode logs. When the
    signature cannot be read at all, the contract form is used and any error
    is the club's.
    """
    hook = getattr(obj, "begin_episode", None)
    if hook is None:
        return None
    try:
        params = inspect.signature(hook).parameters
    except (TypeError, ValueError):
        return hook(log_dir=log_dir)
    takes_it = "log_dir" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return hook(log_dir=log_dir) if takes_it else hook()


def yaw_from_quat(q) -> float:
    """Yaw (rad) from a wxyz quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_from_quat(q) -> float:
    """Angle (rad) between the body z-axis and world up. 0 = upright.

    Equivalent to the combined pitch/roll excursion used by the fall rule.
    """
    w, x, y, z = q
    r33 = 1 - 2 * (x * x + y * y)
    return float(np.arccos(np.clip(r33, -1.0, 1.0)))


def quat_from_yaw(yaw: float):
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class CommandBlender:
    """Linearly blends velocity commands over `blend_s` to avoid step-command falls."""

    def __init__(self, blend_s: float = 0.2):
        self.blend_s = blend_s
        self._from = np.zeros(3)
        self._to = np.zeros(3)
        self._t0 = 0.0

    def set_target(self, cmd, t_now: float):
        self._from = self.value(t_now)
        self._to = np.asarray(cmd, dtype=float).copy()
        self._t0 = t_now

    def value(self, t_now: float) -> np.ndarray:
        if self.blend_s <= 0:
            return self._to.copy()
        frac = np.clip((t_now - self._t0) / self.blend_s, 0.0, 1.0)
        return (1 - frac) * self._from + frac * self._to


class FallTracker:
    """Fall rule: (pelvis height < min_height OR tilt > max_tilt) sustained > persist_s."""

    def __init__(self, min_height=0.4, max_tilt_rad=np.deg2rad(60), persist_s=0.5):
        self.min_height = min_height
        self.max_tilt = max_tilt_rad
        self.persist_s = persist_s
        self.bad_since: float | None = None
        self.fall_time: float | None = None

    @property
    def fallen(self) -> bool:
        return self.fall_time is not None

    def update(self, t: float, height: float, tilt: float) -> bool:
        if self.fallen:
            return True
        bad = height < self.min_height or tilt > self.max_tilt
        if bad:
            if self.bad_since is None:
                self.bad_since = t
            elif t - self.bad_since > self.persist_s:
                self.fall_time = t
        else:
            self.bad_since = None
        return self.fallen
