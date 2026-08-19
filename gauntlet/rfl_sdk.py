"""RFL 0.3 on-robot SDK — the software a real competition G1 already carries.

Modelled on what actual participants use, not invented here. Unitree's G1-Comp
RoboCup SDK exposes three API groups — Visual Recognition (YOLO11), Spatial
Positioning (monocular/binocular → metric positions) and Motion Control
(motion interfaces driven by detection results) — and RoboCup teams report the
same pipeline: detector → inverse camera transform → robot-centric CARTESIAN
world model → A* navigation → behaviour state machine → walk engine
(HULKs RoboCup 2026 software survey; NimbRo; EWS Bascorro).

So the engine owns:
  * PERCEPTION  — detections in metres, limited to what the camera can see
  * WORLD MODEL — a filtered estimate with ball memory, as every team keeps
  * SKILLS      — walk_to / turn_to / go_to_ball / kick_toward / hold, run at
                  control rate on top of the pretrained walking policy

and a team owns only BEHAVIOUR: which skill, aimed where. That is the layer
RoboCup teams write as a state machine and the layer LLCoach drives with an
LLM — and it is the layer this league exists to compare models on.

Nothing here reads privileged state on a team's behalf: object identity comes
from a detector (reliable, as a trained YOLO is), but every reported position
is measured THROUGH the camera model — field of view, occlusion by other
robots, and pixel quantisation all apply, so distant objects carry the
characteristic depth error of monocular ground-plane projection.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

BALL_MEMORY_S = 6.0        # how long a lost ball stays in the world model
ROBOT_RADIUS_M = 0.28      # occlusion + avoidance footprint
MIN_BALL_PIXELS = 25       # detector sensitivity floor
NAV_RES_M = 0.25           # A* grid resolution


# ----------------------------------------------------------------- geometry
class Camera:
    """Pinhole model of one robot's panoramic egocam (square pixels)."""

    def __init__(self, model, cam_name: str, width: int, height: int, fovy_deg: float):
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self.w, self.h = width, height
        self.f = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
        self.cx, self.cy = width / 2, height / 2

    def pose(self, data):
        return np.array(data.cam_xpos[self.id]), np.array(data.cam_xmat[self.id]).reshape(3, 3)

    def pixel_to_ground(self, data, u, v, plane_z):
        """Inverse camera transform: pixel + known object height -> world point.
        MuJoCo cameras look down -z with +x right, +y up."""
        p, R = self.pose(data)
        d_cam = np.array([u - self.cx, self.cy - v, -self.f], dtype=float)
        d_world = R @ d_cam
        if abs(d_world[2]) < 1e-9:
            return None
        t = (plane_z - p[2]) / d_world[2]
        if t <= 0:
            return None
        return p + t * d_world

    def world_to_pixel(self, data, pw):
        p, R = self.pose(data)
        d_cam = R.T @ (np.asarray(pw, dtype=float) - p)
        if d_cam[2] > -1e-6:            # behind the image plane
            return None
        u = self.cx + self.f * (d_cam[0] / -d_cam[2])
        v = self.cy - self.f * (d_cam[1] / -d_cam[2])
        if not (0 <= u < self.w and 0 <= v < self.h):
            return None
        return u, v


def to_robot_frame(pw, self_xy, yaw):
    dx, dy = float(pw[0]) - self_xy[0], float(pw[1]) - self_xy[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy)


def _occluded(cam_xy, target_xy, blockers_xy):
    """2D line-of-sight test against other robot footprints."""
    ax, ay = cam_xy
    bx, by = target_xy
    seg = np.array([bx - ax, by - ay])
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return False
    u = seg / L
    for ox, oy in blockers_xy:
        rel = np.array([ox - ax, oy - ay])
        along = float(rel @ u)
        if along <= 0.15 or along >= L - 0.15:
            continue
        if abs(float(rel[0] * u[1] - rel[1] * u[0])) < ROBOT_RADIUS_M:
            return True
    return False


# --------------------------------------------------------------- perception
def detect_ball_pixels(frame, min_px: int = MIN_BALL_PIXELS):
    """'YOLO' stand-in for the ball: it is the only magenta object on the
    pitch, so colour segmentation is a faithful, fully image-derived detector.
    Occlusion and field of view are handled for free — hidden pixels are
    simply absent. Returns (centroid_u, centroid_v, pixel_count) or None."""
    a = frame.astype(np.int16)
    m = (a[:, :, 0] > 170) & (a[:, :, 1] < 115) & (a[:, :, 2] > 130)
    n = int(m.sum())
    if n < min_px:
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean()), n


class WorldModel:
    """Per-robot filtered state, as every RoboCup team keeps: last known ball
    (in field coordinates, with an age) plus the latest robot sightings.

    Ball memory is what stops the classic 'ball is behind me so I walk away'
    failure: a real robot remembers where the ball was and turns to it."""

    VEL_WINDOW_S = 1.2      # baseline for the velocity estimate
    VEL_SMOOTH = 0.35       # light low-pass; the LSQ fit already de-noises

    def __init__(self):
        self.ball_xy = None
        self.ball_t = -1e9
        self.ball_seen_now = False
        self.ball_vel = (0.0, 0.0)
        self._hist = []       # [(t, x, y)] recent sightings, for velocity
        self.robots = []      # [{"team": "ours"|"theirs", "xy": (x, y)}]

    def update_ball(self, xy, t):
        if xy is None:
            self.ball_seen_now = False
            return
        self.ball_xy, self.ball_t, self.ball_seen_now = tuple(xy), t, True
        # filtered ball VELOCITY, as every real world model keeps. Without it
        # a team cannot lead a moving ball at all — measured tail-chase left
        # pursuers a median 1.3 m behind after 2 s.
        self._hist.append((t, xy[0], xy[1]))
        del self._hist[:-6]
        win = [h for h in self._hist if t - h[0] <= self.VEL_WINDOW_S]
        if len(win) >= 3 and win[-1][0] - win[0][0] > 0.2:
            # least-squares slope over the window: far steadier than a
            # two-point difference when each sighting carries ~0.25 m of error
            ts = np.array([h[0] for h in win])
            ts = ts - ts.mean()
            var = float((ts * ts).sum()) or 1e-9
            mvx = float((ts * (np.array([h[1] for h in win])
                               - np.mean([h[1] for h in win]))).sum()) / var
            mvy = float((ts * (np.array([h[2] for h in win])
                               - np.mean([h[2] for h in win]))).sum()) / var
            k = self.VEL_SMOOTH
            self.ball_vel = (k * self.ball_vel[0] + (1 - k) * mvx,
                             k * self.ball_vel[1] + (1 - k) * mvy)

    def ball_future(self, t, horizon_s):
        """Where the ball is heading, if anyone cares to use it."""
        if self.ball_xy is None:
            return None
        return (self.ball_xy[0] + self.ball_vel[0] * horizon_s,
                self.ball_xy[1] + self.ball_vel[1] * horizon_s)

    def ball_age(self, t):
        return None if self.ball_xy is None else t - self.ball_t

    def ball_valid(self, t):
        age = self.ball_age(t)
        return age is not None and age <= BALL_MEMORY_S


def perceive(model, data, cam: Camera, frame, self_xy, self_yaw, self_index,
             all_xy, teams, ball_z, t, wm: WorldModel,
             pitch=None, wall_m=0.75, min_px=MIN_BALL_PIXELS):
    """Run one perception cycle for a robot and fold it into its world model.

    Returns the team-facing 'detections' block: robot-centric metres, plus a
    field-coordinate copy (a localised robot has both, and RoboCup teams use
    both frames). Only what the camera could actually see is reported."""
    # --- ball: fully image-derived (magenta detector + inverse transform)
    ball_field = None
    hit = detect_ball_pixels(frame, min_px)
    if hit is not None:
        u, v, npx = hit
        pw = cam.pixel_to_ground(data, round(u), round(v), ball_z)
        if pw is not None:
            ball_field = (float(pw[0]), float(pw[1]))
    wm.update_ball(ball_field, t)

    # --- other robots: identity from the detector, position through the
    #     camera model (FOV + occlusion + pixel quantisation)
    cam_p, _ = cam.pose(data)
    seen = []
    for j, xy in enumerate(all_xy):
        if j == self_index:
            continue
        px = cam.world_to_pixel(data, (xy[0], xy[1], 0.35))
        if px is None:
            continue
        blockers = [o for k, o in enumerate(all_xy)
                    if k not in (j, self_index)]
        if _occluded((cam_p[0], cam_p[1]), xy, blockers):
            continue
        pw = cam.pixel_to_ground(data, round(px[0]), round(px[1]), 0.0)
        if pw is None:
            continue
        seen.append({"team": "ours" if teams[j] == teams[self_index] else "theirs",
                     "xy": (float(pw[0]), float(pw[1]))})
    wm.robots = seen

    def pack(xy):
        rx, ry = to_robot_frame(xy, self_xy, self_yaw)
        return {"forward_m": round(rx, 2), "left_m": round(ry, 2),
                "distance_m": round(float(math.hypot(rx, ry)), 2),
                "bearing_deg": round(math.degrees(math.atan2(ry, rx)), 1),
                "field_xy": [round(xy[0], 2), round(xy[1], 2)]}

    det = {"ball": None, "teammates": [], "opponents": []}
    if wm.ball_xy is not None and wm.ball_valid(t):
        b = pack(wm.ball_xy)
        b["seen_now"] = bool(wm.ball_seen_now)
        b["age_s"] = round(float(wm.ball_age(t)), 1)
        if pitch is not None:
            px_, py_ = pitch
            b["against_wall"] = bool(abs(wm.ball_xy[0]) > px_ - wall_m
                                     or abs(wm.ball_xy[1]) > py_ - wall_m)
        vx_, vy_ = wm.ball_vel
        b["velocity_mps"] = [round(vx_, 2), round(vy_, 2)]
        b["speed_mps"] = round(float(math.hypot(vx_, vy_)), 2)
        det["ball"] = b
    for r in seen:
        det["teammates" if r["team"] == "ours" else "opponents"].append(pack(r["xy"]))
    return det


# ------------------------------------------------------------------- skills
class SkillRunner:
    """Motion Control layer: turns a behaviour-level skill into velocity
    commands every control step, exactly as the G1-Comp SDK's motion
    interfaces turn detection results into movement.

    Navigation is A* over a coarse grid re-planned at behaviour rate and
    tracked with pure pursuit at control rate — the standard split.
    """

    NAMES = ("go_to_ball", "kick_toward", "walk_to", "turn_to", "hold")
    MAX_LEAD_S = 2.5

    def __init__(self, pitch_x, pitch_y, ball_radius, envelope_vx=(0.0, 1.0),
                 max_wz=1.0, goal_half_w=1.6):
        self.px, self.py = pitch_x, pitch_y
        self.ball_r = ball_radius
        self.gw = goal_half_w
        self.vx_max = envelope_vx[1]
        self.wz_max = max_wz
        self.skill = "hold"
        self.target = None          # field coords
        self.track = None           # "ball" -> live target, re-solved each step
        self.lead_s = 0.0           # how far ahead of it to aim (team's choice)
        self.path = []              # remaining waypoints (field coords)

    # -- planning (behaviour rate)
    def set_skill(self, name, target, wm, self_xy, attack_goal_xy, t,
                  lead_s=0.0):
        self.skill = name if name in self.NAMES else "hold"
        # walk_to accepts a MOVING target ("ball"): the point is then re-solved
        # every control step instead of freezing a 2 s-stale waypoint. Whether
        # to track, and how far to lead, stays the team's decision.
        self.track = "ball" if isinstance(target, str) and target == "ball" else None
        self.lead_s = float(max(0.0, min(self.MAX_LEAD_S, lead_s or 0.0)))
        self.target = None if self.track else target
        self.path = []
        goal = self._goal_point(wm, self_xy, attack_goal_xy, t)
        if goal is not None and self.skill in ("go_to_ball", "kick_toward", "walk_to"):
            self.path = self._plan(self_xy, goal, wm)

    def stance_and_aim(self, ball, aim):
        """Where to stand to push `ball` toward `aim`, REPAIRED for walls.

        The naive stance (directly opposite the aim) can land outside the
        pitch when the ball is against a wall — a striker then presses into
        the wall forever and pins the ball there, which is exactly how a
        four-robot corner scrum forms (observed: ball held 21 s at the end
        wall because the stance solved to x = -7.33 on a 7 m pitch).

        So rotate the aim as little as necessary until the stance is
        reachable: the ball gets knocked ALONG the wall in roughly the
        intended direction, which is what a real player does.
        """
        off = self.ball_r + 0.45
        bx, by = ball
        base = math.atan2(aim[1] - by, aim[0] - bx)
        deltas = (0, 20, -20, 40, -40, 60, -60, 80, -80, 100, -100,
                  120, -120, 145, -145, 170, -170, 180)

        def stance_of(a):
            return (bx - math.cos(a) * off, by - math.sin(a) * off)

        def reachable(st):
            return abs(st[0]) <= self.px - 0.3 and abs(st[1]) <= self.py - 0.3

        # first choice: a stance we can stand in AND a push that does not
        # drive the ball further into a wall
        for require_progress in (True, False):
            for delta in deltas:
                a = base + math.radians(delta)
                st = stance_of(a)
                if not reachable(st):
                    continue
                if require_progress and self._rams_wall(bx, by, a):
                    continue
                return st, (bx + math.cos(a) * 2.5, by + math.sin(a) * 2.5)
        return (bx, by), aim

    WALL_HUG_M = 0.55      # "already against" that wall
    INTO_WALL = 0.25       # direction component that counts as pushing into it

    def _rams_wall(self, bx, by, ang):
        """True only if the ball is ALREADY against a wall and this push drives
        it further in. Deliberately narrow: an earlier version rejected any
        push whose 1 m lookahead neared a wall, which vetoed ordinary attacking
        play and cost goals (unopposed striker 3 -> 1, all-chase 2 -> 0)."""
        dx, dy = math.cos(ang), math.sin(ang)
        in_mouth = abs(by) <= self.gw - 0.15   # end wall is open here: shoot!
        if bx > self.px - self.WALL_HUG_M and dx > self.INTO_WALL and not in_mouth:
            return True
        if bx < -(self.px - self.WALL_HUG_M) and dx < -self.INTO_WALL and not in_mouth:
            return True
        if by > self.py - self.WALL_HUG_M and dy > self.INTO_WALL:
            return True
        if by < -(self.py - self.WALL_HUG_M) and dy < -self.INTO_WALL:
            return True
        return False

    def _aim_for(self, wm, attack_goal_xy):
        return (self.target if (self.skill == "kick_toward" and self.target)
                else attack_goal_xy)

    def live_target(self, wm, t):
        """Current aim point for a tracked (moving) target."""
        if self.track != "ball" or not wm.ball_valid(t):
            return None
        return wm.ball_future(t, self.lead_s)

    def ball_point(self, wm, t):
        """The ball position this skill plays to: now, or led by lead_s."""
        if not wm.ball_valid(t):
            return None
        if self.lead_s > 0.0:
            fut = wm.ball_future(t, self.lead_s)
            if fut is not None:
                return (float(np.clip(fut[0], -self.px, self.px)),
                        float(np.clip(fut[1], -self.py, self.py)))
        return wm.ball_xy

    def _goal_point(self, wm, self_xy, attack_goal_xy, t):
        """Where this skill wants the body to be, in field coords."""
        if self.skill == "walk_to":
            return self.live_target(wm, t) if self.track else self.target
        if self.skill in ("go_to_ball", "kick_toward"):
            bp = self.ball_point(wm, t)
            if bp is None:
                return None
            stance, _ = self.stance_and_aim(bp, self._aim_for(wm, attack_goal_xy))
            return stance
        return None

    def _plan(self, start, goal, wm):
        from .course import astar
        nx = int(2 * self.px / NAV_RES_M)
        ny = int(2 * self.py / NAV_RES_M)
        n = max(nx, ny)
        grid = np.zeros((n, n), dtype=bool)
        grid[nx:, :] = True
        grid[:, ny:] = True

        def cell(p):
            i = int(np.clip((p[0] + self.px) / NAV_RES_M, 0, nx - 1))
            j = int(np.clip((p[1] + self.py) / NAV_RES_M, 0, ny - 1))
            return i, j

        for r in wm.robots:                       # avoid known robots
            ci, cj = cell(r["xy"])
            k = int(math.ceil(ROBOT_RADIUS_M * 1.6 / NAV_RES_M))
            grid[max(0, ci - k):ci + k + 1, max(0, cj - k):cj + k + 1] = True
        s, g = cell(start), cell(goal)
        grid[s] = False
        grid[g] = False
        path = astar(grid, s, g)
        if not path:
            return [goal]
        pts = [((i + 0.5) * NAV_RES_M - self.px, (j + 0.5) * NAV_RES_M - self.py)
               for i, j in path[1:]]
        return pts[::3] + [goal] if len(pts) > 3 else pts + [goal]

    # -- tracking (control rate)
    def step(self, wm, self_xy, self_yaw, attack_goal_xy, t):
        if self.skill == "hold":
            return (0.0, 0.0, 0.0)
        if self.skill == "turn_to":
            aim = self.target or (wm.ball_xy if wm.ball_valid(t) else None)
            return (0.0, 0.0, self._turn(self_xy, self_yaw, aim)) if aim else (0.0, 0.0, 0.6)

        if self.skill in ("go_to_ball", "kick_toward"):
            if not wm.ball_valid(t):
                return (0.0, 0.0, 0.7)          # sweep for a lost ball
            bp = self.ball_point(wm, t)
            if bp is None:
                return (0.0, 0.0, 0.7)
            bx, by = bp
            stance, aim = self.stance_and_aim(
                bp, self._aim_for(wm, attack_goal_xy))
            dx, dy = aim[0] - bx, aim[1] - by
            n = math.hypot(dx, dy) or 1.0
            d_stance = math.hypot(stance[0] - self_xy[0], stance[1] - self_xy[1])
            head_err = self._wrap(math.atan2(dy, dx) - self_yaw)
            if d_stance < 0.55:
                if abs(head_err) < 0.5:
                    # STRIKE: accelerate through the ball toward the aim point
                    return (self.vx_max, 0.0,
                            self._turn(self_xy, self_yaw, (bx + dx / n, by + dy / n)))
                # standing on the stance but not lined up: PIVOT, never freeze.
                # (returning zeros here deadlocked strikers for ~12 s at a time)
                return (0.12, 0.0,
                        float(np.clip(head_err / 0.4, -self.wz_max, self.wz_max)))
            # NEVER BARGE THROUGH THE BALL ON THE WAY TO THE STANCE. The
            # stance sits on the far side whenever we approach from the
            # attacking side, and a straight walk shoves the ball toward our
            # own goal (fixture 1: every one of five goals was last touched
            # by the conceding side). If the straight leg to the stance
            # passes within a body's width of the live ball, orbit: aim for
            # the tangent point beside the ball nearer the stance and swing
            # around before pushing.
            cur = wm.ball_xy if wm.ball_valid(t) else (bx, by)
            fut = (wm.ball_future(t, 0.6) or cur) if wm.ball_valid(t) else cur
            if (self._crosses_ball(self_xy, stance, cur)
                    or self._crosses_ball(self_xy, stance, fut)):
                self.path = []            # planned legs may also cross it
                # swing around where the ball is HEADED, not where it was
                return self._follow(self._orbit_point(self_xy, stance, fut),
                                    self_xy, self_yaw)
            return self._follow(stance, self_xy, self_yaw)

        if self.track:
            live = self.live_target(wm, t)
            if live is None:
                return (0.0, 0.0, 0.7)      # lost it: sweep
            self.path = []                  # a moving target needs no stale path
            return self._follow(live, self_xy, self_yaw)
        return self._follow(self.target, self_xy, self_yaw)

    BALL_CLEAR_M = 0.55      # keep this much clearance when passing the ball
    ORBIT_R_M = 0.80         # radius of the swing around it

    def _crosses_ball(self, a, b, ball):
        """Does the straight leg a->b pass within BALL_CLEAR_M of the ball?"""
        ax, ay = a
        bx_, by_ = b
        vx_, vy_ = bx_ - ax, by_ - ay
        L2 = vx_ * vx_ + vy_ * vy_
        if L2 < 1e-9:
            return False
        u = ((ball[0] - ax) * vx_ + (ball[1] - ay) * vy_) / L2
        if u <= 0.0 or u >= 1.0:      # ball not between us and the stance
            return False
        px_ = ax + u * vx_
        py_ = ay + u * vy_
        return math.hypot(ball[0] - px_, ball[1] - py_) < self.BALL_CLEAR_M

    def _orbit_point(self, self_xy, stance, ball):
        """Tangent waypoint beside the ball, on whichever side reaches the
        stance sooner, clamped inside the pitch."""
        vx_, vy_ = self_xy[0] - ball[0], self_xy[1] - ball[1]
        n = math.hypot(vx_, vy_) or 1.0
        cands = [(ball[0] - vy_ / n * self.ORBIT_R_M,
                  ball[1] + vx_ / n * self.ORBIT_R_M),
                 (ball[0] + vy_ / n * self.ORBIT_R_M,
                  ball[1] - vx_ / n * self.ORBIT_R_M)]
        w = min(cands, key=lambda c: math.hypot(c[0] - stance[0],
                                                c[1] - stance[1]))
        return (float(np.clip(w[0], -self.px + 0.3, self.px - 0.3)),
                float(np.clip(w[1], -self.py + 0.3, self.py - 0.3)))

    def _follow(self, goal, self_xy, self_yaw):
        if goal is None:
            return (0.0, 0.0, 0.0)
        while self.path and math.hypot(self.path[0][0] - self_xy[0],
                                       self.path[0][1] - self_xy[1]) < 0.45:
            self.path.pop(0)
        wp = self.path[0] if self.path else goal
        err = self._wrap(math.atan2(wp[1] - self_xy[1], wp[0] - self_xy[0]) - self_yaw)
        dist = math.hypot(goal[0] - self_xy[0], goal[1] - self_xy[1])
        if dist < 0.25:
            return (0.0, 0.0, 0.0)
        # slow down for hard turns and on arrival, else full stride
        vx = self.vx_max * (0.25 if abs(err) > 1.0 else
                            0.6 if abs(err) > 0.45 else 1.0)
        vx = min(vx, max(0.2, dist))
        return (vx, 0.0, float(np.clip(err / 0.5, -self.wz_max, self.wz_max)))

    def _turn(self, self_xy, self_yaw, aim):
        err = self._wrap(math.atan2(aim[1] - self_xy[1], aim[0] - self_xy[0]) - self_yaw)
        return float(np.clip(err / 0.5, -self.wz_max, self.wz_max))

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
