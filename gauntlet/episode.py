"""Episode runner: one agent driving the G1 through one course.

Two timing modes:

- "paused" (oracle): the sim does not advance while `agent.decide()` runs.
  Latency is invisible; every agent gets exactly 2 Hz of decisions. This
  isolates judgment from speed and is fully deterministic for deterministic
  agents.
- "realtime" (competition): the sim advances at wall-clock speed. Decisions
  run in a worker thread against a snapshot of the world; the robot keeps
  executing its last command while the agent thinks, and a reply steers the
  robot wherever it is when the reply arrives. Latency is a first-class cost,
  exactly as on a physical robot. One request is in flight at a time, with a
  minimum request period of 0.5 s (2 Hz cap).

Physics runs at 500 Hz, the low-level policy at 50 Hz in both modes.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import mujoco
import numpy as np

from .course import CourseSpec
from .envelope import load_envelope
from .g1_policy import G1PolicyController
from .render import EpisodeRenderer
from .scene import build_scene_xml
from .util import CommandBlender, FallTracker, tilt_from_quat, yaw_from_quat

TIME_LIMIT_S = 90.0
DECISION_PERIOD_S = 0.5  # 2 Hz cap in both modes
BLEND_S = 0.2
BUMP_DEBOUNCE_S = 0.5
# Command watchdog (standard robot-bridge practice): rotation and lateral
# rates expire this long after a command lands; forward speed persists. Bounds
# each command's total turn to wz x ROT_HOLD_S no matter how slow the brain,
# which prevents high-latency agents from spiralling on stale turn rates.
ROT_HOLD_S = 2.0

CP_RGBA_NEXT = (0.15, 0.9, 0.35, 0.55)
CP_RGBA_FUTURE = (0.2, 0.75, 0.4, 0.18)
CP_RGBA_DONE = (0.55, 0.55, 0.55, 0.12)


@dataclass
class EpisodeResult:
    agent: str
    course: str
    repeat: int
    seed: int | None
    mode: str = "paused"
    checkpoints: int = 0
    completed: bool = False
    completion_time_s: float | None = None
    elapsed_s: float = 0.0
    score: float = 0.0
    fell: bool = False
    fall_time_s: float | None = None
    bumps: int = 0
    invalid_actions: int = 0
    clipped_actions: int = 0
    decisions: int = 0
    missed_deadlines: int = 0
    mean_decision_latency_s: float | None = None
    max_decision_latency_s: float | None = None
    effective_decision_hz: float | None = None
    realtime_max_lag_s: float | None = None
    spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    checkpoint_times: list[float] = field(default_factory=list)
    bump_events: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def validate_action(raw, envelope) -> tuple[tuple[float, float, float] | None, str]:
    """Returns (clamped command | None, status in ok|clipped|ignored_invalid)."""
    if not isinstance(raw, dict):
        return None, "ignored_invalid"
    try:
        vx, vy, wz = float(raw["vx"]), float(raw["vy"]), float(raw["wz"])
    except (KeyError, TypeError, ValueError):
        return None, "ignored_invalid"
    if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(wz)):
        return None, "ignored_invalid"
    c = (
        float(np.clip(vx, *envelope["vx"])),
        float(np.clip(vy, *envelope["vy"])),
        float(np.clip(wz, *envelope["wz"])),
    )
    status = "ok" if np.allclose(c, (vx, vy, wz), atol=1e-9) else "clipped"
    return c, status


def obstacle_entries(course: CourseSpec, t: float) -> list[dict]:
    """The obstacle part of the observation JSON (shared with the race runner)."""
    obstacles = []
    for ob in course.obstacles:
        obstacles.extend(ob.obs_entries(t))
    return obstacles


def build_observation(course: CourseSpec, data, t: float, cp_next: int,
                      fallen: bool, last_action_result: str,
                      decision_interval_s: float = DECISION_PERIOD_S,
                      blocked: bool = False) -> dict:
    q = data.qpos
    pos = [round(float(q[0]), 2), round(float(q[1]), 2)]
    vel = [round(float(data.qvel[0]), 2), round(float(data.qvel[1]), 2)]
    heading = round(yaw_from_quat(q[3:7]), 2)

    cps = course.checkpoints
    nxt = cps[cp_next]
    dist = float(np.hypot(nxt[0] - q[0], nxt[1] - q[1]))

    obstacles = obstacle_entries(course, t)

    return {
        "time_remaining_s": round(TIME_LIMIT_S - t, 1),
        "decision_interval_s": round(decision_interval_s, 2),
        "self": {
            "position": pos,
            "heading_rad": heading,
            "velocity": vel,
            "fallen": bool(fallen),
            "blocked": bool(blocked),
        },
        "next_checkpoint": {
            "index": cp_next,
            "position": [round(float(nxt[0]), 2), round(float(nxt[1]), 2)],
            "distance_m": round(dist, 1),
        },
        "remaining_checkpoints": [
            [round(float(c[0]), 2), round(float(c[1]), 2)] for c in cps[cp_next + 1 :]
        ],
        "obstacles": obstacles,
        "last_action_result": last_action_result,
    }


class _AsyncDecider:
    """Runs agent.decide() in a worker thread; at most one request in flight.

    The sim thread calls submit() with a snapshot observation, then poll()s
    every physics step; the completed reply is returned exactly once.
    abandon() voids a request stuck in flight (a hung provider call) so the
    decider can take a fresh one.
    """

    def __init__(self, agent):
        self.agent = agent
        self._lock = threading.Lock()
        self._busy = False
        self._result = None  # (raw, request_t, latency_s, error, obs)
        self._gen = 0        # bumped by abandon(); stale replies are dropped
        self._since = 0.0    # wall clock of the in-flight submit

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def in_flight_s(self) -> float:
        """Wall seconds the current request has been in flight (0 if idle)."""
        with self._lock:
            return time.time() - self._since if self._busy else 0.0

    def abandon(self):
        """Give up on a hung request: a provider call that never returns
        must cost one decision, not the rest of the episode. The worker
        thread may still finish eventually; its reply is void."""
        with self._lock:
            self._gen += 1
            self._busy = False
            self._result = None

    def submit(self, obs: dict, request_t: float):
        with self._lock:
            self._busy = True
            self._since = time.time()
            gen = self._gen

        def work():
            t0 = time.time()
            error = None
            try:
                raw = self.agent.decide(obs)
            except Exception as e:  # an agent crash must not kill the episode
                raw, error = None, f"{type(e).__name__}: {e}"
            with self._lock:
                if gen != self._gen:
                    return           # abandoned while out: reply is void
                self._result = (raw, request_t, time.time() - t0, error, obs)
                self._busy = False

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        with self._lock:
            result, self._result = self._result, None
            return result


def run_episode(course: CourseSpec, agent, repeat: int = 0,
                spawn_jitter: tuple[float, float] = (0.0, 0.0),
                video_path: str | Path | None = None,
                log_dir: str | Path | None = None,
                course_name: str | None = None,
                mode: str = "paused",
                realtime_factor: float = 1.0,
                decision_deadline_s: float | None = None,
                request_period_s: float | None = None) -> EpisodeResult:
    assert mode in ("paused", "realtime"), mode
    t_wall = time.time()
    spawn = (course.spawn[0] + spawn_jitter[0], course.spawn[1] + spawn_jitter[1],
             course.spawn[2])
    xml = build_scene_xml(course, spawn=spawn)
    model = mujoco.MjModel.from_xml_string(xml)
    ctrl = G1PolicyController()
    model.opt.timestep = ctrl.simulation_dt
    data = mujoco.MjData(model)

    envelope = load_envelope()
    blender = CommandBlender(BLEND_S)
    fall = FallTracker()
    dt = ctrl.simulation_dt
    decision_every = int(round(DECISION_PERIOD_S / dt))
    max_steps = int(round(TIME_LIMIT_S / dt))

    # --- static ids
    robot_bodies = set()
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == pelvis_id:
                robot_bodies.add(b)
                break
            p = model.body_parentid[p]
    robot_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in robot_bodies}
    hit_geoms = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("obs_") or name.startswith("wall_"):
            hit_geoms[g] = name
    movers = []
    for ob in course.movers:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mover_{ob.id}")
        movers.append((ob, model.body_mocapid[bid]))
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"cp_{i}")
                for i in range(len(course.checkpoints))]

    def recolor(cp_next: int):
        for i, sid in enumerate(site_ids):
            if sid < 0:
                continue
            rgba = CP_RGBA_DONE if i < cp_next else CP_RGBA_NEXT if i == cp_next else CP_RGBA_FUTURE
            model.site_rgba[sid] = rgba

    renderer = EpisodeRenderer(model, video_path) if video_path else None
    log_dir = Path(log_dir) if log_dir else None
    decisions_f = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        decisions_f = open(log_dir / "decisions.jsonl", "w")

    if hasattr(agent, "begin_episode"):
        agent.begin_episode(log_dir=log_dir)

    result = EpisodeResult(
        agent=getattr(agent, "name", agent.__class__.__name__),
        course=course_name or (f"seed_{course.seed}" if course.seed >= 0 else "hand"),
        repeat=repeat, seed=course.seed if course.seed >= 0 else None,
        mode=mode,
        spawn=tuple(round(v, 3) for v in spawn),
    )

    cp_next = 0
    recolor(cp_next)
    last_action_result = "ok"
    consecutive_invalid = 0
    last_contact_t: dict[str, float] = {}
    latencies: list[float] = []
    max_lag = 0.0
    t = 0.0
    prev_decision_t: float | None = None
    cmd_applied_t: float | None = None
    cmd_vx = 0.0
    rot_expired = False
    # blocked = commanding forward but net displacement ~0 over a 1 s window
    # (instantaneous base velocity oscillates while stepping in place, so a
    #  speed threshold alone never latches)
    blocked_flag = False
    block_mark: tuple | None = None
    request_period = max(DECISION_PERIOD_S, request_period_s or 0.0)

    def decision_interval(t_now: float) -> float:
        return DECISION_PERIOD_S if prev_decision_t is None else t_now - prev_decision_t

    def apply_reply(raw, t_now: float, obs: dict, latency: float, error=None):
        nonlocal consecutive_invalid, last_action_result, prev_decision_t
        nonlocal cmd_applied_t, cmd_vx, rot_expired
        cmd, status = validate_action(raw, envelope)
        if cmd is None:
            consecutive_invalid += 1
            result.invalid_actions += 1
            if consecutive_invalid >= 3:
                blender.set_target((0.0, 0.0, 0.0), t_now)
        else:
            consecutive_invalid = 0
            if status == "clipped":
                result.clipped_actions += 1
            blender.set_target(cmd, t_now)
            cmd_applied_t, cmd_vx, rot_expired = t_now, cmd[0], False
        last_action_result = status
        result.decisions += 1
        prev_decision_t = t_now
        latencies.append(latency)
        if decisions_f:
            decisions_f.write(json.dumps(
                {"t": round(t_now, 2), "obs_t": obs["time_remaining_s"],
                 "obs": obs, "raw": raw, "applied": cmd, "status": status,
                 "latency_s": round(latency, 3), "error": error}) + "\n")

    decider = _AsyncDecider(agent) if mode == "realtime" else None
    last_request_t = -1e9
    wall_start = time.time()

    try:
        for k in range(max_steps):
            # kinematic hazards follow their scripted motion exactly
            for ob, mid in movers:
                pos, quat = ob.pose_at(t)
                data.mocap_pos[mid] = pos
                data.mocap_quat[mid] = quat

            if mode == "paused":
                if k % decision_every == 0:
                    obs = build_observation(course, data, t, cp_next, fall.fallen,
                                            last_action_result,
                                            decision_interval(t),
                                            blocked_flag)
                    t0 = time.time()
                    error = None
                    try:
                        raw = agent.decide(obs)
                    except Exception as e:
                        raw, error = None, f"{type(e).__name__}: {e}"
                    apply_reply(raw, t, obs, time.time() - t0, error)
            else:
                reply = decider.poll()
                if reply is not None:
                    raw, req_t, latency, error, req_obs = reply
                    if decision_deadline_s and latency > decision_deadline_s:
                        # shot clock: stale replies are void (bridge drops them)
                        result.missed_deadlines += 1
                        if decisions_f:
                            decisions_f.write(json.dumps(
                                {"t": round(t, 2), "raw": raw, "applied": None,
                                 "status": "expired_deadline",
                                 "latency_s": round(latency, 3)}) + "\n")
                    else:
                        apply_reply(raw, t, req_obs, latency, error)
                if not decider.busy and t - last_request_t >= request_period - 1e-9:
                    # interval reported to the agent = request-to-request gap
                    interval = (request_period if last_request_t < -1e8
                                else t - last_request_t)
                    obs = build_observation(course, data, t, cp_next, fall.fallen,
                                            last_action_result, interval,
                                            blocked_flag)
                    decider.submit(obs, t)
                    last_request_t = t

            # watchdog: stale rotation/lateral rates expire; forward persists
            if (cmd_applied_t is not None and not rot_expired
                    and t - cmd_applied_t > ROT_HOLD_S):
                blender.set_target((cmd_vx, 0.0, 0.0), t)
                rot_expired = True

            ctrl.set_command(*blender.value(t))
            ctrl.step(model, data)
            t += dt

            if mode == "realtime":
                target = wall_start + t / realtime_factor
                ahead = target - time.time()
                if ahead > 0.002:
                    time.sleep(ahead)
                elif ahead < 0:
                    max_lag = max(max_lag, -ahead)

            fall.update(t, data.qpos[2], tilt_from_quat(data.qpos[3:7]))

            if block_mark is None:
                block_mark = (t, float(data.qpos[0]), float(data.qpos[1]))
            elif t - block_mark[0] >= 1.0:
                mt, mx, my = block_mark
                rate = float(np.hypot(data.qpos[0] - mx, data.qpos[1] - my)) / (t - mt)
                blocked_flag = cmd_vx > 0.3 and rate < 0.12
                block_mark = (t, float(data.qpos[0]), float(data.qpos[1]))

            # checkpoint trigger: pelvis inside zone while not fallen
            if not fall.fallen and cp_next < len(course.checkpoints):
                cx, cy = course.checkpoints[cp_next]
                if np.hypot(data.qpos[0] - cx, data.qpos[1] - cy) <= course.checkpoint_radius:
                    result.checkpoints += 1
                    result.checkpoint_times.append(round(t, 1))
                    cp_next += 1
                    recolor(cp_next if cp_next < len(site_ids) else len(site_ids))

            # obstacle/wall bump events (debounced per geom)
            for ci in range(data.ncon):
                con = data.contact[ci]
                g1, g2 = con.geom1, con.geom2
                label = None
                if g1 in robot_geoms and g2 in hit_geoms:
                    label = hit_geoms[g2]
                elif g2 in robot_geoms and g1 in hit_geoms:
                    label = hit_geoms[g1]
                if label is None:
                    continue
                prev = last_contact_t.get(label)
                if prev is None or t - prev > BUMP_DEBOUNCE_S:
                    result.bumps += 1
                    result.bump_events.append({"t": round(t, 1), "with": label})
                last_contact_t[label] = t

            if renderer:
                renderer.maybe_frame(data, t)

            if result.checkpoints == len(course.checkpoints):
                result.completed = True
                result.completion_time_s = round(t, 1)
                break
            if fall.fallen:
                break
    finally:
        if renderer:
            renderer.close()
        if decisions_f:
            decisions_f.close()

    result.elapsed_s = round(t, 1)
    result.fell = fall.fallen
    result.fall_time_s = round(fall.fall_time, 1) if fall.fallen else None
    # Score: checkpoints * 100 - seconds. Unless the course was completed, the
    # full time budget counts — ending early (e.g. by falling) must not pay.
    elapsed_for_score = result.completion_time_s if result.completed else TIME_LIMIT_S
    result.score = round(result.checkpoints * 100 - elapsed_for_score, 1)
    result.wall_time_s = round(time.time() - t_wall, 1)

    if latencies:
        result.mean_decision_latency_s = round(float(np.mean(latencies)), 3)
        result.max_decision_latency_s = round(float(np.max(latencies)), 3)
    if t > 0:
        result.effective_decision_hz = round(result.decisions / t, 2)
    if mode == "realtime":
        result.realtime_max_lag_s = round(max_lag, 3)

    if log_dir:
        (log_dir / "episode.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
