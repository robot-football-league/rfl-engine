"""Baseline agents and the agent registry.

Every agent implements decide(observation: dict) -> dict with keys vx, vy, wz.
Optional hook: begin_episode(log_dir=...) called once per episode.
"""

from __future__ import annotations

import numpy as np

from .course import ARENA_HALF, GRID_RES, INFLATE, astar, occupancy_from_aabbs, to_cell, to_world
from .envelope import load_envelope
from .util import wrap_angle


class RandomAgent:
    """Floor baseline: uniform random command in the envelope every tick."""

    def __init__(self, seed: int = 0):
        self.name = "random"
        self.envelope = load_envelope()
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def begin_episode(self, log_dir=None):
        self.rng = np.random.default_rng(self._seed)

    def decide(self, obs: dict) -> dict:
        e = self.envelope
        return {
            "vx": float(self.rng.uniform(*e["vx"])),
            "vy": float(self.rng.uniform(*e["vy"])),
            "wz": float(self.rng.uniform(*e["wz"])),
        }


class ScriptedAgent:
    """Ceiling-ish baseline: grid A* to the next checkpoint + pure pursuit,
    with a naive stop-and-wait rule for moving obstacles.

    Plans only from the observation (obstacle AABBs), same information the
    LLM agents get — no access to the course spec.
    """

    LOOKAHEAD_M = 1.1
    TURN_GAIN = 2.2
    V_NOM = 1.0

    def __init__(self, seed: int = 0, arena_half: float = ARENA_HALF):
        self.name = "scripted"
        self.envelope = load_envelope()
        self.half = arena_half
        self.grid = None

    def begin_episode(self, log_dir=None):
        self.grid = None
        self.mover_hist = {}

    def _ensure_grid(self, obs):
        if self.grid is None:
            aabbs = [o["aabb"] for o in obs["obstacles"] if o["type"] == "static"]
            self.grid = occupancy_from_aabbs(aabbs, self.half)

    def _mover_bands(self, obs) -> list:
        """Learn each mover's swept band from observed centers; return AABBs.

        The observation only exposes the current AABB + velocity, so the band
        (travel segment) is estimated online and grows as more sweep is seen.
        """
        bands = []
        for ob in obs["obstacles"]:
            if ob["type"] != "moving":
                continue
            x0, y0, x1, y1 = ob["aabb"]
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            h = self.mover_hist.setdefault(ob["id"], {"origin": c, "axis": None,
                                                      "lo": 0.0, "hi": 0.0})
            v = np.array(ob.get("velocity", [0.0, 0.0]), dtype=float)
            if h["axis"] is None and np.linalg.norm(v) > 0.25:
                h["axis"] = v / np.linalg.norm(v)
            axis = h["axis"]
            if axis is None:
                bands.append([c[0] - 1.0, c[1] - 1.0, c[0] + 1.0, c[1] + 1.0])
                continue
            s = float(np.dot(c - h["origin"], axis))
            h["lo"] = min(h["lo"], s)
            h["hi"] = max(h["hi"], s)
            p_lo = h["origin"] + axis * (h["lo"] - 0.45)
            p_hi = h["origin"] + axis * (h["hi"] + 0.45)
            m = 0.65  # half-width across the travel axis (box half + margin)
            bands.append([min(p_lo[0], p_hi[0]) - m, min(p_lo[1], p_hi[1]) - m,
                         max(p_lo[0], p_hi[0]) + m, max(p_lo[1], p_hi[1]) + m])
        return bands

    @staticmethod
    def _nearest_free(grid, cell):
        """BFS ring search for the closest unblocked cell."""
        if not grid[cell]:
            return cell
        n = grid.shape[0]
        from collections import deque
        q = deque([cell])
        seen = {cell}
        while q:
            ci, cj = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (ci + di, cj + dj)
                if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or nxt in seen:
                    continue
                if not grid[nxt]:
                    return nxt
                seen.add(nxt)
                q.append(nxt)
        return cell

    def decide(self, obs: dict) -> dict:
        if obs["self"]["fallen"]:
            return {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        cmd = self._navigate(obs)
        safety = self._mover_safety(obs, cmd)
        return safety if safety is not None else cmd

    def _navigate(self, obs: dict) -> dict:
        e = self.envelope
        self._ensure_grid(obs)

        pos = np.array(obs["self"]["position"], dtype=float)
        heading = float(obs["self"]["heading_rad"])
        target = np.array(obs["next_checkpoint"]["position"], dtype=float)

        # plan on statics + each mover's learned swept band (virtual obstacle)
        bands = self._mover_bands(obs)
        grid = self.grid
        if bands:
            grid = self.grid | occupancy_from_aabbs(bands, self.half)
        start = self._nearest_free(grid, to_cell(pos, self.half))
        goal = self._nearest_free(grid, to_cell(target, self.half))
        path = astar(grid, start, goal)
        if path is None and bands:
            # Band swallows the only corridor: a timed crossing is required.
            # Prefer crossing near the band's ends (where a periodic hazard
            # dwells and the safety window is longest) by keeping only the
            # middle 55% of each band blocked; fall back to bare statics.
            cores = []
            for (x0, y0, x1, y1) in bands:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                hx, hy = 0.55 * (x1 - x0) / 2, 0.55 * (y1 - y0) / 2
                cores.append([cx - hx, cy - hy, cx + hx, cy + hy])
            core_grid = self.grid | occupancy_from_aabbs(cores, self.half)
            start = self._nearest_free(core_grid, to_cell(pos, self.half))
            goal = self._nearest_free(core_grid, to_cell(target, self.half))
            path = astar(core_grid, start, goal)
            if path is None:
                start = self._nearest_free(self.grid, to_cell(pos, self.half))
                goal = self._nearest_free(self.grid, to_cell(target, self.half))
                path = astar(self.grid, start, goal)
        if path is None:
            look = target  # cannot happen on a feasible course; slow probe below
        else:
            pts = [np.array(to_world(c, self.half)) for c in path]
            look = pts[-1]
            acc = 0.0
            prev = pos
            for w in pts:
                acc += float(np.linalg.norm(w - prev))
                prev = w
                if acc >= self.LOOKAHEAD_M:
                    look = w
                    break

        err = wrap_angle(float(np.arctan2(look[1] - pos[1], look[0] - pos[0])) - heading)
        wz = float(np.clip(self.TURN_GAIN * err, *e["wz"]))
        if abs(err) > 1.1:
            vx = 0.05
        elif path is None:
            vx = 0.3
        else:
            vx = float(np.clip(self.V_NOM * np.cos(err), 0.0, e["vx"][1]))
        return {"vx": vx, "vy": 0.0, "wz": wz}

    def _mover_safety(self, obs: dict, cmd: dict):
        """Naive moving-obstacle rule: stop if our straight-line future and the
        mover's constant-velocity future come close; back away if it is
        already close and closing (waiting in its lane is fatal). Returns an
        override command or None."""
        e = self.envelope
        pos = np.array(obs["self"]["position"], dtype=float)
        heading = float(obs["self"]["heading_rad"])
        u = np.array([np.cos(heading), np.sin(heading)])
        for ob in obs["obstacles"]:
            if ob["type"] != "moving":
                continue
            x0, y0, x1, y1 = ob["aabb"]
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            v = np.array(ob.get("velocity", [0.0, 0.0]), dtype=float)
            rel = c - pos
            d = float(np.linalg.norm(rel))
            if d > 5.0:
                continue
            closing = float(np.dot(v, -rel / max(d, 1e-6)))
            if d < 1.9 and closing > 0.1:
                back = wrap_angle(float(np.arctan2(-rel[1], -rel[0])) - heading)
                return {"vx": float(np.clip(0.55 * np.cos(back), *e["vx"])),
                        "vy": float(np.clip(0.55 * np.sin(back), *e["vy"])),
                        "wz": 0.0}
            for tau in np.arange(0.0, 3.1, 0.3):
                mine = pos + u * min(0.8 * tau, 2.4)
                theirs = c + v * tau
                if np.linalg.norm(mine - theirs) < 1.6:
                    return {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        return None


class OracleAgent(ScriptedAgent):
    """CHEATING course-certification baseline — never a benchmark entrant.

    Reads the true hazard kinematics from the course spec and gates its
    navigation on exact future clearance. Its job is to prove a course is
    completable with perfect timing; benchmark agents only ever see the
    observation JSON.
    """

    MARGIN_M = 0.7
    ACCEL = 0.8      # observed standing-start acceleration, m/s^2
    SIM_DT = 0.2
    SIM_MAX_S = 6.0

    def __init__(self, course, seed: int = 0):
        super().__init__(seed=seed, arena_half=course.arena_half)
        self.name = "oracle"
        self.course = course

    def decide(self, obs: dict) -> dict:
        from .episode import TIME_LIMIT_S
        if obs["self"]["fallen"]:
            return {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        cmd = self._navigate(obs)
        t = TIME_LIMIT_S - obs["time_remaining_s"]
        pos = np.array(obs["self"]["position"], dtype=float)
        heading = float(obs["self"]["heading_rad"])

        # Committed: already inside a hazard's sweep envelope. Never stop
        # there — stopping mid-lane converts a near-miss into a hit. Sprint
        # through on the current pursuit line with gentle steering only.
        if any(self._in_lane(ob, pos) for ob in self.course.movers):
            return {"vx": self.envelope["vx"][1], "vy": 0.0,
                    "wz": float(np.clip(cmd["wz"], -0.5, 0.5))}

        # At standoff: simulate the transit with honest acceleration from the
        # actual current speed; enter only if the whole transit stays clear.
        c, s = np.cos(heading), np.sin(heading)
        u = np.array([c * cmd["vx"] - s * cmd["vy"],
                      s * cmd["vx"] + c * cmd["vy"]])
        speed_cmd = float(np.linalg.norm(u))
        if speed_cmd < 0.05:
            return cmd  # turning in place outside every lane is safe
        u_dir = u / speed_cmd
        if not self._transit_clear(pos, u_dir, obs, t):
            # hold at the standoff, keep aligning with the path
            return {"vx": 0.0, "vy": 0.0,
                    "wz": float(np.clip(cmd["wz"], -0.8, 0.8))}
        return cmd

    def _transit_clear(self, pos, u_dir, obs, t: float) -> bool:
        v = float(np.linalg.norm(obs["self"]["velocity"]))
        vmax = self.envelope["vx"][1]
        p = pos.copy()
        entered = False
        for k in range(int(self.SIM_MAX_S / self.SIM_DT)):
            v = min(vmax, v + self.ACCEL * self.SIM_DT)
            p = p + u_dir * v * self.SIM_DT
            tt = t + (k + 1) * self.SIM_DT
            in_any = False
            for ob in self.course.movers:
                if self._in_lane(ob, p):
                    in_any = True
                    if self._danger(ob, p, tt):
                        return False
            if entered and not in_any:
                return True  # crossed out the far side untouched
            entered = entered or in_any
        return True

    def _in_lane(self, ob, p) -> bool:
        """Inside the region the hazard ever sweeps (plus margin)?"""
        m = self.MARGIN_M
        if ob.kind == "sweeper":
            return float(np.hypot(p[0] - ob.x, p[1] - ob.y)) < ob.sx / 2 + m
        c0 = np.array([ob.x, ob.y])
        a = np.array(ob.axis, dtype=float)
        rel = p - c0
        along = float(np.dot(rel, a))
        perp = float(np.linalg.norm(rel - a * along))
        reach = (ob.amplitude if ob.kind == "moving"
                 else ob.radius * np.sin(ob.swing))
        return (abs(along) < reach + ob.circumradius + m and
                perp < ob.circumradius + m)

    def _danger(self, ob, p, t: float) -> bool:
        m = self.MARGIN_M
        if ob.kind == "sweeper":
            center = np.array([ob.x, ob.y])
            rel = p - center
            r = float(np.linalg.norm(rel))
            half = ob.sx / 2
            if r > half + m:
                return False
            th = ob._theta(t)
            d = np.array([np.cos(th), np.sin(th)])
            along = float(np.clip(np.dot(rel, d), -half, half))
            return float(np.linalg.norm(p - (center + d * along))) < m
        return float(np.linalg.norm(p - ob.center_at(t))) < m + ob.circumradius


def make_agent(spec: str, seed: int = 0, run_dir=None, prompt: str | None = None,
               arena_half: float = ARENA_HALF, course=None,
               history_n: int | None = None):
    """spec: 'random' | 'scripted' | 'oracle' | 'llm:<provider>:<model>'

    prompt selects the LLM system prompt variant (e.g. 'race_v1' for races);
    arena_half sizes the scripted agent's planning grid; course enables the
    oracle and (for LLMs) bakes the static layout into the system prompt so
    per-tick observations only carry moving hazards; history_n tunes the LLM
    context window (cost knob). Baselines ignore what doesn't apply.
    """
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec == "scripted":
        return ScriptedAgent(seed=seed, arena_half=arena_half)
    if spec == "oracle":
        if course is None:
            raise ValueError("oracle needs the course spec (certification only)")
        return OracleAgent(course, seed=seed)
    if spec in ("vint", "nomad"):
        if course is None:
            raise ValueError(f"{spec} needs the course spec (builds its topological map)")
        from .visnav_agent import VisNavAgent
        return VisNavAgent(course, model_type=spec, seed=seed)
    if spec.startswith("llm:"):
        import json
        from .llm import LLMAgent
        _, provider, model = spec.split(":", 2)
        kwargs = {}
        if prompt:
            kwargs["prompt"] = prompt
        if history_n is not None:
            kwargs["history_n"] = history_n
        if course is not None:
            statics = [{"id": o.id,
                        "aabb": [round(float(v), 2) for v in o.aabb_at(0)]}
                       for o in course.statics]
            kwargs["static_block"] = json.dumps(statics, separators=(",", ":"))
        return LLMAgent(provider=provider, model=model, **kwargs)
    raise ValueError(f"unknown agent spec: {spec}")
