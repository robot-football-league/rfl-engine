"""Course spec, seeded generation, and grid-based feasibility checking.

A course is fully described by a JSON-serializable CourseSpec: same seed ->
identical spec -> identical MJCF world. Feasibility is verified on a 0.1 m
occupancy grid built from obstacle AABBs inflated by 0.6 m, which guarantees
a >= 1.2 m-wide corridor between consecutive checkpoints. The fork's narrow
gap (0.9-1.1 m) closes under that inflation, so the guaranteed corridor is
always the detour — the gap is the optional risky shortcut.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

ARENA_HALF = 10.0
CHECKPOINT_RADIUS = 1.5
GRID_RES = 0.1
INFLATE = 0.6  # robot clearance radius used for feasibility + planning


KINETIC_KINDS = ("moving", "sweeper", "pendulum")


@dataclass
class Obstacle:
    id: int
    kind: str  # "static" | "moving" | "sweeper" | "pendulum"
    shape: str  # "box" | "cylinder"
    x: float
    y: float
    sx: float  # full extent along local x (diameter for cylinders; boom length for sweepers)
    sy: float
    height: float
    yaw: float = 0.0
    role: str = "scatter"  # "scatter" | "fork" | "mover" | "hazard" | "wall"
    # kinetic parameters:
    #  moving:   center = (x, y) + axis * amplitude * sin(2*pi*t/period + phase)
    #  sweeper:  boom centered on post (x, y, z), yaw = 2*pi*t/period + phase
    #  pendulum: bob swings on rod of length `radius` from pivot (x, y, z) along
    #            `axis`, angle = swing * sin(2*pi*t/period + phase)
    axis: tuple[float, float] | None = None
    amplitude: float = 0.0
    period: float = 0.0
    phase: float = 0.0
    z: float | None = None
    radius: float = 0.0
    swing: float = 0.0

    # ---------------------------------------------------------------- kinematics

    def _theta(self, t: float) -> float:
        if self.kind == "sweeper":
            return 2 * np.pi * t / self.period + self.phase
        if self.kind == "pendulum":
            return self.swing * np.sin(2 * np.pi * t / self.period + self.phase)
        return 0.0

    def center_at(self, t: float) -> np.ndarray:
        if self.kind == "moving":
            s = self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
            return np.array([self.x + self.axis[0] * s, self.y + self.axis[1] * s])
        if self.kind == "pendulum":
            th = self._theta(t)
            off = self.radius * np.sin(th)
            return np.array([self.x + self.axis[0] * off, self.y + self.axis[1] * off])
        return np.array([self.x, self.y])

    def pose_at(self, t: float):
        """(pos3, quat4) for the mocap body of a kinetic obstacle."""
        if self.kind == "moving":
            c = self.center_at(t)
            return (c[0], c[1], self.height / 2), (1.0, 0.0, 0.0, 0.0)
        if self.kind == "sweeper":
            th = self._theta(t)
            return ((self.x, self.y, self.z),
                    (np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)))
        if self.kind == "pendulum":
            th = self._theta(t)
            c = self.center_at(t)
            zc = self.z - self.radius * np.cos(th)
            # rotate about the horizontal axis perpendicular to the swing direction
            rx, ry = self.axis[1], -self.axis[0]
            s = np.sin(th / 2)
            return (c[0], c[1], zc), (np.cos(th / 2), s * rx, s * ry, 0.0)
        raise ValueError(f"pose_at on kind={self.kind}")

    def velocity_at(self, t: float) -> np.ndarray:
        if self.kind == "moving":
            w = 2 * np.pi / self.period
            ds = self.amplitude * w * np.cos(w * t + self.phase)
            return np.array([self.axis[0] * ds, self.axis[1] * ds])
        if self.kind == "pendulum":
            w = 2 * np.pi / self.period
            th_dot = self.swing * w * np.cos(w * t + self.phase)
            th = self._theta(t)
            ds = self.radius * np.cos(th) * th_dot
            return np.array([self.axis[0] * ds, self.axis[1] * ds])
        return np.zeros(2)

    def aabb_at(self, t: float = 0.0) -> tuple[float, float, float, float]:
        if self.kind == "sweeper":
            th = self._theta(t)
            hx, hy = self.sx / 2, self.sy / 2
            c, s = abs(np.cos(th)), abs(np.sin(th))
            ex, ey = hx * c + hy * s, hx * s + hy * c
            return (self.x - ex, self.y - ey, self.x + ex, self.y + ey)
        cx, cy = self.center_at(t)
        if self.shape == "cylinder":
            r = self.sx / 2
            return (cx - r, cy - r, cx + r, cy + r)
        hx, hy = self.sx / 2, self.sy / 2
        c, s = abs(np.cos(self.yaw)), abs(np.sin(self.yaw))
        ex = hx * c + hy * s
        ey = hx * s + hy * c
        return (cx - ex, cy - ey, cx + ex, cy + ey)

    def obs_entries(self, t: float) -> list[dict]:
        """Observation JSON entries. A sweeper reports its two arm tips as
        separate 'moving' obstacles (ids id and id+100) so the schema stays
        uniform: every hazard is an AABB with a velocity."""
        if self.kind == "static":
            return [{"id": self.id, "type": "static",
                     "aabb": [round(float(v), 2) for v in self.aabb_at(t)]}]
        if self.kind == "sweeper":
            entries = []
            th = self._theta(t)
            w = 2 * np.pi / self.period
            half = self.sx / 2
            pad = self.sy / 2 + 0.05
            for k, sign in ((0, 1.0), (100, -1.0)):
                u = np.array([np.cos(th), np.sin(th)]) * sign
                p_in = np.array([self.x, self.y]) + u * (0.25 * half)
                p_tip = np.array([self.x, self.y]) + u * half
                lo = np.minimum(p_in, p_tip) - pad
                hi = np.maximum(p_in, p_tip) + pad
                mid = np.array([self.x, self.y]) + u * (0.65 * half)
                tangent = np.array([-u[1], u[0]]) * (w * 0.65 * half)
                entries.append({
                    "id": self.id + k, "type": "moving",
                    "aabb": [round(float(v), 2) for v in (lo[0], lo[1], hi[0], hi[1])],
                    "velocity": [round(float(tangent[0]), 2), round(float(tangent[1]), 2)],
                })
            return entries
        v = self.velocity_at(t)
        return [{"id": self.id, "type": "moving",
                 "aabb": [round(float(x), 2) for x in self.aabb_at(t)],
                 "velocity": [round(float(v[0]), 2), round(float(v[1]), 2)]}]

    @property
    def circumradius(self) -> float:
        return float(np.hypot(self.sx / 2, self.sy / 2))


@dataclass
class CourseSpec:
    seed: int
    spawn: tuple[float, float, float]  # x, y, yaw
    checkpoints: list[tuple[float, float]]
    obstacles: list[Obstacle]
    arena_half: float = ARENA_HALF
    checkpoint_radius: float = CHECKPOINT_RADIUS
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CourseSpec":
        obs = [Obstacle(**{**o, "axis": tuple(o["axis"]) if o.get("axis") else None})
               for o in d["obstacles"]]
        return cls(
            seed=d["seed"], spawn=tuple(d["spawn"]),
            checkpoints=[tuple(c) for c in d["checkpoints"]],
            obstacles=obs, arena_half=d.get("arena_half", ARENA_HALF),
            checkpoint_radius=d.get("checkpoint_radius", CHECKPOINT_RADIUS),
            meta=d.get("meta", {}),
        )

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CourseSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def movers(self) -> list[Obstacle]:
        """All kinetic obstacles (mocap-driven): sliders, sweepers, pendulums."""
        return [o for o in self.obstacles if o.kind in KINETIC_KINDS]

    @property
    def statics(self) -> list[Obstacle]:
        return [o for o in self.obstacles if o.kind == "static"]


# ---------------------------------------------------------------- occupancy / A*

def occupancy_from_aabbs(aabbs, arena_half=ARENA_HALF, res=GRID_RES, inflate=INFLATE):
    """Boolean occupancy grid; True = blocked. Includes inflated arena walls."""
    n = int(round(2 * arena_half / res))
    grid = np.zeros((n, n), dtype=bool)
    k = max(1, int(np.ceil(inflate / res)))
    grid[:k, :] = True
    grid[-k:, :] = True
    grid[:, :k] = True
    grid[:, -k:] = True
    for (x0, y0, x1, y1) in aabbs:
        i0 = max(0, int(np.floor((x0 - inflate + arena_half) / res)))
        i1 = min(n - 1, int(np.ceil((x1 + inflate + arena_half) / res)))
        j0 = max(0, int(np.floor((y0 - inflate + arena_half) / res)))
        j1 = min(n - 1, int(np.ceil((y1 + inflate + arena_half) / res)))
        if i1 >= i0 and j1 >= j0:
            grid[i0 : i1 + 1, j0 : j1 + 1] = True
    return grid


def to_cell(p, arena_half=ARENA_HALF, res=GRID_RES):
    n = int(round(2 * arena_half / res))
    i = int(np.clip((p[0] + arena_half) / res, 0, n - 1))
    j = int(np.clip((p[1] + arena_half) / res, 0, n - 1))
    return i, j


def to_world(cell, arena_half=ARENA_HALF, res=GRID_RES):
    return ((cell[0] + 0.5) * res - arena_half, (cell[1] + 0.5) * res - arena_half)


_NBRS = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
         (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]


def astar(grid: np.ndarray, start, goal):
    """8-connected A*; returns list of cells (start..goal) or None."""
    n = grid.shape[0]
    if grid[start] or grid[goal]:
        return None
    g = {start: 0.0}
    parent = {}
    h0 = np.hypot(goal[0] - start[0], goal[1] - start[1])
    heap = [(h0, 0.0, start)]
    closed = set()
    while heap:
        f, gc, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        ci, cj = cur
        for di, dj, w in _NBRS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n and 0 <= nj < n) or grid[ni, nj]:
                continue
            nxt = (ni, nj)
            ng = gc + w
            if ng < g.get(nxt, np.inf):
                g[nxt] = ng
                parent[nxt] = cur
                h = np.hypot(goal[0] - ni, goal[1] - nj)
                heapq.heappush(heap, (ng + h, ng, nxt))
    return None


def check_feasible(spec: CourseSpec) -> tuple[bool, list[float]]:
    """Static-obstacle A* over every leg (spawn->cp0->...->cp4). Movers excluded."""
    grid = occupancy_from_aabbs([o.aabb_at(0) for o in spec.statics],
                                spec.arena_half)
    pts = [spec.spawn[:2]] + [tuple(c) for c in spec.checkpoints]
    lengths = []
    for a, b in zip(pts[:-1], pts[1:]):
        path = astar(grid, to_cell(a, spec.arena_half), to_cell(b, spec.arena_half))
        if path is None:
            return False, []
        lengths.append(round(len(path) * GRID_RES, 1))
    return True, lengths


# ---------------------------------------------------------------- generation

def _perp(u):
    return np.array([-u[1], u[0]])


def generate_course(seed: int, max_tries: int = 300) -> CourseSpec:
    rng = np.random.default_rng(seed)
    for attempt in range(max_tries):
        spec = _try_generate(rng, seed)
        if spec is None:
            continue
        ok, leg_lengths = check_feasible(spec)
        if ok:
            spec.meta["attempts"] = attempt + 1
            spec.meta["leg_path_lengths_m"] = leg_lengths
            return spec
    raise RuntimeError(f"course generation failed for seed {seed} after {max_tries} tries")


def _try_generate(rng: np.random.Generator, seed: int) -> CourseSpec | None:
    half = ARENA_HALF

    # --- spawn near one wall, facing inward
    side = int(rng.integers(4))  # 0:S 1:N 2:W 3:E
    along = float(rng.uniform(-5.0, 5.0))
    depth = float(rng.uniform(1.8, 2.5))
    if side == 0:
        spawn_xy = (along, -half + depth); base_yaw = np.pi / 2
    elif side == 1:
        spawn_xy = (along, half - depth); base_yaw = -np.pi / 2
    elif side == 2:
        spawn_xy = (-half + depth, along); base_yaw = 0.0
    else:
        spawn_xy = (half - depth, along); base_yaw = np.pi
    yaw = float(base_yaw + rng.uniform(-0.5, 0.5))
    spawn = (spawn_xy[0], spawn_xy[1], yaw)

    # --- 5 checkpoints as a wandering path across the arena
    checkpoints: list[tuple[float, float]] = []
    prev = np.array(spawn_xy)
    prev_dir = yaw
    for _ in range(5):
        placed = False
        for _ in range(60):
            d = float(rng.uniform(4.5, 7.0))
            dth = float(rng.uniform(-1.2, 1.2))
            ang = prev_dir + dth
            cand = prev + d * np.array([np.cos(ang), np.sin(ang)])
            if np.max(np.abs(cand)) > half - 2.5:
                continue
            if any(np.linalg.norm(cand - np.array(c)) < 3.5 for c in checkpoints):
                continue
            if np.linalg.norm(cand - np.array(spawn_xy)) < 3.5:
                continue
            checkpoints.append((float(cand[0]), float(cand[1])))
            prev_dir = float(np.arctan2(cand[1] - prev[1], cand[0] - prev[0]))
            prev = cand
            placed = True
            break
        if not placed:
            return None

    legs = list(zip([spawn_xy] + checkpoints[:-1], checkpoints))
    leg_vecs = [(np.array(a), np.array(b)) for a, b in legs]
    leg_lens = [float(np.linalg.norm(b - a)) for a, b in leg_vecs]

    obstacles: list[Obstacle] = []
    next_id = 0

    # --- fork: wall with a narrow gap across one long leg
    fork_candidates = [i for i, (ab, L) in enumerate(zip(leg_vecs, leg_lens))
                       if L >= 5.5 and np.max(np.abs((ab[0] + ab[1]) / 2)) <= 6.0]
    if not fork_candidates:
        return None
    fork_leg = int(rng.choice(fork_candidates))
    a, b = leg_vecs[fork_leg]
    u = (b - a) / leg_lens[fork_leg]
    nvec = _perp(u) * (1 if rng.random() < 0.5 else -1)
    M = (a + b) / 2
    gap = float(rng.uniform(0.9, 1.1))

    def _extent_to_boundary(origin, direction):
        """Distance from origin along direction to the arena boundary."""
        dists = []
        for k in range(2):
            if abs(direction[k]) > 1e-9:
                for lim in (half, -half):
                    t = (lim - origin[k]) / direction[k]
                    if t > 0:
                        dists.append(t)
        return min(dists) if dists else 0.0

    side_max_p = _extent_to_boundary(M, nvec) - 2.6
    side_max_m = _extent_to_boundary(M, -nvec) - 2.6
    side_p = min(2.8, side_max_p - gap / 2)
    side_m = min(2.8, side_max_m - gap / 2)
    if side_p < 1.6 or side_m < 1.6:
        return None
    wall_yaw = float(np.arctan2(nvec[1], nvec[0]))
    for sgn, side_len in ((1, side_p), (-1, side_m)):
        c = M + sgn * nvec * (gap / 2 + side_len / 2)
        obstacles.append(Obstacle(
            id=next_id, kind="static", shape="box",
            x=float(c[0]), y=float(c[1]), sx=float(side_len), sy=0.22,
            height=0.8, yaw=wall_yaw, role="fork"))
        next_id += 1
    fork_meta = {"leg": fork_leg, "gap_m": round(gap, 2),
                 "gap_center": [round(float(M[0]), 2), round(float(M[1]), 2)]}

    # --- moving obstacle crossing a different leg
    mover_candidates = [i for i, L in enumerate(leg_lens) if i != fork_leg and L >= 4.6]
    if not mover_candidates:
        return None
    mover_leg = int(rng.choice(mover_candidates))
    a, b = leg_vecs[mover_leg]
    u = (b - a) / leg_lens[mover_leg]
    axis = _perp(u)
    frac = float(rng.uniform(0.42, 0.58))
    P = a + u * frac * leg_lens[mover_leg]
    amp_max = min(_extent_to_boundary(P, axis), _extent_to_boundary(P, -axis)) - 2.2
    amplitude = float(min(rng.uniform(1.5, 2.5), amp_max))
    if amplitude < 1.2:
        return None
    period = float(rng.uniform(4.0, 8.0))
    phase = float(rng.uniform(0, 2 * np.pi))
    mover = Obstacle(
        id=next_id, kind="moving", shape="box",
        x=float(P[0]), y=float(P[1]), sx=0.8, sy=0.8, height=0.9,
        role="mover", axis=(float(axis[0]), float(axis[1])),
        amplitude=amplitude, period=period, phase=phase)
    next_id += 1
    # mover sweep must stay clear of the fork wall
    sweep = [mover.center_at(t) for t in np.linspace(0, period, 16, endpoint=False)]
    for ob in obstacles:
        x0, y0, x1, y1 = ob.aabb_at(0)
        for c in sweep:
            if x0 - 1.2 < c[0] < x1 + 1.2 and y0 - 1.2 < c[1] < y1 + 1.2:
                return None
    obstacles.append(mover)

    # --- scattered static obstacles
    n_scatter = int(rng.integers(8, 16)) - 2  # fork wall counts as 2 obstacle geoms
    keep_clear = [np.array(spawn_xy)] + [np.array(c) for c in checkpoints]
    for _ in range(n_scatter):
        for _ in range(40):
            shape = "box" if rng.random() < 0.6 else "cylinder"
            if shape == "box":
                sx, sy = float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.5, 2.0))
            else:
                sx = sy = float(rng.uniform(0.5, 1.6))
            h = float(rng.uniform(0.5, 1.2))
            oyaw = float(rng.uniform(0, np.pi))
            pos = rng.uniform(-(half - 1.3), half - 1.3, size=2)
            ob = Obstacle(id=next_id, kind="static", shape=shape,
                          x=float(pos[0]), y=float(pos[1]), sx=sx, sy=sy,
                          height=h, yaw=oyaw)
            cr = ob.circumradius
            if any(np.linalg.norm(pos - p) < CHECKPOINT_RADIUS + cr + 0.4 for p in keep_clear):
                continue
            if np.linalg.norm(pos - M) < cr + 2.0:  # keep the fork gap decision clean
                continue
            if any(np.linalg.norm(pos - c) < cr + 1.5 for c in sweep):
                continue
            x0, y0, x1, y1 = ob.aabb_at(0)
            overlap = False
            for other in obstacles:
                if other.kind == "moving":
                    continue
                ox0, oy0, ox1, oy1 = other.aabb_at(0)
                if x0 - 0.5 < ox1 and x1 + 0.5 > ox0 and y0 - 0.5 < oy1 and y1 + 0.5 > oy0:
                    overlap = True
                    break
            if overlap:
                continue
            obstacles.append(ob)
            next_id += 1
            break
        # placement failure for one scatter obstacle is fine; course still valid

    n_static = len([o for o in obstacles if o.kind == "static"])
    if n_static < 8:
        return None

    return CourseSpec(
        seed=seed, spawn=spawn, checkpoints=checkpoints, obstacles=obstacles,
        meta={"fork": fork_meta,
              "mover": {"leg": mover_leg, "period_s": round(period, 1),
                        "amplitude_m": round(amplitude, 1)}},
    )


# ---------------------------------------------------------------- hand-authored courses

def hand_courses() -> dict[str, CourseSpec]:
    """Three fixed smoke-test courses with deliberate geometry."""

    def mk(name, spawn, cps, obs, meta):
        return CourseSpec(seed=-1, spawn=spawn, checkpoints=cps, obstacles=obs,
                          meta={"hand": name, **meta})

    a = mk(
        "hand_a_slalom",
        (-8.0, -8.0, np.pi / 4),
        [(-4.0, -4.0), (0.0, -1.0), (4.0, 1.0), (6.5, 5.0), (2.0, 7.5)],
        [
            Obstacle(0, "static", "box", -2.0, -2.8, 1.6, 0.8, 0.9, yaw=0.3),
            Obstacle(1, "static", "cylinder", 2.0, 0.4, 1.2, 1.2, 1.0),
            Obstacle(2, "static", "box", -5.5, -1.5, 1.5, 1.5, 0.6),
            Obstacle(3, "static", "box", 0.5, 3.5, 2.0, 0.7, 1.1, yaw=1.2),
            Obstacle(4, "static", "cylinder", -3.0, 4.0, 0.9, 0.9, 0.8),
            Obstacle(5, "static", "box", 7.5, -2.0, 1.2, 1.2, 0.7),
            Obstacle(6, "static", "cylinder", 5.0, -5.5, 1.4, 1.4, 1.0),
            Obstacle(7, "static", "box", -7.0, 2.5, 1.8, 0.9, 0.9, yaw=-0.5),
            # fork wall across the cp2->cp3 leg (midpoint ~ (5.25, 3.0)), gap 1.0
            Obstacle(8, "static", "box", 3.55, 4.05, 3.0, 0.22, 0.8, yaw=-0.55, role="fork"),
            Obstacle(9, "static", "box", 7.0, 1.85, 3.0, 0.22, 0.8, yaw=-0.55, role="fork"),
            # mover across the cp3->cp4 leg
            Obstacle(10, "moving", "box", 4.25, 6.25, 0.8, 0.8, 0.9, role="mover",
                     axis=(0.485, 0.874), amplitude=2.0, period=5.0, phase=0.0),
        ],
        {"fork": {"leg": 3, "gap_m": 1.0, "gap_center": [5.25, 3.0]},
         "mover": {"leg": 4, "period_s": 5.0, "amplitude_m": 2.0}},
    )

    b = mk(
        "hand_b_fork_first",
        (0.0, -8.5, np.pi / 2),
        [(0.0, -2.5), (-3.5, 1.5), (0.5, 5.0), (5.0, 5.5), (7.5, 1.0)],
        [
            # fork wall across the spawn->cp0 leg (midpoint (0, -5.5)), gap 0.95
            Obstacle(0, "static", "box", -1.9, -5.5, 2.9, 0.22, 0.8, role="fork"),
            Obstacle(1, "static", "box", 1.9, -5.5, 2.9, 0.22, 0.8, role="fork"),
            Obstacle(2, "static", "box", -1.5, -0.5, 1.4, 1.4, 1.0, yaw=0.8),
            Obstacle(3, "static", "cylinder", -5.5, 3.5, 1.3, 1.3, 0.9),
            Obstacle(4, "static", "box", -1.5, 3.8, 1.8, 0.8, 0.7, yaw=0.2),
            Obstacle(5, "static", "cylinder", 2.5, 2.0, 1.0, 1.0, 1.1),
            Obstacle(6, "static", "box", 2.8, 7.8, 1.6, 1.0, 0.8, yaw=-0.4),
            Obstacle(7, "static", "cylinder", 6.0, 3.2, 1.1, 1.1, 0.6),
            Obstacle(8, "static", "box", -6.5, -3.0, 2.0, 1.0, 1.0, yaw=1.0),
            Obstacle(9, "static", "box", 5.5, -3.5, 1.2, 1.2, 0.9),
            # mover across the cp1->cp2 leg (midpoint ~ (-1.5, 3.25)) -> crossing diagonal
            Obstacle(10, "moving", "box", -1.7, 3.1, 0.8, 0.8, 0.9, role="mover",
                     axis=(-0.658, 0.753), amplitude=1.8, period=6.0, phase=1.57),
        ],
        {"fork": {"leg": 0, "gap_m": 0.95, "gap_center": [0.0, -5.5]},
         "mover": {"leg": 2, "period_s": 6.0, "amplitude_m": 1.8}},
    )

    c = mk(
        "hand_c_timing",
        (8.0, 8.0, -3 * np.pi / 4),
        [(3.0, 3.5), (-2.0, 6.0), (-6.5, 2.0), (-3.0, -3.5), (3.0, -6.5)],
        [
            Obstacle(0, "static", "cylinder", 5.8, 5.8, 1.2, 1.2, 1.0),
            Obstacle(1, "static", "box", 0.5, 5.2, 1.8, 0.9, 0.8, yaw=0.5),
            Obstacle(2, "static", "box", -4.5, 4.6, 1.4, 1.4, 1.1),
            Obstacle(3, "static", "cylinder", -5.0, -1.5, 1.5, 1.5, 0.7),
            Obstacle(4, "static", "box", 0.0, -4.8, 2.0, 0.8, 0.9, yaw=-0.8),
            Obstacle(5, "static", "cylinder", 1.0, 0.0, 1.0, 1.0, 1.0),
            Obstacle(6, "static", "box", -8.0, 5.5, 1.2, 1.2, 0.6),
            Obstacle(7, "static", "box", 6.0, -1.5, 1.5, 1.0, 0.8, yaw=0.3),
            # fork across cp2->cp3 (midpoint ~ (-4.75, -0.75)), gap 1.05
            Obstacle(8, "static", "box", -6.6, -2.0, 2.6, 0.22, 0.8, yaw=0.99, role="fork"),
            Obstacle(9, "static", "box", -2.9, 0.5, 2.6, 0.22, 0.8, yaw=0.99, role="fork"),
            # mover across cp3->cp4 (midpoint (0, -5)) sliding along the leg normal
            Obstacle(10, "moving", "box", 0.0, -5.0, 0.8, 0.8, 0.9, role="mover",
                     axis=(0.447, 0.894), amplitude=2.2, period=4.5, phase=3.14),
        ],
        {"fork": {"leg": 3, "gap_m": 1.05, "gap_center": [-4.75, -0.75]},
         "mover": {"leg": 4, "period_s": 4.5, "amplitude_m": 2.2}},
    )
    return {"hand_a": a, "hand_b": b, "hand_c": c}
