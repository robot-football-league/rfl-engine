"""Disposable post-goal physics and render-only geometry; never advances a policy.

Tracks contain world, model bodies (in MuJoCo order), then effect_geometry().
Robot bodies are pelvis/chest body IDs or names, one per robot. Controllers,
when supplied, expose bound qpos_idx/qvel_idx/ctrl_idx and kps/kds (or kp/kd).
No controller method is called. All distances and times use metres/seconds.
"""

import copy

import mujoco
import numpy as np

DURATION_S = 1.6
HZ = 25
_RADIUS = 3.0
_SPEED = 5.0
_CONFETTI = 32
_RING = 40
_PARK = (0.0, 0.0, -30.0)


def effect_geometry():
    """Independent geometry descriptions, ordered exactly as effect_pose()."""
    palette = [(1., .72, .12, 1.), (1., .28, .12, 1.),
               (.15, .75, 1., 1.), (1., .95, .65, 1.)]
    geometry = [dict(name=f"goal_confetti_{i}", type="box",
                     size=[.035, .018, .004], rgba=list(palette[i % 4]))
                for i in range(_CONFETTI)]
    geometry += [dict(name=f"goal_ring_{i}", type="box",
                      size=[.24, .025, .018], rgba=[1., .75, .25, .8])
                 for i in range(_RING)]
    geometry.append(dict(name="goal_flash", type="sphere", size=[.3, 0., 0.],
                         rgba=[1., .85, .55, 1.]))
    return geometry


def effect_pose(age, origin, seed=0):
    """Ballistic confetti, a 5 m/s tangent ring, and ONE 120 ms warm flash.

    Geometry outside its lifetime is parked, never scaled or strobed. The ring
    disappears at 3 m, precisely the impulse cutoff. Quaternions are wxyz.
    """
    count = _CONFETTI + _RING + 1
    pos = np.tile(_PARK, (count, 1))
    quat = np.tile([1., 0., 0., 0.], (count, 1))
    if not 0 <= age < DURATION_S:
        return pos, quat
    origin = np.asarray(origin, dtype=float)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, _CONFETTI)
    speed = rng.uniform(.8, 2.5, _CONFETTI)
    velocity = np.column_stack((speed * np.cos(theta), speed * np.sin(theta),
                                rng.uniform(3., 5.5, _CONFETTI)))
    pos[:_CONFETTI] = origin + age * velocity
    pos[:_CONFETTI, 2] -= .5 * 9.81 * age**2
    axis = rng.normal(size=(_CONFETTI, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    angle = rng.uniform(0, 2 * np.pi, _CONFETTI) + age * rng.uniform(8, 22, _CONFETTI)
    quat[:_CONFETTI, 0] = np.cos(angle / 2)
    quat[:_CONFETTI, 1:] = axis * np.sin(angle / 2)[:, None]
    landed = np.flatnonzero(pos[:_CONFETTI, 2] < .02)
    pos[landed] = _PARK
    quat[landed] = (1., 0., 0., 0.)
    if age * _SPEED < _RADIUS:
        theta = np.arange(_RING) * (2 * np.pi / _RING)
        ring = slice(_CONFETTI, _CONFETTI + _RING)
        pos[ring, :2] = origin[:2] + age * _SPEED * np.column_stack((np.cos(theta), np.sin(theta)))
        pos[ring, 2] = .04
        quat[ring, 0] = np.cos((theta + np.pi / 2) / 2)
        quat[ring, 3] = np.sin((theta + np.pi / 2) / 2)
    if age < .12:
        pos[-1] = origin
    return pos, quat


def _held_pd(data, controllers):
    held = []
    for controller in (() if controllers is None else controllers):
        # Do not bind: binding itself changes a live controller's state.
        if not all(hasattr(controller, name) for name in ("qpos_idx", "qvel_idx", "ctrl_idx")):
            raise ValueError("goal capture requires already-bound controllers")
        qi, vi, ai = (np.array(getattr(controller, name), dtype=int, copy=True)
                      for name in ("qpos_idx", "qvel_idx", "ctrl_idx"))
        kp = np.array(getattr(controller, "kps", getattr(controller, "kp", 0.)), copy=True)
        kd = np.array(getattr(controller, "kds", getattr(controller, "kd", 0.)), copy=True)
        held.append((qi, vi, ai, data.qpos[qi].copy(), kp, kd))
    return held


def capture(model, data, robot_bodies, controllers=None, powered=True, seed=0, hz=HZ):
    """Simulate a throwaway 1.6 s continuation, returning age/xpos/xquat/origin/impacts.

    Impulses (N s, maximum norm 60) arrive at initial horizontal radius / 5,
    strictly inside 3 m. Their smoothstep falloff reaches zero at the boundary.
    Integration splits at impacts and sample times: hits are not frame-quantized.
    The final 1.6 s sample is always included (even for a non-integral frame count).
    The model is also copied: hiding the ball must not make an infinite ground
    plane push it back out or transfer its contacts to the robots.
    """
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be finite and positive")
    sim_model = copy.copy(model)
    sim = mujoco.MjData(sim_model)
    mujoco.mj_copyData(sim, sim_model, data)
    ball = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball < 0:
        raise ValueError("goal capture requires a free-jointed body named 'ball'")
    joint = int(sim_model.body_jntadr[ball])
    if joint < 0 or sim_model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("goal capture requires a free-jointed body named 'ball'")
    bq, bv = int(sim_model.jnt_qposadr[joint]), int(sim_model.jnt_dofadr[joint])
    # Suppress cloned ball collisions, including any welded child geometry.
    hidden = np.zeros(sim_model.nbody, dtype=bool)
    hidden[ball] = True
    for body in range(ball + 1, sim_model.nbody):
        hidden[body] = hidden[sim_model.body_parentid[body]]
    ball_geoms = hidden[sim_model.geom_bodyid]
    sim_model.geom_contype[ball_geoms] = 0
    sim_model.geom_conaffinity[ball_geoms] = 0
    sim.qfrc_applied[:] = 0
    sim.xfrc_applied[:] = 0
    sim.ctrl[:] = 0
    mujoco.mj_forward(sim_model, sim)
    origin = sim.xpos[ball].copy()
    held = _held_pd(sim, controllers) if powered else []
    bodies = sorted({int(sim_model.body(b).id) if isinstance(b, str) else int(b)
                     for b in robot_bodies})
    impacts = []
    for body in bodies:
        if not 0 < body < sim_model.nbody or hidden[body]:
            raise ValueError("robot_bodies must contain non-world, non-ball body IDs")
        delta = sim.xpos[body, :2] - origin[:2]
        radius = float(np.linalg.norm(delta))
        if radius >= _RADIUS:
            continue
        if radius > 1e-12:
            outward = delta / radius
        else:
            theta = np.random.default_rng(np.random.SeedSequence([seed, body])).uniform(0, 2 * np.pi)
            outward = np.array([np.cos(theta), np.sin(theta)])
        fraction = radius / _RADIUS
        magnitude = 60 * (1 - fraction)**2 * (1 + 2 * fraction)
        impulse = magnitude * np.r_[outward, .18] / np.sqrt(1 + .18**2)
        impacts.append(dict(body=body, distance=radius, age=radius / _SPEED,
                            impulse=impulse.tolist()))
    impacts.sort(key=lambda hit: (hit["age"], hit["body"]))

    def park_ball():
        sim.qpos[bq:bq + 3] = _PARK
        sim.qpos[bq + 3:bq + 7] = (1., 0., 0., 0.)
        sim.qvel[bv:bv + 6] = 0
        mujoco.mj_forward(sim_model, sim)

    park_ball()
    ages = np.r_[np.arange(0., DURATION_S, 1. / hz), DURATION_S]
    positions, quaternions = [], []
    dt = float(sim_model.opt.timestep)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("model timestep must be finite and positive")
    elapsed, hit_index = 0., 0
    for age in ages:
        while True:
            while hit_index < len(impacts) and impacts[hit_index]["age"] <= elapsed + 1e-12:
                hit = impacts[hit_index]
                impulse_dofs = np.zeros(sim_model.nv)
                # Strike chest height, not the feet: allow genuine tipping.
                point = sim.xpos[hit["body"]] + np.array([0., 0., .2])
                mujoco.mj_applyFT(sim_model, sim, np.asarray(hit["impulse"]),
                                 np.zeros(3), point, hit["body"], impulse_dofs)
                delta_v = np.zeros((1, sim_model.nv))
                mujoco.mj_solveM(sim_model, sim, delta_v, impulse_dofs[None, :])
                sim.qvel[:] += delta_v[0]
                hit_index += 1
            if elapsed >= age - 1e-12:
                break
            boundary = min(age, impacts[hit_index]["age"] if hit_index < len(impacts) else age)
            sim_model.opt.timestep = min(dt, boundary - elapsed)
            sim.ctrl[:] = 0
            for qi, vi, ai, target, kp, kd in held:
                sim.ctrl[ai] = kp * (target - sim.qpos[qi]) - kd * sim.qvel[vi]
            mujoco.mj_step(sim_model, sim)
            elapsed += sim_model.opt.timestep
            park_ball()
        effect_pos, effect_quat = effect_pose(float(age), origin, seed)
        positions.append(np.concatenate((sim.xpos, effect_pos)))
        quaternions.append(np.concatenate((sim.xquat, effect_quat)))
    return dict(age=ages, xpos=np.asarray(positions), xquat=np.asarray(quaternions),
                origin=origin, impacts=impacts)


def snapshot(model, data):
    """Small integration-state copy only: no physics, model clone or rendering.

    Retains mocap, warmstart and applied forces as well as qpos/qvel. Restored
    only after the sporting loop; never substitute the final match's state.
    """
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return {"state": state, "rgba": model.geom_rgba.copy()}


def restore(model, saved):
    """Restore a saved goal on an offline model, including corner-panel colours."""
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, saved["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    model.geom_rgba[:] = saved["rgba"]
    mujoco.mj_forward(model, data)
    return data


def controller_specs(controllers):
    """Copy only immutable PD bindings/gains before any club requests start."""
    from types import SimpleNamespace
    return [SimpleNamespace(**{name: np.array(getattr(c, name), copy=True)
            for name in ("qpos_idx", "qvel_idx", "ctrl_idx")},
            kps=np.array(getattr(c, "kps", getattr(c, "kp", 0.)), copy=True),
            kds=np.array(getattr(c, "kds", getattr(c, "kd", 0.)), copy=True))
            for c in controllers]
