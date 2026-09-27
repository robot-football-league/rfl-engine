"""Draw the detached goal celebration without changing the match's model/data.

Ordinary body-local geometry, exactly as the bundle exports it. No screen-space
confetti: the burst stays in the goal mouth from any camera angle.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .goal_explosion import DURATION_S, effect_geometry


def rotation(q):
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q, dtype=float))
    return mat.reshape(3, 3)


def draw_frame(renderer, model, data, capture, index):
    """Pose a renderer's *visual scene*, never the simulation behind it."""
    renderer.renderer.update_scene(data, camera=renderer.cam)
    scn = renderer.renderer.scene
    pos, quat = capture['xpos'][index], capture['xquat'][index]
    rotations = [rotation(q) for q in quat]
    for g in scn.geoms[:scn.ngeom]:
        if g.objtype != mujoco.mjtObj.mjOBJ_GEOM or g.objid < 0:
            continue
        gid = g.objid
        b = model.geom_bodyid[gid]
        g.pos[:] = pos[b] + rotations[b] @ model.geom_pos[gid]
        g.mat[:] = rotations[b] @ rotation(model.geom_quat[gid])
    for k, spec in enumerate(effect_geometry()):
        if scn.ngeom >= scn.maxgeom:
            raise ValueError('goal celebration exceeds renderer geometry budget')
        bid = model.nbody + k
        typ = (mujoco.mjtGeom.mjGEOM_BOX if spec['type'] == 'box'
               else mujoco.mjtGeom.mjGEOM_SPHERE)
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, typ, np.asarray(spec['size'], dtype=float),
                          pos[bid].astype(float), rotations[bid].ravel(),
                          np.asarray(spec['rgba'], dtype=np.float32))
        g.emission = 0.65
        scn.ngeom += 1
    return renderer.renderer.render()


def render_celebration(renderer, model, data, capture, goal_t):
    """Write a fixed-duration insert; the regular frame sampler never advances."""
    count = round(DURATION_S * renderer.fps)
    ages = capture['age']
    cam = renderer.cam
    saved = (cam.lookat.copy(), cam.distance, cam.azimuth, cam.elevation)
    origin = np.asarray(capture['origin'])
    cam.lookat[:] = origin + [-np.sign(origin[0]) * .7, 0., .2]
    cam.distance, cam.azimuth, cam.elevation = 7.5, 90., -38.
    try:
        for n in range(count):
            # Capture at the output rate when video is requested. Nearest sample
            # also permits lower-rate preview renders without moving any clock.
            index = int(np.argmin(np.abs(ages - n / renderer.fps)))
            frame = draw_frame(renderer, model, data, capture, index)
            if renderer.overlay_fn is not None:
                renderer.presentation_xpos = capture['xpos'][index]
                try:
                    frame = renderer.overlay_fn(frame, goal_t)
                finally:
                    renderer.presentation_xpos = None
            renderer.writer.add(frame)
    finally:
        cam.lookat[:] = saved[0]
        cam.distance, cam.azimuth, cam.elevation = saved[1:]
    return count / renderer.fps
