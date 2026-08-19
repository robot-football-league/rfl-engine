"""Builds MJCF scenes: the untouched G1 robot model grafted into generated worlds.

The robot XML (g1_12dof.xml) is used verbatim — joints, inertials, actuators are
never edited. Single-robot scenes re-root it via ElementTree (V1 path, kept
byte-stable); multi-robot scenes attach it with MuJoCo's MjSpec API, which
namespaces every element (r0_pelvis, r1_left_knee_joint, ...) natively.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco
import numpy as np

from . import paths
from .util import quat_from_yaw

ARENA_HALF = 10.0  # 20 m x 20 m arena
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.1
SPAWN_HEIGHT = 0.793  # pelvis free-joint height from g1_12dof.xml

STATIC_RGBA = "0.45 0.55 0.75 1"
FORK_RGBA = "0.85 0.65 0.25 1"
MOVER_RGBA = "0.9 0.25 0.2 1"
WALL_RGBA = "0.55 0.55 0.6 1"

# Egocentric camera: head-height on the pelvis (the 12-DoF model has no
# separate head/torso body), looking along body +x, pitched 10 deg down —
# roughly where the real G1's RealSense sits. fovy ~ D435 vertical FOV.
EGOCAM_POS = (0.06, 0.0, 0.45)
EGOCAM_PITCH_DOWN_RAD = 0.175
EGOCAM_FOVY = 58.0


def _egocam_quat(pitch_down_rad: float = EGOCAM_PITCH_DOWN_RAD) -> list[float]:
    """Quat for a camera on the pelvis looking along +x, pitched down.

    MuJoCo cameras look along their frame's -z with +y as image-up; columns of
    R are (image-right, image-up, -view-dir) in the parent body frame.
    """
    s, c = np.sin(pitch_down_rad), np.cos(pitch_down_rad)
    R = np.array([[0.0, s, -c],
                  [-1.0, 0.0, 0.0],
                  [0.0, c, s]])
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q.tolist()


def _fmt(*vals) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def _texture_assets(asset: ET.Element) -> None:
    """Textured materials for course surfaces. Vision-navigation models are
    trained on real-world imagery; flat-shaded geoms carry almost no visual
    gradient for them (measured: near-constant ViNT waypoints). Checker
    textures give every surface trackable structure at negligible cost."""
    for name, rgb1, rgb2, repeat in (
        ("tex_wall", "0.52 0.52 0.56", "0.38 0.38 0.44", "12 3"),
        ("tex_block", "0.42 0.52 0.74", "0.28 0.38 0.62", "4 4"),
        ("tex_fork", "0.88 0.66 0.22", "0.68 0.48 0.12", "4 4"),
        ("tex_hazard", "0.92 0.18 0.12", "0.96 0.92 0.88", "6 6"),
    ):
        ET.SubElement(asset, "texture", {
            "type": "2d", "name": name, "builtin": "checker",
            "rgb1": rgb1, "rgb2": rgb2, "width": "256", "height": "256",
        })
        ET.SubElement(asset, "material", {
            "name": name.replace("tex_", "mat_"), "texture": name,
            "texrepeat": repeat, "reflectance": "0.05",
        })


def build_scene_xml(course=None, spawn=(0.0, 0.0, 0.0), flat_extent=None) -> str:
    """Return a complete MJCF string.

    course: a CourseSpec (duck-typed; see course.py) or None for open flat ground.
    spawn: (x, y, yaw) of the pelvis free joint.
    """
    robot = ET.parse(paths.G1_XML).getroot()

    root = ET.Element("mujoco", {"model": "g1_gauntlet"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": str(paths.MESH_DIR)})

    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.35 0.35 0.35", "specular": "0.2 0.2 0.2"})
    ET.SubElement(vis, "quality", {"shadowsize": "4096"})

    # Robot's global defaults (joint damping/armature/frictionloss) — copy as-is.
    robot_default = robot.find("default")
    if robot_default is not None:
        root.append(deepcopy(robot_default))

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.45 0.58 0.72", "rgb2": "0.88 0.92 0.96",
        "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
        "rgb1": "0.26 0.31 0.36", "rgb2": "0.19 0.23 0.28",
        "markrgb": "0.62 0.66 0.7", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.12",
    })
    _texture_assets(asset)
    robot_asset = robot.find("asset")
    if robot_asset is not None:
        for child in list(robot_asset):
            asset.append(deepcopy(child))

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {
        "pos": "0 0 14", "dir": "0 0 -1", "directional": "true",
        "diffuse": "0.75 0.75 0.75", "castshadow": "true",
    })

    extent = flat_extent if flat_extent is not None else ARENA_HALF
    ET.SubElement(wb, "geom", {
        "name": "floor", "type": "plane",
        "size": _fmt(extent, extent, 0.05), "material": "groundplane",
    })

    if course is not None:
        _add_course(wb, course)

    pelvis = robot.find("worldbody/body[@name='pelvis']")
    x, y, yaw = spawn
    pelvis = deepcopy(pelvis)
    pelvis.set("pos", _fmt(x, y, SPAWN_HEIGHT))
    pelvis.set("quat", _fmt(*quat_from_yaw(yaw)))
    wb.append(pelvis)

    root.append(deepcopy(robot.find("actuator")))
    return ET.tostring(root, encoding="unicode")


def _arena_xml(course=None, flat_extent=None) -> str:
    """Arena-only MJCF (no robot): floor, lighting, walls/obstacles/checkpoints."""
    root = ET.Element("mujoco", {"model": "g1_gauntlet_arena"})
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.35 0.35 0.35", "specular": "0.2 0.2 0.2"})
    ET.SubElement(vis, "quality", {"shadowsize": "4096"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.45 0.58 0.72", "rgb2": "0.88 0.92 0.96",
        "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
        "rgb1": "0.26 0.31 0.36", "rgb2": "0.19 0.23 0.28",
        "markrgb": "0.62 0.66 0.7", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.12",
    })
    _texture_assets(asset)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {
        "pos": "0 0 14", "dir": "0 0 -1", "directional": "true",
        "diffuse": "0.75 0.75 0.75", "castshadow": "true",
    })
    extent = flat_extent if flat_extent is not None else ARENA_HALF
    ET.SubElement(wb, "geom", {
        "name": "floor", "type": "plane",
        "size": _fmt(extent, extent, 0.05), "material": "groundplane",
    })
    if course is not None:
        _add_course(wb, course)
    return ET.tostring(root, encoding="unicode")


JERSEY_RGBA = ((0.15, 0.4, 0.95, 1.0), (0.95, 0.3, 0.15, 1.0))  # r0 blue, r1 red


def build_duel_model(course=None, spawns=((0.0, 1.2, 0.0), (0.0, -1.2, 0.0)),
                     flat_extent=None) -> mujoco.MjModel:
    """Compile an arena with two namespaced G1s (prefixes r0_, r1_).

    Uses MjSpec attachment, which prefixes every body/joint/geom/actuator of
    each robot instance; the source XML is untouched. Each robot gets a
    colored marker sphere above its head for replay legibility.
    """
    spec = mujoco.MjSpec.from_string(_arena_xml(course, flat_extent))
    for i, (x, y, yaw) in enumerate(spawns):
        child = mujoco.MjSpec.from_file(str(paths.G1_XML))
        frame = spec.worldbody.add_frame(
            pos=[x, y, 0.0], quat=list(quat_from_yaw(yaw)))
        frame.attach_body(child.body("pelvis"), f"r{i}_", "")
        marker = spec.body(f"r{i}_pelvis").add_geom()
        marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
        marker.size[0] = 0.07
        marker.pos = [0.0, 0.0, 0.62]
        marker.rgba = JERSEY_RGBA[i]
        marker.contype = 0
        marker.conaffinity = 0
        marker.group = 1
        cam = spec.body(f"r{i}_pelvis").add_camera()
        cam.name = f"r{i}_egocam"
        cam.pos = list(EGOCAM_POS)
        cam.quat = _egocam_quat()
        cam.fovy = EGOCAM_FOVY
    return spec.compile()


def _add_course(wb: ET.Element, course) -> None:
    half = course.arena_half
    # Four boundary walls.
    walls = [
        ("wall_n", (0, half + WALL_THICKNESS), (half + 2 * WALL_THICKNESS, WALL_THICKNESS)),
        ("wall_s", (0, -half - WALL_THICKNESS), (half + 2 * WALL_THICKNESS, WALL_THICKNESS)),
        ("wall_e", ((half + WALL_THICKNESS), 0), (WALL_THICKNESS, half)),
        ("wall_w", ((-half - WALL_THICKNESS), 0), (WALL_THICKNESS, half)),
    ]
    for name, (cx, cy), (sx, sy) in walls:
        ET.SubElement(wb, "geom", {
            "name": name, "type": "box",
            "size": _fmt(sx, sy, WALL_HEIGHT / 2),
            "pos": _fmt(cx, cy, WALL_HEIGHT / 2),
            "rgba": "1 1 1 1", "material": "mat_wall",
        })

    for ob in course.obstacles:
        if ob.kind == "moving":
            body = ET.SubElement(wb, "body", {
                "name": f"mover_{ob.id}", "mocap": "true",
                "pos": _fmt(ob.x, ob.y, ob.height / 2),
            })
            ET.SubElement(body, "geom", {
                "name": f"obs_{ob.id}", "type": "box",
                "size": _fmt(ob.sx / 2, ob.sy / 2, ob.height / 2),
                "rgba": "1 1 1 1", "material": "mat_hazard",
            })
        elif ob.kind == "sweeper":
            body = ET.SubElement(wb, "body", {
                "name": f"mover_{ob.id}", "mocap": "true",
                "pos": _fmt(ob.x, ob.y, ob.z),
            })
            ET.SubElement(body, "geom", {
                "name": f"obs_{ob.id}", "type": "box",
                "size": _fmt(ob.sx / 2, ob.sy / 2, ob.height / 2),
                "rgba": "1 1 1 1", "material": "mat_hazard",
            })
            # decorative hub on the boom (the post itself is a static obstacle)
            ET.SubElement(body, "geom", {
                "type": "cylinder", "size": _fmt(0.22, ob.height / 2 + 0.02),
                "rgba": "0.25 0.25 0.3 1", "contype": "0", "conaffinity": "0",
                "group": "1",
            })
        elif ob.kind == "pendulum":
            body = ET.SubElement(wb, "body", {
                "name": f"mover_{ob.id}", "mocap": "true",
                "pos": _fmt(ob.x, ob.y, ob.z - ob.radius),
            })
            ET.SubElement(body, "geom", {
                "name": f"obs_{ob.id}", "type": "box",
                "size": _fmt(ob.sx / 2, ob.sy / 2, ob.height / 2),
                "rgba": "1 1 1 1", "material": "mat_hazard",
            })
            ET.SubElement(body, "geom", {  # rod up to the pivot, visual only
                "type": "cylinder", "size": _fmt(0.03, ob.radius / 2),
                "pos": _fmt(0, 0, ob.radius / 2),
                "rgba": "0.35 0.35 0.4 1", "contype": "0", "conaffinity": "0",
                "group": "1",
            })
            # gantry: posts + crossbar spanning the swing, visual only
            ax = ob.axis or (0.0, 1.0)
            for sgn in (1, -1):
                px, py = ob.x + ax[0] * sgn * (ob.radius + 0.6), ob.y + ax[1] * sgn * (ob.radius + 0.6)
                ET.SubElement(wb, "geom", {
                    "type": "cylinder", "size": _fmt(0.06, ob.z / 2),
                    "pos": _fmt(px, py, ob.z / 2),
                    "rgba": "0.4 0.4 0.45 1", "contype": "0", "conaffinity": "0",
                    "group": "1",
                })
            bar_len = ob.radius + 0.6
            bar_yaw = float(np.arctan2(ax[1], ax[0]))
            ET.SubElement(wb, "geom", {
                "type": "box", "size": _fmt(bar_len, 0.05, 0.05),
                "pos": _fmt(ob.x, ob.y, ob.z),
                "quat": _fmt(*quat_from_yaw(bar_yaw)),
                "rgba": "0.4 0.4 0.45 1", "contype": "0", "conaffinity": "0",
                "group": "1",
            })
        elif ob.shape == "cylinder":
            ET.SubElement(wb, "geom", {
                "name": f"obs_{ob.id}", "type": "cylinder",
                "size": _fmt(ob.sx / 2, ob.height / 2),
                "pos": _fmt(ob.x, ob.y, ob.height / 2),
                "rgba": "1 1 1 1",
                "material": "mat_fork" if ob.role == "fork" else "mat_block",
            })
        else:
            attrs = {
                "name": f"obs_{ob.id}", "type": "box",
                "size": _fmt(ob.sx / 2, ob.sy / 2, ob.height / 2),
                "pos": _fmt(ob.x, ob.y, ob.height / 2),
                "rgba": "1 1 1 1",
                "material": "mat_fork" if ob.role == "fork" else "mat_block",
            }
            if getattr(ob, "yaw", 0.0):
                attrs["quat"] = _fmt(*quat_from_yaw(ob.yaw))
            ET.SubElement(wb, "geom", attrs)

    for i, (cx, cy) in enumerate(course.checkpoints):
        ET.SubElement(wb, "site", {
            "name": f"cp_{i}", "type": "cylinder",
            "size": _fmt(course.checkpoint_radius, 0.012),
            "pos": _fmt(cx, cy, 0.013),
            "rgba": "0.2 0.85 0.35 0.35",
        })
