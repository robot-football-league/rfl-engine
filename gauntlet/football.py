"""V4: 2v2 G1 football.

A walled 14 x 9 m pitch, a knee-height ball (r=0.35 m) light enough to be
pushed by a walking G1, 2.6 m goal mouths with post markers and a shallow
netted pocket behind each goal line. Fixed-length matches; most goals wins;
after a goal everything (except fallen robots) resets to kickoff. Fallen
robots stay where they fell for the rest of the match — same no-get-up
realism as the gauntlet — and become pitch furniture.

Team A (blue markers) attacks +x, team B (red) attacks -x, swapping is the
batch protocol's job. Robot indices: 0,1 = team A; 2,3 = team B.
"""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .envelope import load_envelope
from .episode import (BLEND_S, DECISION_PERIOD_S, ROT_HOLD_S, _AsyncDecider,
                      validate_action)
from .g1_policy import G1PolicyController
from .render import EpisodeRenderer
from .scene import SPAWN_HEIGHT, _texture_assets, quat_from_yaw
from .util import CommandBlender, FallTracker, tilt_from_quat, yaw_from_quat

# pitch (meters, half-extents where noted)
PITCH_X = 7.0          # half-length: goal lines at x = +-7
PITCH_Y = 4.5          # half-width
WALL_H = 0.9
WALL_T = 0.1
GOAL_HALF_W = 1.6      # goal mouth: |y| < 1.6 (generous for 0.35 m-ball play)
GOAL_DEPTH = 0.7       # netted pocket behind the line
POST_R = 0.08
CROSSBAR_Z = 1.7       # clear of the robots (tallest geom 1.22 m): a bar at
                       # head height meant heads passing THROUGH it, and a
                       # solid bar down there would just cause pileups. Real
                       # goals sit ~1.3x player height; this matches that and
                       # is solid, so a lofted ball can come back off it.
                       # Scoring still ignores ball height (the ball stays low).

BALL_R = 0.35
BALL_MASS = 0.45
BALL_RGBA = "0.95 0.15 0.75 1"  # high-vis magenta: nothing else on the pitch
                                # is this color, and the old checker pattern
                                # aliased into the checkered grass at range

# football egocam pitches steeper than the race default: a striker must see
# its own feet — at 10 deg the ground closer than ~1.5 m is below the frame
# and the ball vanishes exactly at touch range (measured; cost a match)
FOOTBALL_CAM_PITCH_RAD = 0.38  # ~22 deg
# PANORAMIC wide-angle lens (real hardware: RoboCup humanoids run fisheyes).
# 2:1 sensor at fovy 81.8 deg => 120 deg HORIZONTAL coverage while keeping
# ~4 px/deg — the old 73 deg lens left the ball off-camera 39-59% of the
# match (measured by magenta-pixel counting), which was the single biggest
# reason players "never walked at the ball". Bottom edge sits 62.9 deg below
# horizon: ground blind spot 0.23 m, better than before.
FOOTBALL_CAM_W, FOOTBALL_CAM_H = 480, 240
# BROADCAST FRAME. Every graphic in `overlay` is laid out against BASE_W x
# BASE_H in absolute pixels; gauntlet.draw2d.Scaled multiplies them up to
# whatever TV_W x TV_H actually is, so the layout is authored once. Keep
# BASE 16:9 and TV 16:9 — 4DGSX size their XR screen off the video itself.
BASE_W, BASE_H = 854, 480
# the broadcast's own frame — the cards and this render are concatenated with
# `-c copy`, so they must be the SAME numbers, not two copies of them
from .broadcast import TV_CRF, TV_FPS, TV_H, TV_W  # noqa: E402
FOOTBALL_CAM_FOVY = 81.8
# the frame pair is a MOTION BURST, not a history: the older frame is taken
# PREFRAME_LEAD_S before the request. It used to be the previous request's
# frame (~2 s old) and models anchored on it, steering by a stale world —
# in-match steering accuracy collapsed to a coin flip (measured).
PREFRAME_LEAD_S = 0.35
# The detector runs FASTER than decisions, as on a real robot (30 Hz there).
# Two sightings per 2 s decision left the ball-velocity estimate lagging badly
# (true 0.96 m/s read as 0.3-0.9), so extra cheap low-res cycles feed the
# world model between decisions. Data only: teams still see the full-res
# frames at decision time.
PERCEPT_W, PERCEPT_H = 240, 120
PERCEPT_PERIOD_S = 0.4

MATCH_TIME_S = 90.0
KICKOFF_FREEZE_S = 0.5   # command blend-in after each reset
# The FULL TIME banner appears WITH the whistle, not before it. It does not
# need a lead to be readable: broadcast_audio muxes the match with
# `tpad=stop_mode=clone` (OUTRO_S = 8 s), so whatever is on the final frame
# is held for eight seconds over the long whistle and the sign-off. The
# first cut showed it from T-2.5, which put the graphic on screen a full
# two seconds before the whistle sounded and the players stopped.
FULL_TIME_BANNER_LEAD_S = 0.3   # = broadcast_audio's whistle onset (t_end-0.3)
HALF_BREAK_S = 12.0      # halftime pause (banner + robots reset to
                         # kickoff): long enough to read as a real
                         # interval — whistle, stillness, whistle
# sound-event tape: ball impulse events sampled at 25 Hz for the broadcast
# audio mix (gauntlet/broadcast_audio.py). Data capture only — no audio is
# produced during the match.
EVENT_POLL_S = 0.04
EVENT_DV_MPS = 0.55      # velocity change that counts as an audible impact
# football turns expire after 1 s (vs the 2 s race watchdog): visual servoing
# on a ball needs finer rotation quanta — 2 s holds meant ~90 deg per decision
# and players spun past the ball chronically (measured, match day 2)
FOOTBALL_ROT_HOLD_S = 1.0
KICKOFF_LEAD_S = 3.5       # teams may think during the last seconds of a
                           # stoppage so a decision is READY at the whistle:
                           # otherwise every restart begins with robots
                           # standing still for a full decision round-trip
DECIDE_ABANDON_S = 10.0    # a decision stuck in flight this long is a hung
                           # provider call, not a slow one (p99 is ~2.5 s):
                           # void it and free the decider, or one dead HTTP
                           # connection silences the robot for the match
MGR_ABANDON_S = 30.0       # managers think on a 10 s cadence; same idea
TELEMETRY_PERIOD_S = 1.0  # league-side ground-truth log (analysis, not players)

# SELF-RECOVERY. Real G1-Comp robots get up after a fall (RoboCup humanoid
# rules also let an incapable player be re-entered after a delay), but our
# 12-DoF walking checkpoint has welded arms and provably cannot right itself
# (0/9 in the get-up probe). So a fall costs the robot FALL_RECOVERY_S lying
# still, after which it is restored upright ON THE SPOT — the timed cost of
# the get-up motion a real competition robot performs with its arms.
FALL_RECOVERY_S = 8.0

# GOAL REPLAY: the match halts and the broadcast cuts to the scorer's own
# head camera for the seconds leading up to the goal.
# REFEREE: stuck ball. Walled pitches have no throw-ins, so a ball wedged
# against a wall with bodies around it can stay there for the rest of the
# match (observed: 21 s pin by all four robots). RoboCup's game-stuck rule
# handles this by having the referee move the ball to a free position; we do
# the same and call it a dropped ball.
# CORNER RAMS. Each 45-degree corner panel sits on a linear actuator (a
# pneumatic or electric shaft behind the panel — buildable off the shelf).
# While the ball rests against a panel its arming light counts down; at zero
# the shaft extends, sweeping the corner clear. Slow enough to be safe-ish,
# firm enough to shift a ball and unbalance a robot standing in the way.
CORNER_ARM_S = 4.5          # seconds in the corner before the ram fires
# Trigger is a corner PROXIMITY sensor (photoelectric beam / referee vision in
# real hardware), not a contact switch: a bouncy ball rarely rests against a
# panel, but a ball loitering in the corner — usually because robots are
# shoving it there — is exactly what we want cleared.
CORNER_ZONE_X = 1.5         # how far from the end wall the zone reaches
CORNER_ZONE_Y = 1.6         # ... and from the side wall
CORNER_SLOW_MPS = 0.5       # only counts while the ball is not travelling
CORNER_STROKE_M = 0.65      # how far the panel advances
CORNER_EXTEND_S = 0.9       # extend time (=> ~0.7 m/s panel speed)
CORNER_HOLD_S = 0.4
CORNER_RETRACT_S = 1.2      # retract gently
CORNER_COOLDOWN_S = 2.0
# The end walls straddle the goal line, so they offer only WALL_T of material
# outboard where the side walls offer 2*WALL_T. Extending them outward (the
# inner face never moves) gives a corner panel somewhere to bury its end.
END_WALL_EXT = 0.05

# Referee dropped-ball is DISABLED by default (league experiment: with corner
# rams + anti-entanglement bumpers, flat-wall balls should be freeable by
# play; re-enable with run_match(referee_drop=True) if pins return).
REFEREE_DROP_DEFAULT = False
BALL_STUCK_S = 8.0
BALL_STUCK_M = 0.6
# ...but only while play is ENGAGED with the ball. Counting from kickoff (or
# from a previous drop) fired the rule on a ball nobody had reached yet, and
# then re-fired it on the ball it had just placed at the centre.
STUCK_ENGAGE_M = 2.0
BALL_WALL_M = 0.75        # "against a wall" for the behaviour layer's benefit

REPLAY_S = 5.0
# play_goal_replay emits ONE output frame per buffered snapshot, so the
# buffer has to be sampled at the video frame rate or the replay plays at
# the wrong speed. This constant is only the fallback for a renderer that
# does not state its fps — run_match reads renderer.fps. It happened to
# equal TV_FPS until the broadcast moved to 50, and then every replay ran
# at 2x (5 s of match in 2.5 s of video) for m11-m14 before anyone saw it.
REPLAY_SAMPLE_HZ = 25.0
# VOLUMETRIC EXPORT: full-scene pose track (every body, 25 Hz) written to
# log_dir/states.npz + the compiled model to scene.mjb. Enough to re-pose the
# whole match offline (web/AR replays) without re-running physics. Off by
# default; ~5 MB per 90 s match when on.
STATE_RECORD_HZ = 25.0
BUBBLE_S = 3.5            # how long a speech bubble stays above a player

# managers: one LLM per team, each in its own dugout on the south touchline.
# Full game data arrives every MANAGER_POLL_S; SHOUTING is rationed — at most
# one <=240-char instruction per MANAGER_SHOUT_S (a touchline shout, not a
# control bus). An empty message is a choice to stay silent.
MANAGER_POLL_S = 10.0
MANAGER_SHOUT_S = 20.0
MANAGER_MSG_MAX = 240
# player shouts. There is no radio — robots get the same channel humans
# have: a voice. League rule: plain human-readable language only — every
# shout is logged and shown on the broadcast, so spectators always see the
# full picture. Nothing shouted on the pitch is hidden.
MESSAGE_MAX = 120
# shout discipline: at most one shout per player per cooldown, and no
# repeating yourself. Without this players talked on every single decision
# (228 messages in a 120 s match), which buries the spectator in captions.
PLAYER_SHOUT_COOLDOWN_S = 10.0
SKILL_NAMES = ("go_to_ball", "kick_toward", "walk_to", "turn_to", "hold")
TECH_AREAS = {  # team -> (x0, x1, y0, y1), dugouts flanking the halfway line
    0: (-4.6, -1.6, -6.3, -5.3),
    1: (1.6, 4.6, -6.3, -5.3),
}
MANAGER_SPAWNS = {0: (-3.1, -5.8, np.pi / 2), 1: (3.1, -5.8, np.pi / 2)}

TEAM_RGBA = ((0.15, 0.4, 0.95, 1.0), (0.95, 0.3, 0.15, 1.0))  # default kits
TEAM_COLOR_NAMES = ("blue", "red")
TEAM_NAMES = ("Team A", "Team B")
TEAM_CODES = ("BLU", "RED")
KICKOFFS = (  # (x, y, yaw) per robot index; A faces +x, B faces -x
    (-2.5, 1.2, 0.0), (-2.5, -1.2, 0.0),
    (2.5, 1.2, np.pi), (2.5, -1.2, np.pi),
)
N_ROBOTS = 4


def _stripe_texture(width: int = 512, height: int = 512) -> Path:
    """A mown-turf tile: two bands, light and dark, with a little mottle.

    Real pitches are striped because the mower lays the grass toward and
    away from you, so alternating bands catch the light differently. One
    tile holds exactly one there-and-back pass, so it repeats seamlessly.
    """
    from . import paths
    out = paths.ROOT / "runs" / "assets" / "pitch_stripes.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as _np
    from PIL import Image as _I

    rng = _np.random.default_rng(20260820)
    img = _np.zeros((height, width, 3), dtype=_np.float32)
    # a mown pitch has real contrast between passes — subtle bands just
    # read as dirty grass once the renderer minifies the texture
    light = _np.array([0.283, 0.545, 0.290])
    dark = _np.array([0.156, 0.345, 0.176])
    half = width // 2
    img[:, :half] = light
    img[:, half:] = dark
    # soften the seam: a mower leaves a blend, not a hard edge
    blend = 14
    for k in range(blend):
        f = k / blend
        img[:, half - blend + k] = light * (1 - f) + dark * f
        col = (width - blend + k) % width
        img[:, col] = dark * (1 - f) + light * f
    # fine mottle so the turf is not a flat colour under stadium light
    noise = rng.normal(0.0, 0.008, size=(height, width, 1))
    noise += rng.normal(0.0, 0.010, size=(height // 32 + 1, width // 32 + 1, 1)
                        ).repeat(32, 0).repeat(32, 1)[:height, :width]
    img = _np.clip(img + noise, 0.0, 1.0)
    _I.fromarray((img * 255).astype("uint8")).save(out)
    return out


def _pitch_xml(team_colors=TEAM_RGBA) -> str:
    root = ET.Element("mujoco", {"model": "g1_football_pitch"})
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6",
                                     "ambient": "0.35 0.35 0.35",
                                     "specular": "0.2 0.2 0.2"})
    ET.SubElement(vis, "quality", {"shadowsize": "4096"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.45 0.58 0.72", "rgb2": "0.88 0.92 0.96",
        "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "pitchgrass", "file": str(_stripe_texture()),
        "content_type": "image/png"})
    ET.SubElement(asset, "material", {
        # one texture tile spans a mower's there-and-back, so texrepeat is
        # the number of stripe PAIRS down the pitch
        # texuniform makes texrepeat world-scaled. One tile is a there-and
        # -back mowing pass (2 stripes), so 0.5 = a 2 m pass = 1 m stripes:
        # 14 across the 14 m pitch, 7 per half.
        "name": "pitchgrass", "texture": "pitchgrass",
        "texuniform": "true", "texrepeat": "0.5 0.5", "reflectance": "0.06"})
    _texture_assets(asset)

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 12", "dir": "0 0 -1",
                                "directional": "true",
                                "diffuse": "0.75 0.75 0.75",
                                "castshadow": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane",
                               "size": f"{PITCH_X + 3} {PITCH_Y + 3} 0.05",
                               "conaffinity": "3",
                               "material": "pitchgrass"})

    def wall(name, cx, cy, sx, sy, rgba=None):
        g = {"name": name, "type": "box", "size": f"{sx} {sy} {WALL_H / 2}",
             "pos": f"{cx} {cy} {WALL_H / 2}", "conaffinity": "3"}
        if rgba:
            g["rgba"] = rgba
        else:
            g["rgba"] = "1 1 1 1"
            g["material"] = "mat_wall"
        ET.SubElement(wb, "geom", g)

    # side walls (full length incl. goal pockets)
    wall("wall_n", 0, PITCH_Y + WALL_T, PITCH_X + GOAL_DEPTH + 2 * WALL_T, WALL_T)
    wall("wall_s", 0, -PITCH_Y - WALL_T, PITCH_X + GOAL_DEPTH + 2 * WALL_T, WALL_T)
    # end walls: solid outside the goal mouth, netted pocket behind the mouth
    seg = (PITCH_Y - GOAL_HALF_W) / 2
    # goal pockets are painted with the DEFENDING team's color so vision-only
    # players can tell the ends apart (A attacks +x = B's pocket)
    pocket_rgba = {"e": " ".join(f"{v:.3g}" for v in team_colors[1]),
                   "w": " ".join(f"{v:.3g}" for v in team_colors[0])}
    for sgn, tag in ((1, "e"), (-1, "w")):
        gx = sgn * PITCH_X
        # The end walls STRADDLE the goal line while the side walls sit
        # wholly outboard of PITCH_Y, so an end wall offers a corner panel
        # only WALL_T of material to bury its end in where the side wall
        # offers 2*WALL_T. A panel laid out symmetrically about the corner
        # buried fine in one and punched 82 mm out through the other.
        # Thicken these OUTWARD only: the inner face — the one the ball
        # hits — stays exactly where it was, at gx - sgn * WALL_T.
        wall(f"wall_{tag}_top", gx + sgn * END_WALL_EXT, GOAL_HALF_W + seg,
             WALL_T + END_WALL_EXT, seg)
        wall(f"wall_{tag}_bot", gx + sgn * END_WALL_EXT, -GOAL_HALF_W - seg,
             WALL_T + END_WALL_EXT, seg)
        # net back wall + pocket sides, team-colored
        pc = pocket_rgba[tag]
        wall(f"net_{tag}_back", sgn * (PITCH_X + GOAL_DEPTH), 0, WALL_T, GOAL_HALF_W, rgba=pc)
        wall(f"net_{tag}_top", sgn * (PITCH_X + GOAL_DEPTH / 2), GOAL_HALF_W, GOAL_DEPTH / 2, WALL_T, rgba=pc)
        wall(f"net_{tag}_bot", sgn * (PITCH_X + GOAL_DEPTH / 2), -GOAL_HALF_W, GOAL_DEPTH / 2, WALL_T, rgba=pc)
        # posts + visual crossbar
        for py in (GOAL_HALF_W, -GOAL_HALF_W):
            ET.SubElement(wb, "geom", {
                "type": "cylinder", "size": f"{POST_R} {CROSSBAR_Z / 2}",
                "pos": f"{gx} {py} {CROSSBAR_Z / 2}",
                "rgba": "0.95 0.95 0.95 1"})
        ET.SubElement(wb, "geom", {
            "type": "box", "size": f"{POST_R} {GOAL_HALF_W} {POST_R}",
            "pos": f"{gx} 0 {CROSSBAR_Z}", "rgba": "0.95 0.95 0.95 1",
            "conaffinity": "3"})       # solid, like the posts and walls
    # 45-degree corner bevels: a ball pushed into a corner deflects back into
    # play instead of deadlocking (standard walled-pitch design)
    # BEVEL WIDTH is a physics parameter, not a look: it is the surface a
    # cornered ball rebounds off. Widened 1.1 -> 1.7 on 2026-08-21 so the
    # ram's actuator can be drawn where it actually is. A linear actuator's
    # housing must be DEEPER than its stroke — it swallows the shaft at rest
    # — so a 0.65 m stroke needs ~0.8 m of housing behind the panel, while
    # the recess behind a 1.1 m bevel was 0.64 m and narrowing to a point.
    # Below bev = 1.55 nothing fits and the hardware has to be hidden.
    # Costs 2.7% of playable area. See config/NOTICES.md 2026-08-21.
    bev = 1.7
    # Panel half-length: long enough to seal both wall joints, short enough
    # that both buried ends stay inside the wall skin. Derived, not guessed
    # — the old hand-picked 0.85*bev punched 82 mm out through the end wall.
    panel_half = (np.sqrt(2) * (bev / 2 + WALL_T + 2 * END_WALL_EXT)
                  - WALL_T - 0.02)
    # Actuator, now that there is room for it. HOUSE_D is the housing centre
    # along the panel's outward normal; the shaft is welded to the panel and
    # must still be inside the housing when the panel is fully extended.
    HOUSE_HV = (CORNER_STROKE_M + 0.15) / 2
    HOUSE_HU, HOUSE_HZ = 0.12, 0.14
    HOUSE_D = WALL_T + 0.02 + HOUSE_HV
    SHAFT_TIP = HOUSE_D - HOUSE_HV + CORNER_STROKE_M + 0.11
    # The panel overhangs the 45-degree chord so its joints with the walls
    # seal, which buries each end INSIDE wall material — and a buried end
    # puts the panel's top face at exactly z = WALL_H, coplanar with the
    # wall's own top face. Coplanar faces z-fight, and a z-fight flickers
    # frame to frame on air. So the panel is drawn twice: a collision hull
    # at full height that nobody renders (group 3 — MuJoCo's renderer and
    # the 4DGSX exporter both stop at group 2), and a visible face 4 mm
    # shorter that loses the depth test cleanly. Nothing the ball can
    # touch has moved: the hull is the geometry that was always there.
    PANEL_DROP = 0.004
    k = 0
    for sx in (1, -1):
        for sy in (1, -1):
            cx = sx * (PITCH_X - bev / 2)
            cy = sy * (PITCH_Y - bev / 2)
            yaw = np.arctan2(-sy, -sx) + np.pi / 2
            body = ET.SubElement(wb, "body", {
                "name": f"corner_{k}", "mocap": "true",
                "pos": f"{cx} {cy} {WALL_H / 2}",
                "quat": " ".join(str(v) for v in quat_from_yaw(yaw))})
            ET.SubElement(body, "geom", {       # collision hull, unseen
                "name": f"corner_panel_{k}", "type": "box",
                "size": f"{panel_half} {WALL_T} {WALL_H / 2}",
                "conaffinity": "3", "group": "3",
                "rgba": "0.75 0.75 0.78 1"})
            ET.SubElement(body, "geom", {       # what the camera sees
                "name": f"corner_face_{k}", "type": "box",
                "size": f"{panel_half} {WALL_T} {WALL_H / 2 - PANEL_DROP / 2}",
                "pos": f"0 0 {-PANEL_DROP / 2}",
                "contype": "0", "conaffinity": "0",
                "rgba": "0.75 0.75 0.78 1"})
            # SHAFT: welded to the panel, so it travels with it. Drawn
            # from the panel's back face out to SHAFT_TIP, which is chosen
            # so the tip is STILL inside the housing at full extension —
            # the old rod was exactly one stroke long and pulled clean out
            # of its housing every time the ram fired, leaving a bar
            # hanging in mid-air.
            ET.SubElement(body, "geom", {
                "type": "cylinder",
                "size": f"0.05 {(SHAFT_TIP - WALL_T) / 2}",
                "pos": f"0 {(SHAFT_TIP + WALL_T) / 2} 0.0",
                "quat": "0.7071 0.7071 0 0",
                "rgba": "0.32 0.32 0.36 1", "contype": "0", "conaffinity": "0",
                "group": "1"})
            # HOUSING: fixed in the corner recess, deeper than the stroke.
            outx, outy = sx / np.sqrt(2), sy / np.sqrt(2)
            ET.SubElement(wb, "geom", {
                "name": f"corner_housing_{k}", "type": "box",
                "size": f"{HOUSE_HU} {HOUSE_HV} {HOUSE_HZ}",
                "pos": f"{cx + outx * HOUSE_D} {cy + outy * HOUSE_D} "
                       f"{WALL_H / 2}",
                "quat": " ".join(str(v) for v in quat_from_yaw(yaw)),
                "rgba": "0.22 0.22 0.26 1", "contype": "0",
                "conaffinity": "0"})
            # ...and a plinth, because the housing has to stand on
            # something. Invisible from pitch level behind a 0.9 m panel,
            # but a 4DGSX viewer can orbit into the recess and a box
            # hovering in mid-air is exactly what they would notice.
            ET.SubElement(wb, "geom", {
                "name": f"corner_plinth_{k}", "type": "box",
                "size": f"{HOUSE_HU * 0.8} {HOUSE_HV * 0.7} "
                        f"{(WALL_H / 2 - HOUSE_HZ) / 2}",
                "pos": f"{cx + outx * HOUSE_D} {cy + outy * HOUSE_D} "
                       f"{(WALL_H / 2 - HOUSE_HZ) / 2}",
                "quat": " ".join(str(v) for v in quat_from_yaw(yaw)),
                "rgba": "0.18 0.18 0.21 1", "contype": "0",
                "conaffinity": "0"})
            k += 1
    # PITCH MARKINGS — painted lines, scaled from a full-size pitch
    # (105x68 m -> 14x9 m). Purely cosmetic: every one of these is
    # collision-free and carries NO rules meaning. There is still no
    # offside, no penalty area offence and no keeper; the boxes are paint,
    # because a pitch without them does not read as football.
    PAINT = "0.9 0.95 0.9 0.8"
    LW = 0.04                                    # half-width of a line

    def line(x, y, hx, hy, rgba=PAINT):
        ET.SubElement(wb, "geom", {
            "type": "box", "size": f"{hx} {hy} 0.001",
            "pos": f"{x} {y} 0.011", "rgba": rgba,
            "contype": "0", "conaffinity": "0", "group": "1"})

    def arc(cx, cy, radius, segments=64, start=0.0, sweep=2 * np.pi):
        step = sweep / segments
        # sweep may be negative (arcs drawn clockwise); a geom size never is
        seg_len = radius * abs(step) / 2 + LW * 0.6
        for k in range(segments):
            a = start + (k + 0.5) * step
            ET.SubElement(wb, "geom", {
                "type": "box", "size": f"{seg_len} {LW} 0.001",
                "pos": f"{cx + radius * np.cos(a):.4f} "
                       f"{cy + radius * np.sin(a):.4f} 0.011",
                "quat": f"{np.cos((a + np.pi / 2) / 2):.5f} 0 0 "
                        f"{np.sin((a + np.pi / 2) / 2):.5f}",
                "rgba": PAINT, "contype": "0", "conaffinity": "0",
                "group": "1"})

    line(0, 0, LW, PITCH_Y)                       # halfway line
    arc(0, 0, 1.22)                               # centre circle (9.15 m)
    line(0, 0, 0.07, 0.07)                        # centre spot

    for sgn in (-1.0, 1.0):
        gx = sgn * PITCH_X
        for depth, half_w in ((2.20, 2.67),       # penalty area
                              (0.73, 1.21)):      # goal area
            xin = gx - sgn * depth
            line(xin, 0, LW, half_w)                          # front edge
            for sy in (-1.0, 1.0):                            # the sides
                line(gx - sgn * depth / 2, sy * half_w,
                     depth / 2, LW)
        spot_x = gx - sgn * 1.47
        line(spot_x, 0, 0.06, 0.06)               # penalty spot (11 m)
        # The D: the part of a centre-circle-radius arc, struck from the
        # penalty spot, that falls OUTSIDE the penalty area.
        box_x = gx - sgn * 2.20
        half = float(np.arccos(np.clip(abs(box_x - spot_x) / 1.22, -1.0, 1.0)))
        mid = np.pi if sgn > 0 else 0.0           # pointing away from goal
        arc(spot_x, 0.0, 1.22, segments=22,
            start=mid - half, sweep=2 * half)

    # NO CORNER ARCS. They mark where a corner kick is taken, and there is
    # no corner kick here — nor, after the 1.1 m bevel, any corner to take
    # it from. Struck at (±PITCH_X, ±PITCH_Y) they landed 0.6 m BEHIND the
    # ram panel, in the sealed dead triangle: invisible during play, then
    # popping into shot for a second and a half every time a ram fired.

    # dugout stripes (visual), tinted per team, flanking the halfway line
    for tm, area in TECH_AREAS.items():
        x0, x1, y0, y1 = area
        c = team_colors[tm]
        tint = f"{c[0]:.3g} {c[1]:.3g} {c[2]:.3g} 0.45"
        ET.SubElement(wb, "geom", {
            "type": "box",
            "size": f"{(x1 - x0) / 2} {(y1 - y0) / 2} 0.001",
            "pos": f"{(x0 + x1) / 2} {(y0 + y1) / 2} 0.011",
            "rgba": tint, "contype": "0", "conaffinity": "0", "group": "1"})

    # the ball
    ball = ET.SubElement(wb, "body", {"name": "ball", "pos": f"0 0 {BALL_R}"})
    ET.SubElement(ball, "freejoint", {"name": "ball_free"})
    ET.SubElement(ball, "geom", {
        "name": "ball_geom", "type": "sphere", "size": f"{BALL_R}",
        "mass": f"{BALL_MASS}", "rgba": BALL_RGBA, "conaffinity": "3",
        # rolling friction 0.02 = turf: a 1.3 m/s trundle dies in ~1.5 m, a
        # 3 m/s strike still carries ~13 m. At the old 0.00012 the ball
        # rolled the length of the pitch unassisted — fixture 2's melee
        # squirts became four "own goals" nobody could catch
        "condim": "6", "friction": "0.7 0.003 0.02",
        "solref": "0.02 0.6",  # a bit bouncy
    })
    return ET.tostring(root, encoding="unicode")


# HAIR: purely cosmetic. Fixed capsules welded to the pelvis with ZERO mass
# and no collision geometry, so they add no degrees of freedom, no inertia and
# no contacts — the dynamics are bit-identical to a bare robot (verified). The
# earlier jointed version swung nicely but perturbed the simulation, which is
# not acceptable for a league where teams must be comparable.
# Each style is a list of capsules in pelvis-local metres: (from, to, radius).
def _hair_capsules(style: str):
    S = []
    if style == "short":                      # bob, tucked around the crown
        for sgn in (1, -1):
            for k, (y0, y1) in enumerate(((0.030, 0.055), (0.055, 0.065))):
                S.append(((-0.02 - 0.03 * k, sgn * y0, 0.555),
                          (-0.06 - 0.03 * k, sgn * y1, 0.470), 0.036))
        S.append(((0.005, 0.0, 0.560), (-0.085, 0.0, 0.520), 0.045))
    elif style == "long":                     # falls past the shoulders
        for sgn in (1, -1):
            S.append(((-0.02, sgn * 0.045, 0.560), (-0.075, sgn * 0.070, 0.430), 0.042))
            S.append(((-0.075, sgn * 0.070, 0.430), (-0.080, sgn * 0.075, 0.300), 0.038))
            S.append(((-0.080, sgn * 0.075, 0.300), (-0.070, sgn * 0.070, 0.190), 0.032))
        S.append(((0.005, 0.0, 0.560), (-0.090, 0.0, 0.500), 0.048))
    elif style == "ponytail":                 # bundle sweeping out the back
        S.append(((0.005, 0.0, 0.560), (-0.085, 0.0, 0.525), 0.046))
        S.append(((-0.085, 0.0, 0.525), (-0.155, 0.0, 0.455), 0.040))
        S.append(((-0.155, 0.0, 0.455), (-0.205, 0.0, 0.350), 0.032))
        S.append(((-0.205, 0.0, 0.350), (-0.225, 0.0, 0.265), 0.024))
    elif style == "mohawk":                   # crest along the midline
        for k, (x0, x1) in enumerate(((0.020, -0.005), (-0.005, -0.035),
                                      (-0.035, -0.065), (-0.065, -0.095))):
            h = 0.615 + 0.02 * min(k, 1) - 0.015 * max(0, k - 1)
            S.append(((x0, 0.0, 0.545), (x1, 0.0, h), 0.028))
    return S


def _add_hair(spec, prefix: str, style: str, color):
    """Weld the hairstyle onto the pelvis: massless, collision-free, render-only."""
    caps = _hair_capsules(style)
    if not caps:
        return
    body = spec.body(f"{prefix}pelvis")
    for n, (a, b, rad) in enumerate(caps):
        g = body.add_geom()
        g.name = f"{prefix}hair_{n}"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size[0] = rad
        g.fromto = [*a, *b]
        g.rgba = list(color)
        g.mass = 0.0            # no inertia contribution at all
        g.contype = 0           # no collisions, either side
        g.conaffinity = 0
        g.group = 1


HAIR_STYLES = ("none", "short", "long", "ponytail", "mohawk")


# JERSEYS: like hair, purely cosmetic. The kit design (identity/kit_*.png,
# square) is shown on thin chest and back panels welded to the torso —
# zero mass, zero collision, pure render geometry. The torso meshes stay
# tinted in the kit color underneath, which is also the whole fallback
# when a club has no kit image.
# Jersey panels sit ON the torso shell, not in front of it. Measured off
# the G1 torso mesh in the pelvis frame (the 12-DOF model has no separate
# torso body, so the whole upper body is pelvis geometry):
#
#     chest front peaks at x=+0.080 (y=0, z=0.225), falling to +0.066 at
#     the corners of the patch below; back is flatter, -0.071 to -0.059.
#
# The first cut used x=+0.148/-0.165 — sized to the invisible anti-
# entanglement bumper capsule (r=0.2), not to the body — which left the
# panels floating 68 mm off the chest and 94 mm off the back. The panels
# are now small enough to sit on the flat top of the chest curve, and
# their outer face is a hair proud of its peak: a flat panel MUST clear
# the peak, because anywhere it sinks below the shell the body occludes
# it and the kit simply vanishes there. Worst residual gap ~14 mm at the
# far corners, typically far less.
#
# Re-measure if the G1 mesh is ever revendored: sample the torso mesh
# verts per (y,z) cell and take the outermost x.
JERSEY_HALF = (0.055, 0.060)        # half width (y), half height (z)
JERSEY_FACE = {"front": 0.0802, "back": -0.0711}   # outer face, pelvis frame
JERSEY_T = 0.003                    # half thickness


def _add_jersey(spec, prefix: str, mat_name: str):
    body = spec.body(f"{prefix}pelvis")
    for tag, z, quat in (
            ("front", 0.225, [0.5, 0.5, 0.5, 0.5]),
            ("back", 0.238, [0.5, 0.5, -0.5, -0.5])):
        face = JERSEY_FACE[tag]
        # centre the slab so its OUTER face lands on `face`
        x = face - JERSEY_T if tag == "front" else face + JERSEY_T
        g = body.add_geom()
        g.name = f"{prefix}jersey_{tag}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [*JERSEY_HALF, JERSEY_T]
        g.pos = [x, 0.0, z]
        g.quat = quat                    # +Z faces outward, image upright
        g.material = mat_name
        g.rgba = [1.0, 1.0, 1.0, 1.0]
        g.mass = 0.0
        g.contype = 0
        g.conaffinity = 0
        g.group = 1


def _vivid(col):
    """Team color as TEXT on a dark panel: lighten dark kits so they read."""
    r, g, b = col[:3]
    if 0.299 * r + 0.587 * g + 0.114 * b < 96:
        r, g, b = (int(v + (255 - v) * 0.55) for v in (r, g, b))
    return (r, g, b, 255)


def build_football_model(manager_teams: tuple = (), cameras: bool = False,
                         team_colors=TEAM_RGBA, hair=None,
                         kit_textures=None) -> mujoco.MjModel:
    from . import paths
    from .scene import EGOCAM_POS, _egocam_quat
    spec = mujoco.MjSpec.from_string(_pitch_xml(team_colors))
    kit_mats = {}
    for tm, png in (kit_textures or {}).items():
        if not png:
            continue
        tex = spec.add_texture()
        tex.name = f"kit_tex{tm}"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = str(png)
        tex.content_type = "image/png"
        mat = spec.add_material()
        mat.name = f"kit_mat{tm}"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        kit_mats[tm] = mat.name
    spots = list(KICKOFFS) + [MANAGER_SPAWNS[tm] for tm in sorted(manager_teams)]
    mgr_team_of = {N_ROBOTS + k: tm for k, tm in enumerate(sorted(manager_teams))}
    for i, (x, y, yaw) in enumerate(spots):
        child = mujoco.MjSpec.from_file(str(paths.G1_XML))
        frame = spec.worldbody.add_frame(pos=[x, y, 0.0],
                                         quat=list(quat_from_yaw(yaw)))
        frame.attach_body(child.body("pelvis"), f"r{i}_", "")
        is_mgr = i >= N_ROBOTS
        team = mgr_team_of[i] if is_mgr else i // 2
        # ANTI-ENTANGLEMENT (the sim version of RoboCup's mandated
        # entanglement-safe arm design): arm collision geoms stop colliding
        # with other ROBOTS (still hit ball/walls/floor via channel 2), and a
        # smooth invisible torso capsule carries robot-robot contact — facing
        # robots shove and slide apart instead of hooking forearms.
        for g in spec.body(f"r{i}_pelvis").geoms:
            if g.contype and abs(g.pos[1]) > 0.10:
                g.contype = 2
                g.conaffinity = 0
        bumper = spec.body(f"r{i}_pelvis").add_geom()
        bumper.name = f"r{i}_bumper"
        bumper.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        bumper.size[0] = 0.20
        bumper.fromto = [0.02, 0.0, 0.10, 0.02, 0.0, 0.40]
        bumper.rgba = [0, 0, 0, 0]
        bumper.group = 4
        bumper.mass = 0.0
        bumper.contype = 1
        bumper.conaffinity = 1
        # SHIRT: recolor the welded torso meshes (all pelvis-body geoms) in
        # the team color; the second player wears a darker shade of the kit
        c = team_colors[team]
        shade = 1.0 if (is_mgr or i % 2 == 0) else 0.55
        for g in spec.body(f"r{i}_pelvis").geoms:
            g.rgba = [c[0] * shade, c[1] * shade, c[2] * shade, 1.0]
        if is_mgr:  # the gaffer's cap: small team-colored dot stays
            marker = spec.body(f"r{i}_pelvis").add_geom()
            marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
            marker.size[0] = 0.1
            marker.pos = [0.0, 0.0, 0.62]
            marker.rgba = c
            marker.contype = 0
            marker.conaffinity = 0
            marker.group = 1
        if not is_mgr and team in kit_mats:
            _add_jersey(spec, f"r{i}_", kit_mats[team])
        if hair and not is_mgr:
            h = hair.get(team) or {}
            if isinstance(h, list):   # per-player looks: [p0_look, p1_look]
                h = h[i % 2] if (i % 2) < len(h) else {}
            style = (h or {}).get("style", "none")
            hcol = (h or {}).get("color", [0.93, 0.86, 0.55, 1.0])
            if len(hcol) == 3:
                hcol = [*hcol, 1.0]
            _add_hair(spec, f"r{i}_", style, hcol)
        if cameras and not is_mgr:
            cam = spec.body(f"r{i}_pelvis").add_camera()
            cam.name = f"r{i}_egocam"
            cam.pos = list(EGOCAM_POS)
            cam.quat = _egocam_quat(FOOTBALL_CAM_PITCH_RAD)
            cam.fovy = FOOTBALL_CAM_FOVY
    return spec.compile()


@dataclass
class MatchRobotResult:
    agent: str
    team: str
    fell: bool = False
    fall_time_s: float | None = None
    falls: int = 0
    recoveries: int = 0
    touches: int = 0
    decisions: int = 0
    invalid_actions: int = 0
    missed_deadlines: int = 0
    abandoned: int = 0
    mean_decision_latency_s: float | None = None


@dataclass
class MatchResult:
    mode: str
    match_time_s: float
    teams: dict = field(default_factory=dict)  # {A/B: {name, code, players}}
    halves: int = 1
    half_breaks: list[float] = field(default_factory=list)  # halftime whistle times
    score: list[int] = field(default_factory=lambda: [0, 0])
    winner: str = "draw"  # "A" | "B" | "draw"
    goals: list[dict] = field(default_factory=list)  # {t, team, scorer}
    events: list[dict] = field(default_factory=list)  # sound tape: {t, kind, mag}
    dropped_balls: list[float] = field(default_factory=list)  # referee restarts
    robots: list[MatchRobotResult] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost_usd: float | None = None  # None when no priced model played
    wall_time_s: float = 0.0
    honest_latency: bool = False  # replies charged their wall latency in sim time

    def to_dict(self):
        return asdict(self)


def _match_observation(ctrls, data, ball_qpos_adr, ball_qvel_adr, i, t,
                       score, fallen, last_action_result, match_time_s,
                       decision_interval_s, blocked):
    team = i // 2
    me = ctrls[i]
    mp, mq, mv = me.base_pos(data), me.base_quat(data), me.base_linvel(data)
    bp = data.qpos[ball_qpos_adr:ball_qpos_adr + 3]
    bv = data.qvel[ball_qvel_adr:ball_qvel_adr + 3]

    def other(j):
        c = ctrls[j]
        p, v = c.base_pos(data), c.base_linvel(data)
        return {"position": [round(float(p[0]), 2), round(float(p[1]), 2)],
                "heading_rad": round(yaw_from_quat(c.base_quat(data)), 2),
                "velocity": [round(float(v[0]), 2), round(float(v[1]), 2)],
                "fallen": bool(fallen[j])}

    mates = [j for j in range(N_ROBOTS) if j // 2 == team and j != i]
    opps = [j for j in range(N_ROBOTS) if j // 2 != team]
    attack_sign = 1 if team == 0 else -1
    return {
        "time_remaining_s": round(match_time_s - t, 1),
        "decision_interval_s": round(decision_interval_s, 2),
        "score": {"you": score[team], "them": score[1 - team]},
        "attacking_goal": {"x": attack_sign * PITCH_X, "y_range": [-GOAL_HALF_W, GOAL_HALF_W]},
        "defending_goal": {"x": -attack_sign * PITCH_X, "y_range": [-GOAL_HALF_W, GOAL_HALF_W]},
        "self": {"position": [round(float(mp[0]), 2), round(float(mp[1]), 2)],
                 "heading_rad": round(yaw_from_quat(mq), 2),
                 "velocity": [round(float(mv[0]), 2), round(float(mv[1]), 2)],
                 "fallen": bool(fallen[i]), "blocked": bool(blocked)},
        "ball": {"position": [round(float(bp[0]), 2), round(float(bp[1]), 2)],
                 "velocity": [round(float(bv[0]), 2), round(float(bv[1]), 2)],
                 "radius": BALL_R},
        "teammates": [other(j) for j in mates],
        "opponents": [other(j) for j in opps],
        "last_action_result": last_action_result[i],
    }


def run_match(agents, match_time_s: float = MATCH_TIME_S,
              mode: str = "paused", realtime_factor: float = 1.0,
              decision_deadline_s: float | None = None,
              request_period_s: float | None = None,
              honest_latency: bool = False,
              managers: dict | None = None,  # {team_idx: agent}
              manager_period_s: float = MANAGER_POLL_S,
              obs_mode: str = "full",  # full | camera (camera+shouts only)
              team_colors=TEAM_RGBA, team_color_names=TEAM_COLOR_NAMES,
              team_names=TEAM_NAMES, team_codes=TEAM_CODES, hair=None,
              referee_drop: bool = REFEREE_DROP_DEFAULT,
              player_names=None,   # {team: [name0, name1]}
              halves: int = 1,     # 2 = two halves of match_time_s/2 each
              record_states: bool | None = None,  # None = RFL_EXPORT_STATES env
              kit_textures=None,   # {team: path-to-kit-png} for jersey panels
              badges=None,         # {team: path-to-badge-png} for the scorebug
              video_path=None, log_dir=None) -> MatchResult:
    assert len(agents) == N_ROBOTS and mode in ("paused", "realtime")
    managers = managers or {}
    t_wall = time.time()
    n_bodies = N_ROBOTS + len(managers)
    model = build_football_model(manager_teams=tuple(managers),
                                 cameras=(obs_mode in ("camera", "sdk")),
                                 team_colors=team_colors, hair=hair,
                                 kit_textures=kit_textures)
    ctrls = [G1PolicyController(prefix=f"r{i}_") for i in range(n_bodies)]
    model.opt.timestep = ctrls[0].simulation_dt
    data = mujoco.MjData(model)
    for c in ctrls:
        c.bind(model)
        c._d_cache = data
    dt = ctrls[0].simulation_dt
    decision_every = int(round(DECISION_PERIOD_S / dt))
    max_steps = int(round(match_time_s / dt))
    request_period = max(DECISION_PERIOD_S, request_period_s or 0.0)

    ball_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos_adr = model.jnt_qposadr[ball_jnt]
    ball_qvel_adr = model.jnt_dofadr[ball_jnt]
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    # corner rams: mocap panels driven along their inward normal
    corners = []
    for k in range(4):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"corner_{k}")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"corner_panel_{k}")
        vgid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"corner_face_{k}")
        mid = model.body_mocapid[bid]
        rest = np.array(model.body_pos[bid], dtype=float)
        inward = np.array([-np.sign(rest[0]), -np.sign(rest[1]), 0.0])
        inward /= (np.linalg.norm(inward) or 1.0)
        corners.append({"gid": gid, "vgid": vgid,  # hull collides, face shows
                        "mid": mid, "rest": rest, "inward": inward,
                        "charge": 0.0, "phase": None, "phase_t": 0.0,
                        "last_touch_t": -1e9})
    robot_geoms = [set() for _ in range(N_ROBOTS)]
    for g in range(model.ngeom):
        bname = model.body(model.geom_bodyid[g]).name
        for i in range(N_ROBOTS):
            if bname.startswith(f"r{i}_"):
                robot_geoms[i].add(g)
    # sound tape: wall/net geoms (impact classification for the audio mix)
    wall_gids = {g for g in range(model.ngeom)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     ).startswith(("wall_", "net_"))}
    post_xy = [(sx * PITCH_X, sy * GOAL_HALF_W)
               for sx in (1, -1) for sy in (1, -1)]
    # geom -> owning robot (O(1) lookup for robot-robot contact attribution)
    geom_owner = np.full(model.ngeom, -1, dtype=int)
    for i in range(N_ROBOTS):
        for g in robot_geoms[i]:
            geom_owner[g] = i

    envelope = load_envelope()
    ego_r = None
    if obs_mode in ("camera", "sdk"):
        ego_r = mujoco.Renderer(model, height=FOOTBALL_CAM_H,
                                width=FOOTBALL_CAM_W)
    # rfl-0.3 on-robot SDK: perception + world model + skill execution
    cams, wms, skills = None, None, None
    percept_r, percept_cams, percept_min_px = None, None, 25
    next_percept_t = [0.0]
    if obs_mode == "sdk":
        from .rfl_sdk import Camera, SkillRunner, WorldModel, perceive
        cams = [Camera(model, f"r{i}_egocam", FOOTBALL_CAM_W, FOOTBALL_CAM_H,
                       FOOTBALL_CAM_FOVY) for i in range(N_ROBOTS)]
        percept_r = mujoco.Renderer(model, height=PERCEPT_H, width=PERCEPT_W)
        percept_cams = [Camera(model, f"r{i}_egocam", PERCEPT_W, PERCEPT_H,
                               FOOTBALL_CAM_FOVY) for i in range(N_ROBOTS)]
        percept_min_px = max(6, int(25 * (PERCEPT_W / FOOTBALL_CAM_W) ** 2))
        wms = [WorldModel() for _ in range(N_ROBOTS)]
        skills = [SkillRunner(PITCH_X, PITCH_Y, BALL_R,
                              envelope_vx=(envelope["vx"][0], envelope["vx"][1]),
                              max_wz=envelope["wz"][1],
                              goal_half_w=GOAL_HALF_W) for _ in range(N_ROBOTS)]
    team_of = [j // 2 for j in range(N_ROBOTS)]
    player_names = player_names or {}
    def pname(j):
        lst = player_names.get(team_of[j]) or []
        k = j % 2
        return lst[k] if k < len(lst) and lst[k] else f"#{k + 1}"

    def attack_goal_xy(i):
        return (PITCH_X if team_of[i] == 0 else -PITCH_X, 0.0)

    def player_xy_all():
        return [tuple(float(v) for v in ctrls[j].base_pos(data)[:2])
                for j in range(N_ROBOTS)]

    def run_perception(i, frame, t_now):
        """One detector cycle folded into robot i's world model."""
        c = ctrls[i]
        p = c.base_pos(data)
        last_det[i] = perceive(
            model, data, cams[i], frame, (float(p[0]), float(p[1])),
            yaw_from_quat(c.base_quat(data)), i, player_xy_all(),
            team_of, BALL_R, t_now, wms[i],
            pitch=(PITCH_X, PITCH_Y), wall_m=BALL_WALL_M)
    prev_frame = [None] * N_ROBOTS
    prev_frame_t = [0.0] * N_ROBOTS
    pre_target = [None] * N_ROBOTS  # which upcoming request each pre-frame is for

    # kickoff state snapshots for goal resets
    mujoco.mj_forward(model, data)
    home_qpos = data.qpos.copy()

    blenders = [CommandBlender(BLEND_S) for _ in range(n_bodies)]
    falls = [FallTracker() for _ in range(n_bodies)]
    fallen_flags = [False] * n_bodies
    result = MatchResult(mode=mode, match_time_s=match_time_s, halves=halves,
                         honest_latency=bool(honest_latency and mode == "realtime"),
                         teams={k: {"name": team_names[tm],
                                    "code": team_codes[tm],
                                    "color": team_color_names[tm],
                                    "players": [pname(tm * 2), pname(tm * 2 + 1)]}
                                for tm, k in ((0, "A"), (1, "B"))},
                         robots=[
        MatchRobotResult(agent=getattr(a, "name", str(a)),
                         team="A" if i // 2 == 0 else "B")
        for i, a in enumerate(agents)])

    log_dir = Path(log_dir) if log_dir else None
    if record_states is None:
        record_states = bool(os.environ.get("RFL_EXPORT_STATES"))
    state_rec = None
    if record_states and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        state_rec = {"t": [], "xpos": [], "xquat": [], "next_t": 0.0}
        # everything the exporter needs to rebuild THIS model exactly;
        # a full scene.mjb is ~380 MB/match (embedded meshes), so it is
        # opt-in for debugging only
        (log_dir / "scene_build.json").write_text(json.dumps({
            "manager_teams": sorted(managers),
            "cameras": obs_mode in ("camera", "sdk"),
            "team_colors": [list(c) for c in team_colors],
            "hair": hair,
            "kit_textures": kit_textures or {}}))
        if os.environ.get("RFL_EXPORT_MJB"):
            mujoco.mj_saveModel(model, str(log_dir / "scene.mjb"), None)
    decisions_f = None
    telemetry_f = None
    comms_f = None
    frame_dir = None
    next_telemetry_t = 0.0
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        decisions_f = open(log_dir / "decisions.jsonl", "w")
        # ground-truth positions for league analysis — players never see this
        telemetry_f = open(log_dir / "telemetry.jsonl", "w")
        comms_f = open(log_dir / "comms.jsonl", "w")
        if obs_mode in ("camera", "sdk"):
            frame_dir = log_dir / "frames"
            frame_dir.mkdir(exist_ok=True)
    for i, agent in enumerate(agents):
        if hasattr(agent, "begin_episode"):
            if log_dir:
                (log_dir / f"r{i}").mkdir(parents=True, exist_ok=True)
            agent.begin_episode(log_dir=(log_dir / f"r{i}") if log_dir else None)
    for tm, mgr in managers.items():
        if hasattr(mgr, "begin_episode"):
            if log_dir:
                (log_dir / f"mgr{tm}").mkdir(parents=True, exist_ok=True)
            mgr.begin_episode(log_dir=(log_dir / f"mgr{tm}") if log_dir else None)

    renderer = None
    if video_path:
        renderer = EpisodeRenderer(model, video_path, track_body=None,
                                   width=TV_W, height=TV_H, fps=TV_FPS,
                                   crf=TV_CRF,
                                   distance=9.0, azimuth=90.0, elevation=-38.0)

        # BROADCAST CAMERA. A real match camera sits in one place on the
        # gantry and works by panning and zooming, so that is what this
        # does: the position never moves, the aim follows the action, and
        # the field of view opens just enough to hold every player. All of
        # it is heavily smoothed — a camera that snaps looks like a bug,
        # and one that lags looks like a camera operator.
        # the position the original fixed shot was taken from — keep it, so
        # the new camera starts from a framing we already know reads well
        CAM_HOME = np.array([0.0, -10.64 - (1.0 if managers else 0.0), 8.71])
        cam_state = {"aim": np.array([0.0, 0.0, 0.4]), "fov": 45.0}
        MIN_FOV, MAX_FOV = 38.0, 52.0
        ASPECT = BASE_W / BASE_H

        def aim_camera(t):
            pts = [np.array(ctrls[i].base_pos(data)[:3]) for i in range(N_ROBOTS)
                   if not fallen_flags[i]] or [
                   np.array(ctrls[i].base_pos(data)[:3]) for i in range(N_ROBOTS)]
            ball = np.array([float(data.qpos[ball_qpos_adr]),
                             float(data.qpos[ball_qpos_adr + 1]), 0.2])
            # the ball is the story: weight it like two outfield players
            target = np.vstack(pts + [ball, ball]).mean(axis=0)
            target[2] = 0.45
            # bias toward the middle so the camera never swings to an
            # extreme angle for one stray robot
            target[0] *= 0.45

            # ease toward the target: fast enough to keep up with a break,
            # slow enough that a scrappy midfield does not jitter the shot
            cam_state["aim"] += (target - cam_state["aim"]) * 0.06
            aim = cam_state["aim"]

            v = aim - CAM_HOME
            dist = float(np.linalg.norm(v))
            fwd = v / (dist + 1e-9)
            # camera basis: right is horizontal, up completes the frame
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
            right /= (np.linalg.norm(right) + 1e-9)
            up = np.cross(right, fwd)

            # Size the VERTICAL fov. Horizontal spread is divided by the
            # aspect ratio — the frame is far wider than it is tall, so
            # sizing vertically off horizontal spread zooms way out.
            need = 0.0
            for q in np.vstack(pts + [ball]):
                off = q - CAM_HOME
                fz = float(np.dot(off, fwd)) or 1e-9
                a_v = abs(float(np.dot(off, up))) / fz
                a_h = abs(float(np.dot(off, right))) / fz / ASPECT
                need = max(need, a_v, a_h)
            # 1.45 keeps a comfortable border: nobody should be clipped to
            # the edge of frame, which is what "all players in shot" means
            want = float(np.clip(np.degrees(np.arctan(need)) * 2.0 * 1.45,
                                 MIN_FOV, MAX_FOV))
            cam_state["fov"] += (want - cam_state["fov"]) * 0.05
            model.vis.global_.fovy = cam_state["fov"]

            # aim from the fixed gantry point: derive the orbital params so
            # the camera POSITION stays exactly at CAM_HOME
            renderer.cam.lookat[:] = aim
            renderer.cam.distance = dist
            renderer.cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
            # MuJoCo places the eye at lookat - distance * forward, so the
            # forward vector IS v/dist: elevation is asin(v.z/dist) with no
            # negation. Flipping the sign here mirrors the camera through
            # the pitch and films the match from underground.
            renderer.cam.elevation = float(np.degrees(
                np.arcsin(np.clip(v[2] / (dist + 1e-9), -1.0, 1.0))))

        def project(p):
            """World point -> broadcast pixel via the render camera."""
            try:
                glcam = renderer.renderer.scene.camera[0]
            except AttributeError:
                return None
            pos = np.array(glcam.pos)
            fwd = np.array(glcam.forward)
            up = np.array(glcam.up)
            right = np.cross(fwd, up)
            rel = np.asarray(p) - pos
            z = float(rel @ fwd)
            if z < 0.1:
                return None
            xn = float(rel @ right) / z * glcam.frustum_near
            yn = float(rel @ up) / z * glcam.frustum_near
            half_h = (glcam.frustum_top - glcam.frustum_bottom) / 2
            half_w = half_h * (BASE_W / BASE_H)
            cx = BASE_W / 2 * (1 + xn / half_w)
            cy = BASE_H / 2 * (1 - yn / half_h)
            return cx, cy      # BASE-space pixels; Scaled maps them up

        badge_img = {}
        for tm, bp in (badges or {}).items():
            try:                      # a club without a crest keeps its chip
                from PIL import Image as _I
                badge_img[tm] = _I.open(bp).convert("RGBA")
            except Exception:
                pass

        def overlay(frame, tt):
            from PIL import Image, ImageDraw

            from .draw2d import Scaled
            img = Image.fromarray(frame)
            d = Scaled(ImageDraw.Draw(img, "RGBA"), frame.shape[0] / BASE_H)
            font, font_big, font_sm = d.font(15), d.font(22), d.font(12)
            # lay out in BASE space whatever the real frame is; `d` scales
            h, w = BASE_H, BASE_W

            def ink_on(rgb):
                """Dark or light text so it always reads on a team color."""
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                return ((18, 18, 24, 255) if lum > 150
                        else (255, 255, 255, 255))

            # floating player name plates (broadcast-side, not in-world)
            for j in range(N_ROBOTS):
                pt = project(ctrls[j].base_pos(data) + np.array([0, 0, 0.75]))
                if pt is None:
                    continue
                cx, cy = pt
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                tc = tuple(int(v * 255) for v in team_colors[j // 2][:3]) + (235,)
                ink = ink_on(tc)
                edge = ((30, 30, 40, 220) if ink[0] < 128
                        else (255, 255, 255, 220))
                label = f"{j % 2 + 1} {pname(j)}"
                tw = d.textlength(label, font=font) + 12
                d.rounded_rectangle([cx - tw / 2, cy - 10, cx + tw / 2, cy + 10],
                                    radius=9, fill=tc, outline=edge)
                d.text((cx - tw / 2 + 6, cy - 7), label, fill=ink, font=font)

            c0 = tuple(int(v * 255) for v in team_colors[0][:3]) + (255,)
            c1 = tuple(int(v * 255) for v in team_colors[1][:3]) + (255,)

            # LIVE tag, top right
            d.rounded_rectangle([w - 78, 8, w - 12, 34], radius=7,
                                fill=(12, 12, 18, 215))
            d.ellipse([w - 69, 16, w - 59, 26], fill=(235, 55, 45, 255))
            d.text((w - 52, 21), "LIVE", fill=(255, 255, 255, 255),
                   font=font, anchor="lm")

            # match clock state (counts down within the current half)
            if halves == 2:
                half_len = match_time_s / 2
                in_break = (half_banner[0] > -1e8
                            and 0 <= tt - half_banner[0] < HALF_BREAK_S)
                in_h2 = tt >= half_len
                remaining = max(0.0, (match_time_s if in_h2 else half_len) - tt)
                tag = ("Half Time" if in_break else
                       "Second Half" if in_h2 else "First Half")
            else:
                remaining = max(0.0, match_time_s - tt)
                tag = ""
            mm, ss = int(remaining) // 60, int(remaining) % 60
            clock_str = (f"{tag}  " if tag else "") + f"{mm:02d}:{ss:02d}"

            # classic compact scorebug, top left (kept alongside the bottom
            # board): [chip] RMA | 3 - 2 | SGU [chip]   1H 04:31
            bug_r = 384                     # right edge of the compact bug
            d.rounded_rectangle([10, 8, bug_r, 38], radius=7,
                                fill=(12, 12, 18, 215))

            def txt(xp, s, fill, anchor="lm"):
                d.text((xp, 23), s, fill=fill, font=font, anchor=anchor)
                return d.textlength(s, font=font)

            x = 20
            d.rectangle([x, 15, x + 14, 31], fill=c0)
            x += 20
            x += txt(x, team_codes[0][:3], (255, 255, 255, 255)) + 10
            x += txt(x, f"{score[0]} - {score[1]}", (255, 235, 160, 255)) + 10
            x += txt(x, team_codes[1][:3], (255, 255, 255, 255)) + 6
            d.rectangle([x, 15, x + 14, 31], fill=c1)
            clock_only = f"{mm:02d}:{ss:02d}"
            txt(bug_r - 12, clock_only, (170, 200, 255, 255), anchor="rm")
            if tag:
                txt(bug_r - 12 - d.textlength(clock_only, font=font) - 10,
                    tag, (150, 160, 190, 255), anchor="rm")

            # BOTTOM SCOREBOARD, TV-style: full team names flanking a big
            # centre score, kit chips, clock tab, scorers listed per side
            nameA, nameB = team_names[0][:26], team_names[1][:26]
            side_w = max(d.textlength(nameA, font=font),
                         d.textlength(nameB, font=font)) + 46
            sb_y0, sb_y1 = h - 46, h - 12
            sb_cy = (sb_y0 + sb_y1) / 2
            side_w += 10 if badge_img else 0
            x0b, x1b = w / 2 - 46 - side_w, w / 2 + 46 + side_w
            d.rounded_rectangle([x0b, sb_y0, x1b, sb_y1], radius=9,
                                fill=(12, 12, 18, 228))
            d.rectangle([w / 2 - 46, sb_y0, w / 2 + 46, sb_y1],
                        fill=(26, 28, 44, 255))
            d.text((w / 2 - 16, sb_cy), str(score[0]), font=font_big,
                   anchor="rm", fill=(255, 255, 255, 255))
            d.line([(w / 2, sb_y0 + 7), (w / 2, sb_y1 - 7)],
                   fill=(255, 200, 60, 255), width=2)
            d.text((w / 2 + 16, sb_cy), str(score[1]), font=font_big,
                   anchor="lm", fill=(255, 255, 255, 255))
            if 0 in badge_img:
                d.paste(img, d.sized(badge_img[0], 26, 26),
                        (x0b + 8, sb_cy - 13), badge_img[0])
            else:
                d.rectangle([x0b + 12, sb_y0 + 9, x0b + 26, sb_y1 - 9], fill=c0)
            d.text((w / 2 - 58, sb_cy), nameA, font=font, anchor="rm",
                   fill=(255, 255, 255, 255))
            if 1 in badge_img:
                d.paste(img, d.sized(badge_img[1], 26, 26),
                        (x1b - 34, sb_cy - 13), badge_img[1])
            else:
                d.rectangle([x1b - 26, sb_y0 + 9, x1b - 12, sb_y1 - 9], fill=c1)
            d.text((w / 2 + 58, sb_cy), nameB, font=font, anchor="lm",
                   fill=(255, 255, 255, 255))
            # scorers so far on a full-width row above the bar (own goals
            # marked, minute in match minutes); clock tab stacks above it
            shown_goals = [g for g in result.goals if g["t"] <= tt]
            srow = 0
            if shown_goals:
                def side_list(side):
                    # group by scorer, football-style: "CR-7000 1' 2' 3'"
                    agg: dict[str, list] = {}
                    for g in shown_goals:
                        if g["team"] != side:
                            continue
                        j = g.get("scorer")
                        gt = 0 if side == "A" else 1
                        nm = pname(j) if j is not None else "unknown"
                        if j is not None and j // 2 != gt:
                            nm += " (OG)"
                        agg.setdefault(nm, []).append(
                            f"{int(g['t'] // 60) + 1}'")
                    return "   ".join(f"{nm} {' '.join(ms)}"
                                      for nm, ms in agg.items())

                def fit(s, limit):
                    while s and d.textlength(s, font=font_sm) > limit:
                        s = s[:-2].rstrip() + "…"
                    return s

                half_room = (x1b - x0b) / 2 - 24
                sa = fit(side_list("A"), half_room)
                sb_txt = fit(side_list("B"), half_room)
                if sa or sb_txt:
                    srow = 26
                    d.rounded_rectangle([x0b, sb_y0 - 26, x1b, sb_y0 - 4],
                                        radius=6, fill=(12, 12, 18, 180))
                    if sa:
                        d.text((x0b + 12, sb_y0 - 15), sa, font=font_sm,
                               anchor="lm", fill=(235, 235, 240, 245))
                    if sb_txt:
                        d.text((x1b - 12, sb_y0 - 15), sb_txt, font=font_sm,
                               anchor="rm", fill=(235, 235, 240, 245))
            # clock tab on the bar's left shoulder (or atop the scorer row)
            tab_w = max(100, d.textlength(clock_str, font=font) + 26)
            d.rounded_rectangle([x0b, sb_y0 - srow - 26, x0b + tab_w,
                                 sb_y0 - srow - 4], radius=6,
                                fill=(12, 12, 18, 215))
            d.text((x0b + tab_w / 2, sb_y0 - srow - 15), clock_str, font=font,
                   anchor="mm", fill=(255, 210, 90, 255))

            def banner(text, color=(255, 235, 160, 255)):
                bw = d.textlength(text, font=font) / 2 + 24
                d.rounded_rectangle([w // 2 - bw, 46, w // 2 + bw, 76],
                                    radius=8, fill=(12, 12, 18, 220))
                d.text((w // 2, 61), text, fill=color, font=font, anchor="mm")

            if halves == 2 and half_banner[0] > -1e8:
                dt_h = tt - half_banner[0]
                if dt_h < HALF_BREAK_S - 1.5:
                    banner(f"HALF TIME   {team_codes[0]} {score[0]} - "
                           f"{score[1]} {team_codes[1]}")
                elif dt_h < HALF_BREAK_S + 2.0:
                    banner("SECOND HALF", (170, 220, 255, 255))
            if match_time_s - tt < FULL_TIME_BANNER_LEAD_S:
                banner(f"FULL TIME   {team_codes[0]} {score[0]} - "
                       f"{score[1]} {team_codes[1]}")
            if tt - drop_banner[0] < 2.5:
                d.rounded_rectangle([w // 2 - 150, 46, w // 2 + 150, 76],
                                    radius=8, fill=(12, 12, 18, 220))
                d.text((w // 2 - 132, 54), "REFEREE: DROPPED BALL (was stuck)",
                       fill=(255, 220, 120, 255), font=font)
            # GOAL banner for 3 s after each goal
            if result.goals and tt - result.goals[-1]["t"] < 3.0:
                gteam = 0 if result.goals[-1]["team"] == "A" else 1
                d.rounded_rectangle([w // 2 - 130, 46, w // 2 + 130, 76],
                                    radius=8, fill=(12, 12, 18, 220))
                d.text((w // 2 - 112, 54),
                       f"GOAL!  {team_names[gteam][:24]}", font=font,
                       fill=_vivid(c0 if gteam == 0 else c1))
            # corner ram countdown, drawn at the corner it belongs to
            for cn in corners:
                frac = cn["charge"] / CORNER_ARM_S
                if frac <= 0.05 and cn["phase"] is None:
                    continue
                pt = project(np.array([cn["rest"][0], cn["rest"][1], 1.0]))
                if pt is None:
                    continue
                cxp, cyp = pt
                if cn["phase"] is not None:
                    label, col = "PUSH", (255, 80, 70, 255)
                else:
                    label = f"{max(0.0, CORNER_ARM_S - cn['charge']):.1f}"
                    col = (255, 210, 90, 255)
                tw = d.textlength(label, font=font) + 12
                d.rounded_rectangle([cxp - tw / 2, cyp - 12, cxp + tw / 2, cyp + 10],
                                    radius=5, fill=(12, 12, 18, 205),
                                    outline=col, width=2)
                d.text((cxp - tw / 2 + 6, cyp - 8), label, fill=col, font=font)

            # SPEECH BUBBLES: player shouts float above the speaker and track
            # them, so spectators always see who said what and where
            pending = []
            for j in range(N_ROBOTS):
                msg, said_t = bubbles[j]
                if not msg or tt - said_t > BUBBLE_S or fallen_flags[j]:
                    continue
                pt = project(ctrls[j].base_pos(data) + np.array([0, 0, 1.05]))
                if pt is None:
                    continue
                words, line, rows_ = msg.split(), "", []
                for wd in words:
                    trial = f"{line} {wd}".strip()
                    if d.textlength(trial, font=font) > 190 and line:
                        rows_.append(line)
                        line = wd
                    else:
                        line = trial
                rows_.append(line)
                rows_ = rows_[:3]
                bw = max(d.textlength(r_, font=font) for r_ in rows_) + 16
                bh = 15 * len(rows_) + 10
                pending.append({"j": j, "rows": rows_, "bw": bw, "bh": bh,
                                "ax": pt[0], "ay": pt[1]})
            # nearest speaker keeps its natural spot; others stack upward so
            # bubbles never cover each other when players are close together
            pending.sort(key=lambda b: -b["ay"])
            placed = []
            for b in pending:
                x0 = min(max(b["ax"] - b["bw"] / 2, 4), w - b["bw"] - 4)
                y0 = b["ay"] - b["bh"] - 14
                for _ in range(N_ROBOTS):
                    clash = next((q for q in placed
                                  if x0 < q[2] and q[0] < x0 + b["bw"]
                                  and y0 < q[3] and q[1] < y0 + b["bh"]), None)
                    if clash is None:
                        break
                    y0 = clash[1] - b["bh"] - 6      # sit above the blocker
                y0 = max(y0, 4)
                placed.append((x0, y0, x0 + b["bw"], y0 + b["bh"]))
                tc = tuple(int(v * 255) for v in team_colors[b["j"] // 2][:3])
                # near-white kits get a darkened edge, or the outline and
                # leader line vanish against the white bubble / grey walls
                if 0.299 * tc[0] + 0.587 * tc[1] + 0.114 * tc[2] > 200:
                    tc = tuple(int(v * 0.55) for v in tc)
                cx_ = x0 + b["bw"] / 2
                # leader line back to the speaker's head, so a raised bubble
                # still clearly belongs to its player
                d.line([(cx_, y0 + b["bh"]), (b["ax"], b["ay"] - 2)],
                       fill=tc + (220,), width=2)
                d.rounded_rectangle([x0, y0, x0 + b["bw"], y0 + b["bh"]],
                                    radius=7, fill=(255, 255, 255, 235),
                                    outline=tc + (255,), width=2)
                for li_, r_ in enumerate(b["rows"]):
                    d.text((x0 + 8, y0 + 5 + 15 * li_), r_,
                           fill=(20, 20, 28, 255), font=font)
            lines = [s for s in (subtitles[0], subtitles[1]) if s]
            if lines:
                band = 14 * len(lines) + 12
                sub_b = sb_y0 - srow - 30   # above the scoreboard stack
                d.rectangle([0, sub_b - band, w, sub_b], fill=(0, 0, 0, 160))
                for li, s in enumerate(lines):
                    home = (s.startswith("MGR A")
                            or f" {team_codes[0][:3]}:" in s)
                    base = c0 if home else c1
                    col = tuple(min(255, v + 90) for v in base[:3]) + (255,)
                    d.text((8, sub_b - band + 6 + 14 * li), s[:130],
                           font=font_sm, fill=col)
            return np.asarray(img)

        renderer.overlay_fn = overlay

    last_action_result = ["ok"] * N_ROBOTS
    consecutive_invalid = [0] * N_ROBOTS
    latencies = [[] for _ in range(N_ROBOTS)]
    deciders = [_AsyncDecider(a) for a in agents] if mode == "realtime" else None
    # HONEST LATENCY: a completed reply is held here until sim time reaches
    # request_t + wall_latency, so thinking costs the same number of SIM
    # seconds it cost wall seconds — the way a physical robot would pay for
    # it. Without this, a loop running Nx slower than sim time (see the
    # realtime-sleep note below) divides every decider's effective latency
    # by N, and N has varied 3.4x-29x across season 2 with the machine's
    # mood. While a reply is held the brain counts as busy: no new request.
    arrival: list = [None] * N_ROBOTS
    last_request_t = [-1e9] * N_ROBOTS
    prev_decision_t: list[float | None] = [None] * N_ROBOTS
    cmd_applied_t: list[float | None] = [None] * N_ROBOTS
    cmd_vx = [0.0] * N_ROBOTS
    rot_expired = [False] * N_ROBOTS
    blocked_flag = [False] * N_ROBOTS
    block_mark: list[tuple | None] = [None] * N_ROBOTS
    last_det = [{"ball": None, "teammates": [], "opponents": []}
                for _ in range(N_ROBOTS)]
    last_skill = [{"skill": "hold", "target": None, "status": "ok"}
                  for _ in range(N_ROBOTS)]
    teammate_msg = [""] * N_ROBOTS
    opponent_msg = [""] * N_ROBOTS    # latest opposition shout overheard
    bubbles: list[tuple] = [("", -1e9)] * N_ROBOTS   # (text, said_at) per player
    say_ok_t = [0.0] * N_ROBOTS                      # shout cooldown per player
    last_said = [""] * N_ROBOTS
    recover_at: list[float | None] = [None] * N_ROBOTS
    stuck_ref = [0.0, (0.0, 0.0)]   # (since_t, ball xy) for the stuck rule
    drop_banner = [-1e9]
    replay_buf: list = []          # rolling qpos snapshots for goal replays
    next_replay_t = [0.0]
    # one snapshot per output frame => sample at the WRITER's rate, never a
    # constant; any mismatch between the two is a playback-speed change
    replay_hz = float(getattr(renderer, "fps", REPLAY_SAMPLE_HZ)
                      or REPLAY_SAMPLE_HZ)
    score = [0, 0]
    last_touch: list[int | None] = [None]  # robot index of last ball touch
    last_touch_team = {0: (None, -1e9), 1: (None, -1e9)}  # per team (j, t)
    touch_t = [-1e9] * N_ROBOTS
    # Opening kickoff gets the same treatment as any other restart: a short
    # hold while both teams decide, then play begins ON the whistle instead
    # of after a decision round-trip of standing about.
    freeze_until = KICKOFF_FREEZE_S
    last_reset_t = [0.0]        # when the pitch was last set for a restart
    held_reply = [None] * N_ROBOTS   # decided during the break, applied at
                                     # the whistle
    t = 0.0
    # halves + sound tape state
    half_banner = [-1e9]           # halftime banner start (match t)
    half_done = [False]
    ev_prev_v = [0.0, 0.0]         # ball velocity at the last event poll
    ev_contact: set = set()        # contact classes seen since the last poll
    next_event_t = [0.0]
    last_ev_t = {"kick": -1e9, "wall": -1e9, "miss": -1e9}
    chance = [None]                # armed near-miss watcher {until, s, best}
    # tackle attribution: last robot-robot contact time per pair
    last_rr = np.full((N_ROBOTS, N_ROBOTS), -1e9)
    TACKLE_WINDOW_S = 1.2
    last_through_t = [-1e9]        # "through on goal" debounce
    THROUGH_MIN_GOAL_M = 2.5
    THROUGH_LANE_W = 1.0
    THROUGH_MIN_SPEED = 0.35   # m/s of ball travel toward the goal
    THROUGH_CLEAR_M = 0.9      # no rival this close: a scrum is not a clear run

    # ------------------------------------------------ manager machinery
    team_message = {0: "", 1: ""}
    tactics_f = open(log_dir / "tactics.jsonl", "w") if log_dir else None
    subtitles = {0: "", 1: ""}  # per-team lines for the overlay closure
    mgr_next_t = {tm: 0.0 for tm in managers}       # next data poll
    mgr_arrival: dict = {tm: None for tm in managers}  # honest-latency hold
    mgr_shout_ok = {tm: 0.0 for tm in managers}     # next allowed shout
    mgr_deciders = ({tm: _AsyncDecider(a) for tm, a in managers.items()}
                    if mode == "realtime" else None)
    mgr_bodies = {tm: N_ROBOTS + k for k, tm in enumerate(sorted(managers))}

    def manager_obs(team):
        bp = data.qpos[ball_qpos_adr:ball_qpos_adr + 3]
        bv = data.qvel[ball_qvel_adr:ball_qvel_adr + 3]
        players = []
        for j in range(N_ROBOTS):
            p = ctrls[j].base_pos(data)
            players.append({
                "id": f"r{j}", "team": "A" if j // 2 == 0 else "B",
                "yours": j // 2 == team,
                "position": [round(float(p[0]), 2), round(float(p[1]), 2)],
                "heading_rad": round(yaw_from_quat(ctrls[j].base_quat(data)), 2),
                "fallen": bool(fallen_flags[j])})
        body = mgr_bodies[team]
        mp = ctrls[body].base_pos(data)
        return {
            "time_remaining_s": round(match_time_s - t, 1),
            "score": {"you": score[team], "them": score[1 - team]},
            "you_attack": {"goal_color": team_color_names[1 - team],
                           "x": PITCH_X if team == 0 else -PITCH_X,
                           "heading": 0.0 if team == 0 else 3.14},
            "ball": {"position": [round(float(bp[0]), 2), round(float(bp[1]), 2)],
                     "velocity": [round(float(bv[0]), 2), round(float(bv[1]), 2)]},
            "players": players,
            "your_body": {"position": [round(float(mp[0]), 2), round(float(mp[1]), 2)],
                          "heading_rad": round(yaw_from_quat(ctrls[body].base_quat(data)), 2),
                          "technical_area": list(TECH_AREAS[team]),
                          "fallen": bool(fallen_flags[body])},
            "last_instruction": team_message[team],
            "seconds_until_shout_allowed": max(
                0.0, round(mgr_shout_ok[team] - t, 1)),
        }

    def apply_manager_reply(team, raw, t_now):
        if not isinstance(raw, dict):
            return
        msg = raw.get("message")
        if isinstance(msg, str) and msg.strip():
            if t_now + 1e-9 < mgr_shout_ok[team]:
                if tactics_f:  # shouted too soon: the league drops it
                    tactics_f.write(json.dumps(
                        {"t": round(t_now, 1),
                         "team": "A" if team == 0 else "B",
                         "suppressed": msg.strip()[:MANAGER_MSG_MAX]}) + "\n")
            else:
                msg = msg.strip()[:MANAGER_MSG_MAX]
                team_message[team] = msg
                mgr_shout_ok[team] = t_now + MANAGER_SHOUT_S
                subtitles[team] = f"MGR {'A' if team == 0 else 'B'} {t_now:.0f}s: {msg}"
                print(f"  [tactic] {subtitles[team]}")
                if tactics_f:
                    tactics_f.write(json.dumps(
                        {"t": round(t_now, 1),
                         "team": "A" if team == 0 else "B",
                         "message": msg}) + "\n")
                    tactics_f.flush()
        mv = raw.get("move")
        body = mgr_bodies[team]
        if isinstance(mv, dict) and not fallen_flags[body]:
            cmd, _ = validate_action(mv, envelope)
            if cmd is not None:
                blenders[body].set_target(cmd, t_now)

    def decision_interval(i, t_now):
        return (DECISION_PERIOD_S if prev_decision_t[i] is None
                else t_now - prev_decision_t[i])

    def apply_skill_reply(i, raw, t_now, obs, latency, error=None):
        """Behaviour-layer reply: {"skill", "target", "say"} (raw velocities
        still accepted, so pixel-driven entrants keep working)."""
        status = "ok"
        chosen = None
        if isinstance(raw, dict) and "skill" in raw:
            name = str(raw.get("skill", "")).strip()
            tgt = raw.get("target")
            target = None
            if isinstance(tgt, str) and tgt.strip().lower() == "ball":
                target = "ball"          # moving target: re-solved every step
            elif isinstance(tgt, (list, tuple)) and len(tgt) >= 2:
                try:
                    target = (float(np.clip(float(tgt[0]), -PITCH_X, PITCH_X)),
                              float(np.clip(float(tgt[1]), -PITCH_Y, PITCH_Y)))
                except (TypeError, ValueError):
                    target = None
            if name not in SKILL_NAMES:
                status = "ignored_invalid"
            else:
                try:
                    lead = float(raw.get("lead_s") or 0.0)
                except (TypeError, ValueError):
                    lead = 0.0
                skills[i].set_skill(name, target, wms[i],
                                    tuple(float(v) for v in ctrls[i].base_pos(data)[:2]),
                                    attack_goal_xy(i), t_now, lead_s=lead)
                if name == "hold":
                    # a stale blender target would otherwise keep a robot
                    # walking after it asked to stand still
                    blenders[i].set_target((0.0, 0.0, 0.0), t_now)
                chosen = {"skill": name,
                          "target": (target if isinstance(target, str)
                                     else list(target) if target else None),
                          "lead_s": lead}
                last_skill[i] = {**chosen, "status": "ok"}
            # spectator-visible player shout
            say = raw.get("say")
            if isinstance(say, str) and say.strip():
                deliver_message(i, say, t_now)
        elif isinstance(raw, dict) and {"vx", "vy", "wz"} & set(raw):
            cmd, status = validate_action(raw, envelope)
            if cmd is not None:
                skills[i].skill = "hold"
                blenders[i].set_target(cmd, t_now)
                cmd_applied_t[i], cmd_vx[i], rot_expired[i] = t_now, cmd[0], False
                chosen = {"raw_velocity": list(cmd)}
        else:
            status = "ignored_invalid"
        if status == "ignored_invalid":
            consecutive_invalid[i] += 1
            result.robots[i].invalid_actions += 1
            if consecutive_invalid[i] >= 3:
                skills[i].set_skill("hold", None, wms[i], (0, 0), (0, 0), t_now)
        else:
            consecutive_invalid[i] = 0
        last_action_result[i] = status
        result.robots[i].decisions += 1
        prev_decision_t[i] = t_now
        latencies[i].append(latency)
        if decisions_f:
            obs_log = {k: v for k, v in obs.items() if not k.startswith("_")}
            decisions_f.write(json.dumps(
                {"robot": i, "t": round(t_now, 2), "obs": obs_log, "raw": raw,
                 "applied": chosen, "status": status,
                 "latency_s": round(latency, 3), "error": error}) + "\n")

    def log_dropped(i, t_now, latency, status, obs=None):
        """Record a decision the engine THREW AWAY.

        Late and hung replies never reach apply_reply, so without this they
        are invisible in a team's own decisions.jsonl — the club sees only
        the calls that beat the clock and concludes all is well. Silent
        failure is the worst kind: log it where they will find it."""
        if decisions_f:
            obs_log = {k: v for k, v in (obs or {}).items()
                       if not k.startswith("_")}
            decisions_f.write(json.dumps(
                {"robot": i, "t": round(t_now, 2), "obs": obs_log,
                 "raw": None, "applied": None, "status": status,
                 "latency_s": round(latency, 3),
                 "error": f"decision discarded: {status}"}) + "\n")
            decisions_f.flush()

    def deliver_message(i, text, t_now):
        """A player shout. Human-readable by league rule, logged in full and
        shown to spectators — nothing shouted on the pitch is hidden.

        Everyone in earshot hears it, and on a pitch this size that is
        everyone: the teammate it was meant for AND both opponents (their
        obs carries it as opponent_says). Same rule humans play under —
        shout "square it!" and the defender hears it too."""
        clean = "".join(ch for ch in str(text) if ch.isprintable()).strip()[:MESSAGE_MAX]
        if not clean:
            return
        # shout discipline: cooldown per player, and no repeats
        too_soon = t_now < say_ok_t[i]
        repeat = clean.lower() == last_said[i].lower()
        if too_soon or repeat:
            if comms_f:
                comms_f.write(json.dumps(
                    {"t": round(t_now, 1), "from": f"r{i}",
                     "team": team_names[team_of[i]], "number": i % 2 + 1,
                     "suppressed": clean,
                     "reason": "repeat" if repeat else "cooldown"}) + "\n")
            return
        say_ok_t[i] = t_now + PLAYER_SHOUT_COOLDOWN_S
        last_said[i] = clean
        mate = i + 1 if i % 2 == 0 else i - 1
        teammate_msg[mate] = clean
        for j in range(N_ROBOTS):
            if team_of[j] != team_of[i]:
                opponent_msg[j] = clean
        bubbles[i] = (clean, t_now)     # spectators see it above the player
        print(f"  [shout] {t_now:5.1f}s #{i % 2 + 1} "
              f"{team_codes[team_of[i]]}: {clean}")
        if comms_f:
            comms_f.write(json.dumps({"t": round(t_now, 1), "from": f"r{i}",
                                      "team": team_names[team_of[i]],
                                      "number": i % 2 + 1, "text": clean}) + "\n")
            comms_f.flush()

    def apply_reply(i, raw, t_now, obs, latency, error=None):
        if obs_mode == "sdk":
            return apply_skill_reply(i, raw, t_now, obs, latency, error)
        cmd, status = validate_action(raw, envelope)
        if cmd is None:
            consecutive_invalid[i] += 1
            result.robots[i].invalid_actions += 1
            if consecutive_invalid[i] >= 3:
                blenders[i].set_target((0.0, 0.0, 0.0), t_now)
        else:
            consecutive_invalid[i] = 0
            blenders[i].set_target(cmd, t_now)
            cmd_applied_t[i], cmd_vx[i], rot_expired[i] = t_now, cmd[0], False
        last_action_result[i] = status
        result.robots[i].decisions += 1
        prev_decision_t[i] = t_now
        latencies[i].append(latency)
        if decisions_f:
            obs_log = {k: v for k, v in obs.items() if not k.startswith("_")}
            decisions_f.write(json.dumps(
                {"robot": i, "t": round(t_now, 2), "obs": obs_log, "raw": raw,
                 "applied": cmd, "status": status,
                 "latency_s": round(latency, 3), "error": error}) + "\n")

    def render_ego(i, t_now):
        ego_r.update_scene(data, camera=f"r{i}_egocam")
        return ego_r.render()

    def next_request_due(i, t_now):
        """When robot i's next observation is expected (either timing mode)."""
        if mode == "paused":
            return (math.floor(t_now / DECISION_PERIOD_S) + 1) * DECISION_PERIOD_S
        if last_request_t[i] < -1e8:
            return t_now
        return last_request_t[i] + request_period

    def maybe_pre_frame(t_now):
        """Render each robot's motion-burst frame PREFRAME_LEAD_S before its
        next decision (once per decision, so 2 renders per player per tick)."""
        for i in range(N_ROBOTS):
            if fallen_flags[i]:
                continue
            due = next_request_due(i, t_now)
            if pre_target[i] != due and t_now >= due - PREFRAME_LEAD_S:
                prev_frame[i] = render_ego(i, t_now)
                prev_frame_t[i] = t_now
                pre_target[i] = due
                if cams is not None:
                    # free extra detector cycle: the frame is already rendered,
                    # so the world model gets two updates per decision
                    run_perception(i, prev_frame[i], t_now)

    def obs_for(i, t_now, interval):
        if obs_mode == "sdk":
            c = ctrls[i]
            p = c.base_pos(data)
            mv = c.base_linvel(data)
            frame = render_ego(i, t_now)
            if frame_dir is not None:
                from PIL import Image
                Image.fromarray(frame).save(
                    frame_dir / f"r{i}_t{t_now:06.1f}.jpg", quality=80)
            run_perception(i, frame, t_now)
            ag, dg = attack_goal_xy(i), attack_goal_xy(1 - team_of[i])
            obs = {
                "time_remaining_s": round(match_time_s - t_now, 1),
                "decision_interval_s": round(interval, 2),
                "you": {"id": f"r{i}", "number": i % 2 + 1,
                        "team": team_names[team_of[i]],
                        "attack_goal_xy": [round(ag[0], 1), 0.0],
                        "defend_goal_xy": [round(dg[0], 1), 0.0]},
                "score": {"you": score[team_of[i]], "them": score[1 - team_of[i]]},
                # localization output (real analog: EKF on field lines)
                "self": {"field_xy": [round(float(p[0]), 2), round(float(p[1]), 2)],
                         "heading_rad": round(yaw_from_quat(c.base_quat(data)), 2),
                         "velocity": [round(float(mv[0]), 2), round(float(mv[1]), 2)],
                         "fallen": bool(fallen_flags[i]),
                         "blocked": bool(blocked_flag[i])},
                "detections": last_det[i],
                "referee": {"ball_stuck_s": round(max(0.0, t_now - stuck_ref[0]), 1),
                            "dropped_ball_after_s": BALL_STUCK_S},
                "field": {"length_m": 2 * PITCH_X, "width_m": 2 * PITCH_Y,
                          "goal_width_m": 2 * GOAL_HALF_W},
                "teammate_says": teammate_msg[i],
                "opponent_says": opponent_msg[i],
                "last_skill": last_skill[i],
                "camera": {"frames": 2, "note": "raw frames also attached "
                           "(_frames) if you prefer your own vision"},
            }
            obs["_frames"] = [prev_frame[i] if prev_frame[i] is not None else frame,
                              frame]
            obs["_frame"] = frame
            return obs
        if obs_mode == "camera":
            # the real-life contract: onboard senses + the coach's shouts only.
            # No positions of anything — the ball, mates, opponents, and the
            # colored goals exist solely in the attached camera frames.
            c = ctrls[i]
            mv = c.base_linvel(data)
            obs = {
                "time_remaining_s": round(match_time_s - t_now, 1),
                "decision_interval_s": round(interval, 2),
                "you": {"id": f"r{i}",
                        "team": team_names[i // 2],
                        "attack_goal_color": team_color_names[1 - i // 2],
                        "attack_goal_heading": 0.0 if i // 2 == 0 else 3.14},
                "score": {"you": score[i // 2], "them": score[1 - i // 2]},
                "self": {"heading_rad": round(yaw_from_quat(c.base_quat(data)), 2),
                         "velocity": [round(float(mv[0]), 2), round(float(mv[1]), 2)],
                         "fallen": bool(fallen_flags[i]),
                         "blocked": bool(blocked_flag[i])},
                "manager_says": team_message[i // 2],
                "last_action_result": last_action_result[i],
            }
            px = render_ego(i, t_now)
            if frame_dir is not None:
                from PIL import Image
                fname = f"r{i}_t{t_now:06.1f}.jpg"
                Image.fromarray(px).save(frame_dir / fname, quality=80)
                obs["_frame_file"] = fname
            # motion burst: [frame from PREFRAME_LEAD_S ago, frame NOW]
            fresh = (prev_frame[i] is not None
                     and 0.0 < t_now - prev_frame_t[i] <= 2 * PREFRAME_LEAD_S)
            old = prev_frame[i] if fresh else px
            obs["_frames"] = [old, px]
            obs["_frame"] = px
            obs["camera"] = {"frames": 2,
                             "dt_s": round(t_now - prev_frame_t[i], 2) if fresh else 0.0}
            return obs
        obs = _match_observation(ctrls, data, ball_qpos_adr, ball_qvel_adr,
                                 i, t_now, score, fallen_flags,
                                 last_action_result, match_time_s, interval,
                                 blocked_flag[i])
        obs["manager_says"] = team_message[i // 2]
        return obs

    def kickoff_reset():
        """Ball + ALL players back to kickoff spots, upright — a restart is
        a restart: fallen robots are stood up (their recovery clock is moot
        once play is being reset around them anyway)."""
        data.qpos[ball_qpos_adr:ball_qpos_adr + 7] = home_qpos[
            ball_qpos_adr:ball_qpos_adr + 7]
        data.qvel[ball_qvel_adr:ball_qvel_adr + 6] = 0.0
        last_touch[0] = None          # play is not engaged until someone touches it
        last_touch_team[0] = last_touch_team[1] = (None, -1e9)
        for i, c in enumerate(ctrls):
            if i >= N_ROBOTS:         # manager keeps pacing the dugout
                continue
            adr = model.jnt_qposadr[model.body(f"r{i}_pelvis").jntadr[0]]
            vadr = model.jnt_dofadr[model.body(f"r{i}_pelvis").jntadr[0]]
            data.qpos[adr:adr + 19] = home_qpos[adr:adr + 19]
            data.qvel[vadr:vadr + 18] = 0.0
            if fallen_flags[i]:
                c.reset()             # policy action history + gait clock
                falls[i] = FallTracker()
                fallen_flags[i] = False
                recover_at[i] = None
                result.robots[i].recoveries += 1
            blocked_flag[i] = False
            block_mark[i] = None
            blenders[i].set_target((0.0, 0.0, 0.0), t)
            if skills is not None:
                skills[i].skill, skills[i].path = "hold", []
        for j in range(N_ROBOTS):
            bubbles[j] = ("", -1e9)   # a restart wipes the shouts: chatter
            teammate_msg[j] = ""      # from before it reads as nonsense
            opponent_msg[j] = ""      # floating over teleported players
            held_reply[j] = None      # decided against the old positions
        last_reset_t[0] = t
        mujoco.mj_forward(model, data)

    def play_goal_replay(scorer, goal_t):
        """Halt the match and cut to the scorer's own head camera for the
        seconds before the goal, then resume. Returns (wall_s, video_s):
        wall-clock spent rendering and seconds of video inserted."""
        if renderer is None or not replay_buf or scorer is None:
            return 0.0, 0.0
        from PIL import Image, ImageDraw, ImageFont
        try:
            font_r = ImageFont.load_default(size=17)
        except TypeError:
            font_r = ImageFont.load_default()
        cam_name = f"r{scorer}_egocam"
        try:
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        except Exception:
            return 0.0, 0.0
        t_wall0 = time.time()
        # the "scorer cam" tag blinks on a period in SECONDS, not in frames
        blink_n = max(1, round(replay_hz * 0.24))
        save_q, save_v = data.qpos.copy(), data.qvel.copy()
        # name the scorer, not just their shirt: "#1 CR-7000 (RMA)"
        who = (f"#{scorer % 2 + 1} {pname(scorer)} "
               f"({team_codes[team_of[scorer]]})")
        col = _vivid(tuple(int(v * 255)
                           for v in team_colors[team_of[scorer]][:3]))
        wrote = 0
        for n, (snap_t, snap_q) in enumerate(replay_buf):
            data.qpos[:] = snap_q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            try:
                renderer.renderer.update_scene(data, camera=cam_name)
            except Exception:
                break
            frame = renderer.renderer.render()
            img = Image.fromarray(frame)
            dr = ImageDraw.Draw(img, "RGBA")
            wv, hv = img.size
            # top-left: who scored; top-right: REPLAY, in the slot the
            # LIVE tag occupies during play
            cap = f"GOAL {who}"
            cap_w = dr.textlength(cap, font=font_r)
            dr.rounded_rectangle([10, 8, 30 + cap_w, 38], radius=7,
                                 fill=(12, 12, 18, 220))
            dr.text((20, 15), cap, fill=col, font=font_r)
            rep_w = dr.textlength("REPLAY", font=font_r)
            dr.rounded_rectangle([wv - 40 - rep_w, 8, wv - 12, 34],
                                 radius=7, fill=(12, 12, 18, 220))
            dr.ellipse([wv - 31 - rep_w, 16, wv - 21 - rep_w, 26],
                       fill=(235, 55, 45, 255))
            dr.text((wv - 16 - rep_w, 21), "REPLAY",
                    fill=(255, 255, 255, 255), font=font_r, anchor="lm")
            if (n // blink_n) % 2 == 0:   # blinking "scorer cam" tag
                tag = f"{who} head camera"
                tw_ = dr.textlength(tag, font=font_r)
                dr.rounded_rectangle([14, hv - 34, 26 + tw_, hv - 8],
                                     radius=6, fill=(12, 12, 18, 190))
                dr.text((20, hv - 30), tag,
                        fill=(255, 255, 255, 235), font=font_r)
            renderer.writer.add(np.asarray(img))
            wrote += 1
        data.qpos[:] = save_q
        data.qvel[:] = save_v
        mujoco.mj_forward(model, data)
        replay_buf.clear()
        print(f"  [replay] goal by {who}: {wrote} frames from the scorer's "
              f"head camera ({time.time() - t_wall0:.1f}s to render)")
        return time.time() - t_wall0, round(wrote / replay_hz, 2)

    wall_start = time.time()
    try:
        for k in range(max_steps):
            if percept_r is not None and t >= next_percept_t[0] - 1e-9:
                for i in range(N_ROBOTS):
                    if fallen_flags[i]:
                        continue
                    percept_r.update_scene(data, camera=f"r{i}_egocam")
                    c_ = ctrls[i]
                    p_ = c_.base_pos(data)
                    perceive(model, data, percept_cams[i], percept_r.render(),
                             (float(p_[0]), float(p_[1])),
                             yaw_from_quat(c_.base_quat(data)), i,
                             player_xy_all(), team_of, BALL_R, t, wms[i],
                             pitch=(PITCH_X, PITCH_Y), wall_m=BALL_WALL_M,
                             min_px=percept_min_px)
                next_percept_t[0] += PERCEPT_PERIOD_S
            if ego_r is not None:
                maybe_pre_frame(t)
            if mode == "paused":
                if k % decision_every == 0 and t >= freeze_until:
                    for i in range(N_ROBOTS):
                        if fallen_flags[i]:
                            continue
                        obs = obs_for(i, t, decision_interval(i, t))
                        t0 = time.time()
                        error = None
                        try:
                            raw = agents[i].decide(obs)
                        except Exception as e:
                            raw, error = None, f"{type(e).__name__}: {e}"
                        apply_reply(i, raw, t, obs, time.time() - t0, error)
                for tm, mgr in managers.items():
                    if t >= mgr_next_t[tm]:
                        try:
                            raw = mgr.decide(manager_obs(tm))
                        except Exception:
                            raw = None
                        apply_manager_reply(tm, raw, t)
                        mgr_next_t[tm] = t + manager_period_s
            else:
                for i in range(N_ROBOTS):
                    if fallen_flags[i]:
                        continue
                    if deciders[i].in_flight_s > DECIDE_ABANDON_S:
                        stuck = deciders[i].in_flight_s
                        deciders[i].abandon()
                        result.robots[i].abandoned += 1
                        log_dropped(i, t, stuck, "abandoned_hung_call")
                        print(f"  [bridge] {t:5.1f}s robot {i} decision "
                              f"abandoned after {DECIDE_ABANDON_S:.0f}s in "
                              "flight (hung provider call)")
                    reply = deciders[i].poll()
                    if honest_latency:
                        if reply is not None:      # done thinking in WALL terms
                            arrival[i] = reply
                            reply = None
                        if (arrival[i] is not None
                                and t >= arrival[i][1] + arrival[i][2]):
                            reply = arrival[i]     # sim has caught up: deliver
                            arrival[i] = None
                    if reply is not None:
                        raw, req_t, latency, error, req_obs = reply
                        if decision_deadline_s and latency > decision_deadline_s:
                            result.robots[i].missed_deadlines += 1
                            log_dropped(i, t, latency, "missed_deadline",
                                        req_obs)
                        elif t < freeze_until:
                            if req_t >= last_reset_t[0]:
                                # decided against the restart positions: keep
                                # it warm and play it the instant we kick off
                                held_reply[i] = (raw, req_obs, latency, error)
                            # otherwise it describes a world that no longer
                            # exists — void, as before
                        else:
                            apply_reply(i, raw, t, req_obs, latency, error)
                    if (held_reply[i] is not None and t >= freeze_until):
                        raw_h, obs_h, lat_h, err_h = held_reply[i]
                        held_reply[i] = None
                        apply_reply(i, raw_h, t, obs_h, lat_h, err_h)
                    if (not deciders[i].busy
                            and arrival[i] is None
                            and t >= freeze_until - KICKOFF_LEAD_S
                            and t >= last_reset_t[0]
                            and t - last_request_t[i] >= request_period - 1e-9):
                        interval = (request_period if last_request_t[i] < -1e8
                                    else t - last_request_t[i])
                        deciders[i].submit(obs_for(i, t, interval), t)
                        last_request_t[i] = t
                for tm, dec in (mgr_deciders or {}).items():
                    if dec.in_flight_s > MGR_ABANDON_S:
                        dec.abandon()
                    reply = dec.poll()
                    if honest_latency:
                        if reply is not None:
                            mgr_arrival[tm] = reply
                            reply = None
                        if (mgr_arrival[tm] is not None
                                and t >= mgr_arrival[tm][1] + mgr_arrival[tm][2]):
                            reply = mgr_arrival[tm]
                            mgr_arrival[tm] = None
                    if reply is not None:
                        apply_manager_reply(tm, reply[0], t)
                    if (not dec.busy and mgr_arrival[tm] is None
                            and t >= mgr_next_t[tm]):
                        dec.submit(manager_obs(tm), t)
                        mgr_next_t[tm] = t + manager_period_s

            for i in range(N_ROBOTS):
                if (cmd_applied_t[i] is not None and not rot_expired[i]
                        and t - cmd_applied_t[i] > FOOTBALL_ROT_HOLD_S):
                    blenders[i].set_target((cmd_vx[i], 0.0, 0.0), t)
                    rot_expired[i] = True

            stopped = t < freeze_until      # half time / after a goal
            for i, c in enumerate(ctrls):
                if stopped and i < N_ROBOTS:
                    c.set_command(0.0, 0.0, 0.0)
                    cmd_vx[i] = 0.0
                    c.apply_control(model, data)
                    continue
                if (skills is not None and i < N_ROBOTS
                        and skills[i].skill != "hold" and not fallen_flags[i]):
                    # closed loop: the skill re-issues velocity every control
                    # step off the world model, so no staleness and no
                    # rotation watchdog is needed
                    xy = tuple(float(v) for v in c.base_pos(data)[:2])
                    cmd = skills[i].step(wms[i], xy, yaw_from_quat(c.base_quat(data)),
                                         attack_goal_xy(i), t)
                    cmd_vx[i] = cmd[0]      # keeps the "blocked" detector live
                    c.set_command(*cmd)
                else:
                    c.set_command(*blenders[i].value(t))
                c.apply_control(model, data)
            mujoco.mj_step(model, data)
            for c in ctrls:
                c.advance(data)
            t += dt

            if mode == "realtime":
                # This sleep only engages when the loop is running FASTER
                # than sim time, and a league match never is — m1-m11 ran
                # 3.4x to 9.2x slower, paced by how long the gaffers take to
                # answer. So the wall-clock-per-sim-second ratio is a real
                # quantity that varies per match, and anything that slows the
                # loop raises it: making the render heavier (854x480/25 ->
                # 1280x720/50 on 2026-08-21) is one such thing.
                #
                # It matters because the deciders are ASYNCHRONOUS: an agent
                # thinks in wall time while the sim advances, so a slower
                # loop means fewer sim steps pass before its answer lands,
                # and the players effectively react sooner. That is a
                # fairness property, not a cosmetic one.
                #
                # Measured, not assumed: m11 (the first 720p50 match) came in
                # at 6.88x, inside the 3.37x-9.22x spread of the ten 480p25
                # matches before it. The render's contribution is swamped by
                # LLM latency variance, so no notice was owed. Re-check this
                # if the render ever gets heavier again, or if the gaffers
                # get much faster and stop dominating the ratio.
                target = wall_start + t / realtime_factor
                ahead = target - time.time()
                if ahead > 0.002:
                    time.sleep(ahead)

            for i, c in enumerate(ctrls):
                if not fallen_flags[i]:
                    falls[i].update(t, c.base_pos(data)[2],
                                    tilt_from_quat(c.base_quat(data)))
                    if falls[i].fallen:
                        fallen_flags[i] = True
                        if i < N_ROBOTS:
                            result.robots[i].falls += 1
                            recover_at[i] = t + FALL_RECOVERY_S
                            # tackle attribution: who touched them last?
                            j_by = int(np.argmax(last_rr[i]))
                            by = (j_by if last_rr[i, j_by] > t - TACKLE_WINDOW_S
                                  else None)
                            ev = {"t": round(t, 2), "kind": "fall", "who": i}
                            note = ""
                            if by is not None:
                                ev["by"] = by
                                ev["opponent"] = team_of[by] != team_of[i]
                                note = (f" (contact from {pname(by)}"
                                        f" {team_codes[team_of[by]]})")
                            result.events.append(ev)
                            print(f"  [down] {t:5.1f}s #{i % 2 + 1} "
                                  f"{team_codes[team_of[i]]} — recovering"
                                  + note)
                        else:
                            print(f"  [down] {t:5.1f}s manager body {i}")
                        blenders[i].set_target((0.0, 0.0, 0.0), t)
                    if i >= N_ROBOTS:
                        continue  # manager body: no blocked bookkeeping
                    p = c.base_pos(data)
                    if block_mark[i] is None:
                        block_mark[i] = (t, float(p[0]), float(p[1]))
                    elif t - block_mark[i][0] >= 1.0:
                        mt, mx, my = block_mark[i]
                        rate = float(np.hypot(p[0] - mx, p[1] - my)) / (t - mt)
                        blocked_flag[i] = cmd_vx[i] > 0.3 and rate < 0.12
                        block_mark[i] = (t, float(p[0]), float(p[1]))

            # SELF-RECOVERY: after the get-up interval, stand the robot back
            # up where it fell (see FALL_RECOVERY_S)
            for i in range(N_ROBOTS):
                if not fallen_flags[i] or recover_at[i] is None or t < recover_at[i]:
                    continue
                jadr = model.jnt_qposadr[model.body(f"r{i}_pelvis").jntadr[0]]
                vadr = model.jnt_dofadr[model.body(f"r{i}_pelvis").jntadr[0]]
                px = float(np.clip(data.qpos[jadr], -PITCH_X + 0.4, PITCH_X - 0.4))
                py = float(np.clip(data.qpos[jadr + 1], -PITCH_Y + 0.4, PITCH_Y - 0.4))
                # stand up on the spot: kickoff free-joint + leg pose, keep x/y
                data.qpos[jadr:jadr + 19] = home_qpos[jadr:jadr + 19]
                data.qpos[jadr], data.qpos[jadr + 1] = px, py
                data.qvel[vadr:vadr + 18] = 0.0
                mujoco.mj_forward(model, data)
                ctrls[i].reset()        # clear policy action history + gait clock
                falls[i] = FallTracker()
                fallen_flags[i] = False
                recover_at[i] = None
                blocked_flag[i] = False
                block_mark[i] = None
                if skills is not None:
                    skills[i].skill, skills[i].path = "hold", []
                blenders[i].set_target((0.0, 0.0, 0.0), t)
                result.robots[i].recoveries += 1
                print(f"  [up] {t:5.1f}s #{i % 2 + 1} {team_codes[team_of[i]]} "
                      f"back on its feet")

            # touchline governor: each manager's body stays in its dugout
            for tm, body in mgr_bodies.items():
                if fallen_flags[body]:
                    continue
                p = ctrls[body].base_pos(data)
                x0, x1, y0, y1 = TECH_AREAS[tm]
                if not (x0 - 0.2 <= p[0] <= x1 + 0.2 and y0 - 0.2 <= p[1] <= y1 + 0.2):
                    cxa, cya = (x0 + x1) / 2, (y0 + y1) / 2
                    yaw = yaw_from_quat(ctrls[body].base_quat(data))
                    err = (np.arctan2(cya - p[1], cxa - p[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
                    blenders[body].set_target(
                        (0.4, 0.0, float(np.clip(err, -1.0, 1.0))), t)

            # ball touches: last_touch tracks possession for goal credit;
            # per-robot 1 s debounce keeps a scrum from counting every frame
            for ci in range(data.ncon):
                g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
                o1, o2 = geom_owner[g1], geom_owner[g2]
                if o1 >= 0 and o2 >= 0 and o1 != o2:
                    last_rr[o1, o2] = last_rr[o2, o1] = t  # tackle memory
                if ball_geom in (g1, g2):
                    other_g = g2 if g1 == ball_geom else g1
                    if other_g in wall_gids:
                        ev_contact.add("wall")
                    for cn in corners:
                        if other_g == cn["gid"]:
                            cn["last_touch_t"] = t
                            ev_contact.add("wall")
                    for i in range(N_ROBOTS):
                        if other_g in robot_geoms[i]:
                            ev_contact.add("robot")
                            if t - touch_t[i] > 1.0:
                                result.robots[i].touches += 1
                                # same debounce, same instant: the tape and
                                # the per-robot total can never disagree
                                result.events.append(
                                    {"t": round(t, 2), "kind": "touch",
                                     "who": i})
                            touch_t[i] = t
                            last_touch[0] = i
                            last_touch_team[team_of[i]] = (i, t)
                            # THROUGH ON GOAL: a player driving the ball at
                            # the opposition net with nobody in the way.
                            # THREE conditions, all necessary — an empty
                            # lane alone called defenders retreating toward
                            # their OWN goal "through on goal" (10 of 54
                            # calls in fixture 3 were wrong that way):
                            #   1. the lane ball -> their goal is clear
                            #   2. the player is BEHIND the ball (the ball
                            #      is goal-ward of them, so their momentum
                            #      carries it forward, not backward)
                            #   3. the touch actually sent the ball
                            #      goal-ward, not sideways or back
                            if t - last_through_t[0] > 6.0:
                                gx_, gy_ = attack_goal_xy(i)
                                bx_ = float(data.qpos[ball_qpos_adr])
                                by_ = float(data.qpos[ball_qpos_adr + 1])
                                dgx, dgy = gx_ - bx_, gy_ - by_
                                dg = math.hypot(dgx, dgy)
                                if dg > THROUGH_MIN_GOAL_M:
                                    ux, uy = dgx / dg, dgy / dg
                                    pi_ = ctrls[i].base_pos(data)
                                    # (2) ball goal-ward of the player
                                    behind = ((bx_ - pi_[0]) * ux
                                              + (by_ - pi_[1]) * uy) > 0.05
                                    # (3) ball travelling goal-ward
                                    bvx_ = float(data.qvel[ball_qvel_adr])
                                    bvy_ = float(data.qvel[ball_qvel_adr + 1])
                                    goalward = (bvx_ * ux + bvy_ * uy) > THROUGH_MIN_SPEED
                                    clear = behind and goalward
                                    for jj in range(N_ROBOTS):
                                        if not clear or jj == i:
                                            continue
                                        pj = ctrls[jj].base_pos(data)
                                        u = ((pj[0] - bx_) * ux
                                             + (pj[1] - by_) * uy)
                                        lat = (-(pj[0] - bx_) * uy
                                               + (pj[1] - by_) * ux)
                                        if 0.3 < u < dg and abs(lat) < THROUGH_LANE_W:
                                            clear = False
                                            break
                                        # (4) not a scrum: nobody breathing
                                        # down their neck at the ball
                                        if math.hypot(pj[0] - bx_,
                                                      pj[1] - by_) < THROUGH_CLEAR_M:
                                            clear = False
                                            break
                                    if clear:
                                        last_through_t[0] = t
                                        result.events.append(
                                            {"t": round(t, 2),
                                             "kind": "through", "who": i})
                                        print(f"  [through] {t:5.1f}s "
                                              f"{pname(i)} "
                                              f"({team_codes[team_of[i]]}) "
                                              f"has a clear run at goal")
                            break

            if renderer is not None and t >= next_replay_t[0] - 1e-9:
                replay_buf.append((t, data.qpos.copy()))
                del replay_buf[:-int(REPLAY_S * replay_hz)]
                next_replay_t[0] += 1.0 / replay_hz

            if state_rec is not None and t >= state_rec["next_t"] - 1e-9:
                state_rec["t"].append(t)
                state_rec["xpos"].append(data.xpos.astype(np.float32))
                state_rec["xquat"].append(data.xquat.astype(np.float32))
                state_rec["next_t"] += 1.0 / STATE_RECORD_HZ

            if telemetry_f is not None and t >= next_telemetry_t - 1e-9:
                bp = data.qpos[ball_qpos_adr:ball_qpos_adr + 3]
                telemetry_f.write(json.dumps({
                    "t": round(t, 1),
                    "ball": [round(float(bp[0]), 2), round(float(bp[1]), 2)],
                    "robots": [[round(float(v), 2) for v in ctrls[j].base_pos(data)[:2]]
                               for j in range(N_ROBOTS)],
                    "score": list(score)}) + "\n")
                next_telemetry_t += TELEMETRY_PERIOD_S

            # CORNER RAMS: charge while the ball leans on a panel, then fire
            for cn in corners:
                if cn["phase"] is None:
                    bxx = float(data.qpos[ball_qpos_adr])
                    byy = float(data.qpos[ball_qpos_adr + 1])
                    bsp = float(np.hypot(data.qvel[ball_qvel_adr],
                                         data.qvel[ball_qvel_adr + 1]))
                    in_zone = (np.sign(bxx) == np.sign(cn["rest"][0])
                               and np.sign(byy) == np.sign(cn["rest"][1])
                               and abs(bxx) > PITCH_X - CORNER_ZONE_X
                               and abs(byy) > PITCH_Y - CORNER_ZONE_Y
                               and bsp < CORNER_SLOW_MPS)
                    if in_zone or t - cn["last_touch_t"] < 0.4:
                        cn["charge"] = min(CORNER_ARM_S, cn["charge"] + dt)
                    else:
                        cn["charge"] = max(0.0, cn["charge"] - dt * 0.7)
                    if cn["charge"] >= CORNER_ARM_S:
                        cn["phase"], cn["phase_t"] = "extend", t
                        result.events.append(
                            {"t": round(t, 2), "kind": "ram", "mag": 1.0})
                        print(f"  [ram] {t:5.1f}s corner actuator firing at "
                              f"({cn['rest'][0]:+.1f},{cn['rest'][1]:+.1f})")
                else:
                    el = t - cn["phase_t"]
                    if cn["phase"] == "extend":
                        f = min(1.0, el / CORNER_EXTEND_S)
                        if f >= 1.0:
                            cn["phase"], cn["phase_t"] = "hold", t
                    elif cn["phase"] == "hold":
                        f = 1.0
                        if el >= CORNER_HOLD_S:
                            cn["phase"], cn["phase_t"] = "retract", t
                    elif cn["phase"] == "retract":
                        f = max(0.0, 1.0 - el / CORNER_RETRACT_S)
                        if f <= 0.0:
                            cn["phase"], cn["charge"] = None, 0.0
                            cn["phase_t"] = t
                    else:
                        f = 0.0
                    data.mocap_pos[cn["mid"]] = cn["rest"] + cn["inward"] * (
                        CORNER_STROKE_M * f)
                # arming light on the panel itself (an indicator strip in real
                # hardware): grey when idle, amber charging, red when firing
                frac = cn["charge"] / CORNER_ARM_S
                if cn["phase"] is not None:
                    model.geom_rgba[cn["vgid"]] = (0.95, 0.15, 0.1, 1.0)
                elif frac > 0.05:
                    model.geom_rgba[cn["vgid"]] = (0.75 + 0.2 * frac,
                                                   0.75 - 0.45 * frac,
                                                   0.78 - 0.6 * frac, 1.0)
                else:
                    model.geom_rgba[cn["vgid"]] = (0.75, 0.75, 0.78, 1.0)

            # SOUND TAPE: ball impulses sampled at 25 Hz, classified by what
            # the ball touched since the last poll. Consumed after the match
            # by gauntlet/broadcast_audio.py — kicks, wall thuds, post pings,
            # near-miss "oooh"s all key off this.
            if t >= next_event_t[0] - 1e-9:
                bvx = float(data.qvel[ball_qvel_adr])
                bvy = float(data.qvel[ball_qvel_adr + 1])
                dv = math.hypot(bvx - ev_prev_v[0], bvy - ev_prev_v[1])
                bx_ = float(data.qpos[ball_qpos_adr])
                by_ = float(data.qpos[ball_qpos_adr + 1])
                if dv > EVENT_DV_MPS:
                    kind = None
                    # `last_touch[0] is not None` is not paranoia. A RESTART
                    # (goal, half time, stuck-ball drop) zeroes the ball and
                    # clears last_touch, but it does NOT clear ev_contact —
                    # so a robot touch from BEFORE the restart can still be
                    # sitting in the set on the next poll, while the jump
                    # that tripped dv is the restart teleporting the ball.
                    # That is not a kick and there is nobody to credit it
                    # to. Crediting it crashed m28 on `int(None)` after a
                    # 10-0 half made restarts frequent enough to collide
                    # with a touch inside one 25 Hz window.
                    # Only the kick branch is gated: a corner ram panel can
                    # legitimately fire an untouched ball into a wall, and
                    # those wall/post sounds must still be recorded.
                    if ("robot" in ev_contact and last_touch[0] is not None
                            and t - last_ev_t["kick"] > 0.3):
                        kind = "kick"
                    elif "wall" in ev_contact and t - last_ev_t["wall"] > 0.4:
                        kind = ("post" if any(math.hypot(bx_ - px, by_ - py) < 0.35
                                              for px, py in post_xy) else "wall")
                    if kind:
                        last_ev_t["kick" if kind == "kick" else "wall"] = t
                        ev = {"t": round(t, 2), "kind": kind,
                              "mag": round(dv, 2)}
                        if kind == "kick":
                            # a robot touched inside this poll window, so the
                            # last toucher IS the kicker. Without it the tape
                            # says a kick happened but not by whom, and every
                            # consumer has to re-derive it from positions.
                            ev["who"] = int(last_touch[0])
                        result.events.append(ev)
                # near-miss watcher: a fast ball closing on a goal mouth arms
                # a "chance"; if no goal follows, that was a near miss
                for s in (1, -1):
                    if (s * bvx > 0.9 and s * bx_ > PITCH_X - 3.0
                            and abs(by_) < GOAL_HALF_W + 0.8):
                        gap = abs(s * PITCH_X - bx_) + max(
                            0.0, abs(by_) - GOAL_HALF_W)
                        if chance[0] is None or t > chance[0]["until"]:
                            chance[0] = {"until": t + 2.5, "best": gap}
                        else:
                            chance[0]["until"] = t + 2.5
                if chance[0] is not None:
                    gap0 = min(abs(PITCH_X - abs(bx_)) + max(
                        0.0, abs(by_) - GOAL_HALF_W), chance[0]["best"])
                    chance[0]["best"] = gap0
                    if t > chance[0]["until"]:
                        if chance[0]["best"] < 1.4 and t - last_ev_t["miss"] > 4.0:
                            result.events.append(
                                {"t": round(t, 2), "kind": "near_miss",
                                 "mag": round(1.4 - chance[0]["best"], 2)})
                            last_ev_t["miss"] = t
                        chance[0] = None
                ev_prev_v[0], ev_prev_v[1] = bvx, bvy
                ev_contact.clear()
                next_event_t[0] += EVENT_POLL_S

            # HALFTIME: at the midpoint everything resets to kickoff and play
            # pauses for HALF_BREAK_S (second half is a fresh kickoff; ends
            # are not swapped — the pocket colors are the teams' identities)
            if halves == 2 and not half_done[0] and t >= match_time_s / 2:
                half_done[0] = True
                result.half_breaks.append(round(t, 1))
                kickoff_reset()
                freeze_until = t + HALF_BREAK_S
                half_banner[0] = t
                chance[0] = None
                print(f"  [half] {t:5.1f}s HALF TIME "
                      f"{team_codes[0]} {score[0]} - {score[1]} {team_codes[1]}")

            # REFEREE: ball stuck against a wall / in a scrum -> dropped ball
            bpos = (float(data.qpos[ball_qpos_adr]), float(data.qpos[ball_qpos_adr + 1]))
            engaged = last_touch[0] is not None and any(
                not fallen_flags[j]
                and math.hypot(ctrls[j].base_pos(data)[0] - bpos[0],
                               ctrls[j].base_pos(data)[1] - bpos[1]) < STUCK_ENGAGE_M
                for j in range(N_ROBOTS))
            moved = math.hypot(bpos[0] - stuck_ref[1][0],
                               bpos[1] - stuck_ref[1][1]) > BALL_STUCK_M
            if moved or not engaged:
                stuck_ref[0], stuck_ref[1] = t, bpos
            elif (referee_drop
                  and t - stuck_ref[0] >= BALL_STUCK_S and t >= freeze_until
                  and math.hypot(bpos[0], bpos[1]) > 0.7):   # already central

                data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = (0.0, 0.0, BALL_R)
                data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
                data.qvel[ball_qvel_adr:ball_qvel_adr + 6] = 0.0
                mujoco.mj_forward(model, data)
                result.dropped_balls.append(round(t, 1))
                stuck_ref[0], stuck_ref[1] = t, (0.0, 0.0)
                drop_banner[0] = t
                freeze_until = t + KICKOFF_FREEZE_S
                last_touch[0] = None
                last_touch_team[0] = last_touch_team[1] = (None, -1e9)
                if skills is not None:
                    for j in range(N_ROBOTS):
                        skills[j].path = []
                print(f"  [referee] {t:5.1f}s ball stuck at "
                      f"({bpos[0]:+.1f},{bpos[1]:+.1f}) -> dropped ball at centre")

            # goal: ball center crosses the goal plane inside the mouth
            # (center past the posts = ball 90%+ over the line)
            bx = float(data.qpos[ball_qpos_adr])
            by = float(data.qpos[ball_qpos_adr + 1])
            if abs(bx) > PITCH_X and abs(by) < GOAL_HALF_W:
                scoring_team = 0 if bx > 0 else 1  # A attacks +x
                score[scoring_team] += 1
                # goal credit, football-style: the scoring team's last
                # toucher gets it if they forced it within 3 s (a defender's
                # glancing deflection doesn't steal the goal); otherwise the
                # last toucher of record concedes an own goal
                atk_j, atk_t = last_touch_team[scoring_team]
                scorer_idx = (atk_j if atk_j is not None and t - atk_t <= 3.0
                              else last_touch[0])
                chance[0] = None      # the chance came off: no near-miss cheer
                result.goals.append({
                    "t": round(t, 1), "team": "A" if scoring_team == 0 else "B",
                    "scorer": scorer_idx})
                # cut to the replay BEFORE resetting, so the buffered run-up
                # is what spectators see; the match clock is halted meanwhile
                spent, replay_vs = play_goal_replay(scorer_idx, t)
                result.goals[-1]["replay_s"] = replay_vs
                if mode == "realtime":
                    wall_start += spent      # replay time is not match time
                kickoff_reset()
                freeze_until = t + KICKOFF_FREEZE_S

            if renderer:
                aim_camera(t)
                renderer.maybe_frame(data, t)
    finally:
        if state_rec is not None and state_rec["t"]:
            np.savez_compressed(
                log_dir / "states.npz",
                t=np.asarray(state_rec["t"], dtype=np.float32),
                xpos=np.stack(state_rec["xpos"]),
                xquat=np.stack(state_rec["xquat"]),
                body_names=np.array([model.body(b).name
                                     for b in range(model.nbody)]),
                hz=np.float32(STATE_RECORD_HZ))
        if renderer:
            renderer.close()
        if decisions_f:
            decisions_f.close()
        if tactics_f:
            tactics_f.close()
        if telemetry_f:
            telemetry_f.close()
        if comms_f:
            comms_f.close()
        if ego_r is not None:
            ego_r.close()
        if percept_r is not None:
            percept_r.close()

    for i in range(N_ROBOTS):
        r = result.robots[i]
        r.fell = falls[i].fallen
        r.fall_time_s = round(falls[i].fall_time, 1) if falls[i].fallen else None
        if latencies[i]:
            r.mean_decision_latency_s = round(float(np.mean(latencies[i])), 3)
    result.score = score
    result.winner = ("draw" if score[0] == score[1]
                     else ("A" if score[0] > score[1] else "B"))
    # cost roll-up across players + managers (API usage fields; $0 for local)
    from .llm import estimate_cost
    def billable(obj, depth=0):
        """Every metered brain reachable from a team's player object.

        Clubs legitimately WRAP an LLM agent inside their own class (own
        prompt, own safety rails), so metering only the object the team
        returned would miss that spend entirely and silently defeat the
        per-match cap. Walk one level of attributes to find the real
        agents, and never count the same object twice."""
        found, seen_ids = [], set()
        stack = [(obj, depth)]
        while stack:
            o, d = stack.pop()
            if id(o) in seen_ids or d > 2:
                continue
            seen_ids.add(id(o))
            if getattr(o, "episode_usage", None) is not None:
                found.append(o)
            for v in list(vars(o).values()) if hasattr(o, "__dict__") else []:
                if hasattr(v, "__dict__"):
                    stack.append((v, d + 1))
                elif isinstance(v, (list, tuple)):
                    stack.extend((x, d + 1) for x in v if hasattr(x, "__dict__"))
        return found

    cost = None
    brains, seen = [], set()
    for top in list(agents) + list(managers.values()):
        for b in billable(top):
            if id(b) not in seen:
                seen.add(id(b))
                brains.append(b)
    for brain in brains:
        u = getattr(brain, "episode_usage", None) or {}
        tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        result.tokens_in += tin
        result.tokens_out += tout
        c = estimate_cost(getattr(brain, "name", ""), tin, tout,
                          u.get("cache_read_input_tokens", 0) or 0,
                          u.get("cache_creation_input_tokens", 0) or 0)
        if c is not None:
            cost = (cost or 0.0) + c
    result.est_cost_usd = round(cost, 4) if cost is not None else None
    if result.est_cost_usd is not None:
        print(f"  [cost] match est. ${result.est_cost_usd:.3f} "
              f"({result.tokens_in} in / {result.tokens_out} out tokens)")
    result.wall_time_s = round(time.time() - t_wall, 1)
    if log_dir:
        (log_dir / "match.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


class FootballScriptedAgent:
    """Role-switching baseline: nearest standing teammate attacks the ball
    (gets behind it, pushes it goalward); the other covers the goal mouth.
    Same interval-aware steering formulas the prompts teach."""

    def __init__(self, index: int, seed: int = 0):
        self.i = index
        self.name = "scripted"

    def begin_episode(self, log_dir=None):
        pass

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _steer(self, obs, tx, ty, fast):
        # approach stances must be reachable: clamp into the pitch interior
        # (an approach point computed inside a wall pins the ball instead)
        tx = float(np.clip(tx, -PITCH_X + 0.5, PITCH_X - 0.5))
        ty = float(np.clip(ty, -PITCH_Y + 0.5, PITCH_Y - 0.5))
        px, py = obs["self"]["position"]
        h = obs["self"]["heading_rad"]
        interval = max(obs.get("decision_interval_s", 0.5), 0.5)
        err = self._wrap(np.arctan2(ty - py, tx - px) - h)
        wz = float(np.clip(err / min(interval, 2.0), -1.0, 1.0))
        vx = 0.8 if (fast and abs(err) < 0.5) else (0.15 if abs(err) > 1.2 else 0.45)
        return {"vx": vx, "vy": 0.0, "wz": wz}

    def decide(self, obs):
        if obs["self"]["fallen"]:
            return {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        px, py = obs["self"]["position"]
        bx, by = obs["ball"]["position"]
        gx = obs["attacking_goal"]["x"]
        ogx = obs["defending_goal"]["x"]
        # scrum breaker: pushing something immovable -> disengage diagonally
        if obs["self"].get("blocked"):
            side = 1.0 if (self.i % 2 == 0) else -1.0
            return {"vx": -0.4, "vy": 0.5 * side, "wz": 0.0}
        my_d2 = (bx - px) ** 2 + (by - py) ** 2
        mate = obs["teammates"][0] if obs["teammates"] else None
        mate_d2 = (np.inf if (mate is None or mate["fallen"]) else
                   (bx - mate["position"][0]) ** 2 + (by - mate["position"][1]) ** 2)
        # overload: when the ball is deep in the opponent half, BOTH players
        # attack (2-v-1 on the keeper); otherwise nearest attacks, other holds
        attack_sign = 1.0 if gx > 0 else -1.0
        ball_deep = bx * attack_sign > 3.0
        if my_d2 <= mate_d2 or ball_deep:  # attacker
            # near the goal, shoot at the mouth corner farthest from the
            # nearest standing defender instead of dead center
            aim_y = 0.0
            if abs(gx - bx) < 2.8:
                keepers = [o for o in obs["opponents"] if not o["fallen"]
                           and abs(o["position"][0] - gx) < 2.2]
                if keepers:
                    ky = min(keepers, key=lambda o: abs(o["position"][1] - by))["position"][1]
                    aim_y = (GOAL_HALF_W - 0.35) * (-1.0 if ky > 0 else 1.0)
            # doorstep: ball already in the mouth band right at the line —
            # ram it straight through, no finesse
            if abs(gx - bx) < 1.2 and abs(by) < GOAL_HALF_W + 0.3:
                if my_d2 < 1.0 ** 2:
                    return self._steer(obs, gx + (0.5 if gx > 0 else -0.5), by,
                                       fast=True) | {"vx": 1.0}
            dirx, diry = gx - bx, aim_y - by
            n = float(np.hypot(dirx, diry)) or 1.0
            # flank bias: approach the ball slightly off-axis so two mirrored
            # attackers meet shoulder-to-shoulder, not head-on deadlocked
            flank = 0.35 if (self.i % 2 == 0) else -0.35
            tx = bx - dirx / n * (BALL_R + 0.45) - diry / n * flank
            ty = by - diry / n * (BALL_R + 0.45) + dirx / n * flank
            wrong_side = (px - bx) * (gx - bx) > 0 and abs(py - by) < 1.0
            if wrong_side:  # loop around, don't own-goal it
                ty = by + (1.4 if py >= by else -1.4)
                tx = bx - dirx / n * 0.2
                return self._steer(obs, tx, ty, fast=False)
            close = (px - tx) ** 2 + (py - ty) ** 2 < 0.35 ** 2
            if close:  # drive through the ball; full sprint on the doorstep
                cmd = self._steer(obs, bx + dirx / n, by + diry / n, fast=True)
                if abs(gx - bx) < 2.8 and abs(cmd["wz"]) < 0.4:
                    cmd["vx"] = 1.0  # shot burst (envelope max)
                return cmd
            return self._steer(obs, tx, ty, fast=True)
        # defender: hold the mouth, mirror the ball's y, clear if it comes near
        hx = ogx + (1.1 if ogx < 0 else -1.1)
        hy = float(np.clip(by, -(GOAL_HALF_W - 0.2), GOAL_HALF_W - 0.2))
        if my_d2 < 2.2 ** 2:
            dirx, diry = gx - bx, 0.0 - by
            n = float(np.hypot(dirx, diry)) or 1.0
            return self._steer(obs, bx + dirx / n * 0.3, by + diry / n * 0.3, fast=True)
        if (px - hx) ** 2 + (py - hy) ** 2 < 0.3 ** 2:
            face = self._wrap(np.arctan2(by - py, bx - px) - obs["self"]["heading_rad"])
            return {"vx": 0.0, "vy": 0.0, "wz": float(np.clip(face, -1.0, 1.0))}
        return self._steer(obs, hx, hy, fast=False)


def make_football_agent(spec: str, index: int, seed: int = 0,
                        prompt: str = "football_v1"):
    if spec == "scripted":
        return FootballScriptedAgent(index, seed=seed)
    from .agents import make_agent
    ag = make_agent(spec, seed=seed, prompt=prompt)
    if prompt != "football_v1":      # behaviour layer: free-form reply keys
        ag.reply_keys = None
    return ag


def make_football_manager(spec: str, seed: int = 0):
    if spec == "mock":
        return MockManager()
    from .agents import make_agent
    mgr = make_agent(spec, seed=seed, prompt="football_manager_v1")
    mgr.reply_keys = None  # free-form reply: accept the outermost JSON object
    return mgr


class MockManager:
    """Offline shakedown manager: canned tactics + touchline pacing."""
    name = "mock-manager"
    MSGS = ["Press high, force them wide!",
            "Drop deep, protect the middle!",
            "Both of you forward - we need a goal!",
            "Pin the ball in their corner and grind!"]

    def __init__(self):
        self._n = 0

    def begin_episode(self, log_dir=None):
        self._n = 0

    def decide(self, obs):
        self._n += 1
        return {"message": self.MSGS[(self._n - 1) % len(self.MSGS)],
                "move": {"vx": 0.4, "vy": 0.0,
                         "wz": 0.6 if self._n % 2 else -0.6}}
