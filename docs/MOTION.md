# MOTION.md — joint states and actions

Every match writes `motion.npz` and `motion.json` next to its pose track, and
the 4DGSX exporter copies both into the bundle. They are the *states and
actions* half of the archive: `track.bin` says where each body was, which can
be watched; these say what each joint was doing and what was commanded of it,
which can be trained on.

Written by `football.run_match` whenever the pose track is (`record_states`,
which the league sets for every fixture). `RFL_SKIP_MOTION=1` turns it off.
Copied into the bundle by `volumetric.export_motion`.

## The arrays — `motion.npz`

Float32 throughout. `nrobots` is 4, in fixed order r0, r1 (team A), r2, r3
(team B); managers are not recorded, because they are not playing.

| array | shape | what it is |
|---|---|---|
| `t` | `[nframes]` | match seconds — the same clock as `states.npz`, `track.bin` and `hud.json`, so a goal at t=157.1 indexes straight in |
| `qpos` | `[nframes, nrobots, 19]` | free joint (3 translation + 4 quaternion `wxyz`, MuJoCo order) then the 12 hinges |
| `qvel` | `[nframes, nrobots, 18]` | free joint (3 linear world-frame + 3 angular body-frame) then the 12 hinge velocities |
| `ctrl` | `[nframes, nrobots, 12]` | the actuator command applied at that step — a **torque**, N·m |
| `target_q` | `[nframes, nrobots, 12]` | the locomotion policy's joint position targets, rad — **the action half** |
| `action` | `[nframes, nrobots, 12]` | the policy's raw output; `target_q = action * action_scale + default_angles` |
| `cmd` | `[nframes, nrobots, 3]` | `(vx, vy, wz)` base velocity command — the football decision |
| `ball_qpos` | `[nframes, 7]` | the ball's free joint |
| `ball_qvel` | `[nframes, 6]` | the ball's velocity |
| `hz` | scalar | control rate, 50.0 |

The ball is in there because without it these are not states of the *game*:
four humanoids and no ball is a walking dataset.

## Which array is the action

`ctrl` is a **torque**, not a position target. Everything under the hood is
one frozen PD loop:

    tau = (target_q - q) * kp + (0 - dq) * kd

recomputed every physics step (500 Hz) from a `target_q` the policy refreshes
every 10th step (50 Hz). `motion.json` carries `kps`, `kds`,
`default_angles`, `action_scale` and `decimation`, so:

- **to train on actions, use `target_q`** — it is the policy's decision, one
  value per control period, held across it;
- **`ctrl` is derived** — the row stored is the torque on the *first* physics
  step of each control period, not a constant held across it. Reproduce the
  full 500 Hz torque by applying the law above to `target_q`. Checked on a
  real match: reconstruction matches the recorded `ctrl` to 1.2e-5 N·m.
- **`cmd` is the only action in the file that is nobody's motor controller.**
  The locomotion policy is frozen, identical for all four robots and
  unchanged all season; the *club* differs only in the `(vx, vy, wz)` it asks
  for. If you want the part of this dataset that is football rather than
  locomotion, it is `cmd`.

## Rates, clock, ordering

- **Control rate 50 Hz, constant for the whole match** — `t` steps by exactly
  0.02 s from 0.0, through kickoffs, freezes, goals and half time. Physics is
  500 Hz (`simulation_dt` 0.002, `control_decimation` 10). Sampling is on the
  physics step index, so no game state can perturb the cadence.
- Not resampled to `track.bin`'s 25 Hz. Decimate if you want less; the steps
  between two samples cannot be invented afterwards.
- **Joint and actuator order is stable** — it is `g1_policy.JOINT_ORDER`, a
  constant in the engine, and the actuators carry their joints' names. It
  would only move if the robot model changed, and `motion.json` names both
  orders per bundle anyway, so read them rather than hardcoding.
- Robot `rN` in this file is the same robot as the `rN_*` bodies in
  `scene.json` (`body_prefix`).

## The robot

`unitree_g1_12dof` — 12 leg joints, not the 29-DoF G1. Unitree's own model,
BSD-3-Clause, used unmodified; `motion.json` carries its sha256. It is
already published: `rfl-engine` vendors `g1_12dof.xml`, the meshes it
references, the pretrained policy and the deploy config, and
`football.build_football_model` (also published) builds the pitch, walls,
goals and ball around it. The kinematics are fully reconstructible from
public files.

## Where it goes, and what it does not touch

Bundle root, beside `track.bin`. **`scene.json` is deliberately not
changed**: no player fetches these files, the spec's must-ignore rule covers
files a reader does not know, and a bundle carrying motion plays exactly as
it did before. Publishing uploads whatever is in the bundle directory.

Not included, and not planned: camera frames (the bundle already renders),
rewards or shaping terms, and anything about how a club's agent is
implemented. States and actions only.

## Caveats worth knowing before you build on it

- **Only matches rendered after this landed have it.** There is nothing to
  back-fill from: the pose track is 25 Hz link poses, and the actions were
  never written down.
- **A published bundle cannot be topped up.** `publish.bundle_fingerprint`
  hashes `scene.json` + `hud.json` only, so re-publishing an existing bundle
  with motion added reads as "unchanged" and uploads nothing. The files have
  to be there at first publish.
- **The local copy is reclaimed.** `reclaim_aired.sh` deletes `motion.npz`
  from a match directory once the match has aired and its bundle is
  published, exactly as it does `states.npz` and `web/` — ~33 MB per match,
  against a season of 90. The published bundle is the archive copy;
  `motion.json` stays behind either way.
