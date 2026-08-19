# Sim-to-real contract

The benchmark is designed so that, in principle, the same test can be
recreated with physical Unitree G1 units. This document is the contract: the
interfaces that must match between the simulated and physical versions for
results to be comparable. Everything the decision layer sees and does crosses
one of these interfaces; nothing else about the harness is load-bearing.

## 1. Command interface

| Property | Value | Real-robot counterpart |
|---|---|---|
| Schema | `{"vx": m/s, "vy": m/s, "wz": rad/s}`, body frame | Unitree SDK high-level locomotion command (vx, vy, vyaw) — the stock G1 interface |
| Envelope | vx ∈ [−0.8, 1.0], vy ∈ [−0.5, 0.5], wz ∈ [−1.0, 1.0] (clamped) | Re-verify on hardware with the M0-equivalent test; tighten if needed |
| Cadence | ≤ 2 Hz (harness cap); commands blended over 0.2 s and held until replaced | Identical — send at the same cap, let the onboard controller smooth |
| Command timeout | wz and vy expire 2 s after each command (vx persists) — bounds any stale rotation | Identical watchdog in the bridge process (standard command-timeout practice) |
| Invalid reply | previous command held; 3 consecutive → zero velocity | Identical watchdog in the bridge process |

## 2. Low-level controller

Two ways to close the loop on hardware, in order of increasing fidelity:

- **Option A — stock Unitree controller** (recommended first). Drive the
  robot through the official SDK's velocity interface. Easiest and most
  robust; the gait differs from the simulated policy, which shifts absolute
  completion times but not the comparison between decision layers (identical
  for all entrants).
- **Option B — deploy the simulated policy.** The vendored policy
  (`motion.pt`) ships with a real-robot deployment path in the same repo
  (`unitree_rl_gym` → `deploy/deploy_real`, G1). Same controller in sim and
  real closes the loop tighter, at the cost of setup and tuning risk.

Either way, the controller is fixed hardware from the benchmark's point of
view: identical for every competitor, never modified per entrant.

## 3. Timing

- **`realtime` mode is the competition standard** — the sim advances at wall
  clock and never waits, the robot walks on its last command while the agent
  thinks, and replies steer the robot wherever it is on arrival. A physical
  robot behaves this way by nature; no bridge logic is needed beyond
  latest-command-wins.
- **`paused` mode is a sim-only ablation** (judgment with latency removed).
  It cannot exist in reality and must not be used for claims about
  deployability.
- The decision brain's placement (onboard computer, LAN machine, or cloud
  API over Wi-Fi) is part of the entrant, and its latency is part of the
  result. The harness logs per-decision latency and effective decision rate
  in both worlds.

## 4. Observation interface

The JSON schema (see README / `gauntlet/episode.py:build_observation`) is
deliberately silent about *how* state is known. Real-world sources:

| Field | Real source |
|---|---|
| `self.position`, `heading_rad`, `velocity` | Motion capture (recommended: mm accuracy, 100+ Hz) or the G1's onboard lidar SLAM (decimeter class — degrades the contract; prefer mocap for benchmark runs) |
| static `obstacles` AABBs | Surveyed once after course build (tape measure or mocap wand) |
| moving obstacle AABB + velocity | A motorized cart or track dolly driven on the scripted sinusoid, tracked by mocap; velocity from finite difference |
| checkpoint trigger | Computed from mocap pose with the same 1.5 m radius / not-fallen rule |
| fall detection | Same rule (torso height < 0.4 m or tilt > 60° for 0.5 s) from mocap pose |
| `self.blocked` | Same rule from mocap displacement: commanded vx > 0.3 m/s while net planar displacement < 0.12 m/s over a 1 s window (instantaneous base velocity is unusable — a humanoid stepping in place against an obstacle oscillates ±0.3 m/s) |

Accuracy budget: positions good to ±5 cm and fresh to within 100 ms preserve
the contract; both are easy for mocap and marginal for SLAM.

## 5. Course

The course JSON is the ground truth in both worlds. Physical build: foam or
cardboard obstacles placed to the JSON footprints (±10 cm), taped checkpoint
circles, a cart on a marked path for the mover. `arena_half` is a config
knob — a physical recreation will likely scale to ~10 × 10 m to fit a mocap
volume; regenerate courses at that scale and re-run the sim baselines at the
same scale so both worlds use identical course sets.

## 6. Scoring

Identical by construction once the observation contract holds: score =
checkpoints × 100 − seconds, full budget charged unless completed, bumps
logged (real world: from mocap proximity + an observer, since contact sensing
isn't in the contract).

## 7. Honest deltas that remain

- Contact dynamics, floor friction, and battery state differ from sim; they
  are shared by all entrants, so comparative results transfer even where
  absolute numbers shift.
- Falls cost real money: physical runs need a spotter or gantry, foam
  flooring, an e-stop, a geofence margin inside the mocap volume, and far
  fewer episodes than sim (pick a subset of seeds).
- **No robot-vs-robot contact formats on hardware** until the simulated
  contact study (V2-M1) says collisions are survivable, and even then with
  protective gear and low speeds.
- Network jitter for cloud-brained entrants is real and variable; report
  latency distributions alongside scores.
