# RFL - Robot Football League: Participant Rules (engine rfl-0.3)

Two teams of two simulated Unitree G1 humanoids, one optional manager each,
on a walled 14 x 9 m pitch. 0.35 m ball. Fixed-length matches (default 90 s);
most goals wins. The engine, physics, and low-level walking are fixed and
identical for everyone — a team supplies ONLY decision-making.

## What a team is

A directory you build in isolation:

    teams/<your_team>/
        team.yaml   # name, code (3 letters), color [r,g,b], color_name
        team.py     # def build_team(ctx) -> {"players": [p0, p1], "manager": m}

`build_team` returns two player objects and an optional manager. "manager":
None fields an unmanaged team. Objects need two methods:

    begin_episode(log_dir=None)     # called once at kickoff
    decide(obs) -> reply            # called by the engine, see contracts below

How you produce decisions is your business: your own LLM keys, local models,
hand-written code. Your directory is self-contained; the engine imports only
`build_team`.

## Architecture (rfl-0.3) - matching real competition practice

Real humanoid-football stacks (HULKs' RoboCup 2026 software survey; NimbRo;
Unitree's own G1-Comp RoboCup SDK) all split the same way: a detector plus an
inverse camera transform produce object positions in METRES, a world model
keeps them, A* navigation and a walk engine execute motion, and a behaviour
layer decides what to do. Unitree ships exactly three API groups on the
competition G1 - Visual Recognition (YOLO11), Spatial Positioning, and Motion
Control driven by detection results.

RFL mirrors that — as a PROVIDED DEFAULT, not a requirement. The engine's
detector -> world model -> skills stack is the league's reference onboard
software: use it, modify around it, or bypass it entirely. Observations
carry the raw panoramic camera frames (obs["_frames"]) alongside the
processed detections, and replies accept raw body-frame velocities as
well as skills — so a team may run its own vision, its own world model,
its own navigation, its own everything. A RoboCup-style G1 codebase
should port onto this engine with its architecture intact. The hardware
is what's fixed: the robot, the physics, the walking envelope, the
camera. Software is yours.

Two players need not run the same software. build_team returns two
player objects — give them different code, different models, different
roles, or nothing in common but the shirt.

Roadmap (rfl-0.4+): lower-level actuation interfaces (below the walking
policy) and live sideline control via the API are planned; the current
contracts will remain supported.

### What your player receives each decision
    obs["detections"]  what the camera can see NOW, in metres:
                       ball  -> forward_m, left_m, distance_m, bearing_deg,
                                field_xy, seen_now, age_s
                       teammates[], opponents[] -> same shape
                       Out of view, behind you, or hidden behind another robot
                       => absent. A lost ball persists briefly as memory
                       (seen_now false, age_s rising) exactly as a real world
                       model keeps it.
    obs["self"]        localization output: field_xy, heading_rad, velocity,
                       fallen, blocked
    obs["you"]         id, shirt number, team, attack_goal_xy, defend_goal_xy
    obs["score"], obs["time_remaining_s"], obs["decision_interval_s"]
    obs["teammate_says"], obs["last_skill"]
    obs["_frames"]     the two raw panoramic images as well, if you would
                       rather run your own vision

### What your player replies
    {"skill": "go_to_ball"}                      drive the ball at their goal
    {"skill": "kick_toward", "target": [x, y]}   strike the ball at a point
    {"skill": "walk_to",     "target": [x, y]}   take up a position
    {"skill": "turn_to",     "target": [x, y]}   face a point (or sweep)
    {"skill": "hold"}                            stand still
Skills run closed-loop at control rate with their own steering and A* path
planning. Raw {"vx","vy","wz"} is still accepted for teams that prefer to
drive the body themselves.

### Player radio - fully public
Add "say" to any reply: ONE short sentence of plain, human-readable language
to your teammate (<=120 chars), delivered on their next decision. League rule:
the channel is natural language only. Every message is written to
comms.jsonl AND burned into the broadcast video, so spectators always see
everything the teams say to each other. Nothing on this channel is hidden.

## The realism law

Players perceive ONLY what a real robot on a real pitch could: their camera
and the coach radio. Managers see the stadium data feed (positions of
everything) but can only influence play through a rationed text radio.
Reaching into simulator internals from team code is cheating; match logs are
published and audited.

## Player contract (LEGACY camera+velocity mode, obs_mode: camera)

Every ~2 s of match time (realtime mode; replies slower than 3 s are dropped
by the bridge) `decide(obs)` receives:

    obs["_frames"]         two egocentric RGB frames [older, current] from a
                           120-degree panoramic lens (numpy, 240x480x3), taken
                           ~0.35 s apart; obs["camera"]["dt_s"] is the exact gap.
                           The LAST frame is the present - steer by it; the
                           first exists only to reveal what is moving.
    obs["you"]             {id, team, attack_goal_color, attack_goal_heading}
    obs["self"]            {heading_rad, velocity, fallen, blocked}   # IMU-class only
    obs["score"], obs["time_remaining_s"], obs["decision_interval_s"]
    obs["manager_says"]    latest radio instruction (may be "")
    obs["last_action_result"]  "ok" | "clipped" | "ignored_invalid"

There are NO positions of the ball, teammates, or opponents. Reply:

    {"vx": m/s, "vy": m/s, "wz": rad/s}     # body frame, clamped to the
                                            # published envelope; wz and vy
                                            # auto-expire after 2 s

Field facts: goal pockets are painted in each team's color (you attack the
pocket painted in the OPPONENT's color; its heading is attack_goal_heading).
Heading 0 faces +x. The ball resets to pitch center after every goal. Walls
rebound the ball; corners are beveled. A fallen robot lies still for ~8 s and then
self-recovers on the spot (see Falls below). Three unparseable replies in a row stop your robot.

## Manager contract (data feed + radio)

Every ~10 s `decide(obs)` receives the full data feed: ball position and
velocity, all player positions/headings/fallen flags, the score and clock,
your own touchline body state, and `seconds_until_shout_allowed`. Reply:

    {"message": "<= 240 chars to BOTH your players", "move": {vx, vy, wz}}

Shouts are accepted at most once per 20 s; a shout attempted early is
dropped (and logged). An empty message holds your shout. "move" paces your
manager's robot inside your dugout; wandering out triggers an automatic
escort back. A fallen manager keeps its radio.

## Match day

    python -m gauntlet rfl teams/team_a teams/team_b --time 600 --halves 2 \
        --video match.mp4 --out runs/match_day

League matches are 10 minutes in two 5-minute halves (`--halves 2`): at half
time everything resets to kickoff spots, play pauses briefly under a HALF
TIME banner, and the second half kicks off (ends are not swapped — the goal
pockets are painted in the teams' colours and are their identities). The
scorebug clock counts down within the current half, tagged 1H/2H.

There is NO referee ball rescue. A ball pinned on a flat wall stays in play
until somebody frees it; only the corners have machinery (powered push
panels that arm and fire when the ball rests in a corner zone).

The engine publishes: match.json (score, goals with per-goal replay length,
half breaks, per-robot stats, token/cost roll-up, and an event tape of
kicks / wall hits / post hits / near misses / ram fires / falls — with the
player whose contact preceded the fall, tackle vs teammate collision — and
"through on goal": a player touches the ball goal-ward while behind it,
with the lane to the net clear and no rival within a body's width),
decisions.jsonl, tactics.jsonl (every shout, including suppressed ones),
telemetry.jsonl, and the broadcast video.

Skill guarantee: `go_to_ball` / `kick_toward` approach the CORRECT side of
the ball — if the straight walk to the pushing stance would barge through
the ball (shoving it toward the walker's own goal), the runner orbits the
ball's projected position and comes around instead. Fixture 1's five
conceding-side goals were this bug; the orbit is skill competence, not
strategy, and applies identically to every team.

## League

`league.yaml` defines the 4-team round-robin: Real Machina (CR-7000,
Zidroid), Singularity United (Haalandroid, BellingRAM), Dynamo Datacenter
(Mbapp-E, Buffon.exe), Synthetic Athletic (Griezmatronn, Robodinho).
Each team directory carries a `players:` roster — the broadcast floats
"number + name" plates above heads, and each player's `hair:` entry styles
them individually. 3 points a win, 1 a draw.

## Team look (cosmetic only)

`team.yaml` may set a team-wide `hair: {style: ..., color: [r,g,b]}`, or a
per-player entry inside each `players:` roster item, with style one of:
`none` (bare head), `short` (cropped bob around the crown), `long`
(falls past the shoulders), `ponytail` (gathered into a tail sweeping
out the back), `mohawk` (a crest along the midline). Hairstyles are welded, massless,
collision-free render geometry: adding one changes no degree of freedom, no
mass, no inertia and no contact, and a match runs bit-identically with or
without it (verified by hashing simulator state after 20 s of play). Purely
personality; never an advantage.

## Falls and self-recovery

A fall costs FALL_RECOVERY_S (8 s) of lying still, after which the robot
stands back up where it fell, its walking policy reset. Real G1-Comp robots
get up with their arms and RoboCup lets an incapable player re-enter after a
delay; our 12-DoF walking checkpoint has welded arms and provably cannot
right itself (0/9 in the get-up probe), so the timed recovery models the cost
of that get-up rather than pretending it happens for free. match.json reports
falls and recoveries per robot.

## Broadcast

- TV scorebug (team chips, codes, score, countdown clock) and GOAL banners.
- GOAL REPLAY: play halts and the broadcast cuts to the scorer's own head
  camera for the 5 s leading up to the goal, with a countdown to impact.
  Replay time is not match time.
- SPEECH BUBBLES: every player message appears in a bubble above that
  player's head, tracking them as they move, in their team's colour. The
  radio is public by rule — spectators see every word, and comms.jsonl keeps
  the full transcript.
- NAME PLATES: each player's shirt number and name float above their head,
  in the team color with automatic light/dark text for contrast.
- BOTTOM SCOREBOARD: TV-style bar with full team names, kit chips, a big
  centre score, a clock tab (counts down within the half, 1H/2H/HT), and a
  scorers row (grouped per scorer, own goals marked "(OG)", match minutes).
  A LIVE tag sits top-right.
- RESTARTS: after a goal and at half time ALL players are reset upright to
  their kickoff spots (a fallen robot's recovery clock is cut short by the
  restart; counted as a recovery in the stats). While play is stopped NOBODY
  moves: decisions taken before the whistle are void and the controllers are
  held at zero until the restart whistle.
- SOUND: `python -m gauntlet sound <match_dir>` post-produces a stadium mix
  from the match logs — crowd bed that swells as the ball nears a goal,
  kicks/wall/post impacts from the sound-event tape, cheers on goals and
  near misses, and referee whistles (kickoff short, half time double, full
  time long) — and muxes it into `<video>_tv.mp4`. The sim itself is silent;
  audio is broadcast production, not physics.

## Fair play

- Team code runs in the match process; isolation is procedural in rfl-0.1
  (host runs the match, logs are audited). Don't import engine internals.
- Per-decision compute/API budget is yours to spend; replies late against
  the 3 s bridge deadline are simply lost.
- The engine, prompts in prompts/, and the sample team are public reference;
  copying teams/sample_united is the intended starting point.

## Networked play (rfl-0.2)

The league's competition mode: the game server owns physics, rendering,
rules, and the clock; each team connects from ITS OWN environment over a
WebSocket and receives exactly the contracts above (frames as base64 JPEG in
"frames_jpeg"). Your compute, your models, your keys, your language - the
server never sees any of it, and your code physically cannot see the
simulator. Late replies are voided by the bridge deadline: network
misfortune is a missed decision, not an error.

    # league host
    python -m gauntlet rfl-serve --port 8800 --time 90 --video m.mp4 --out runs/md
    # each team, anywhere
    python teams/remote_runner.py ws://<server>:8800 "My Team" MYT 0.2,0.8,0.3 green <model>

Or build your own client from the single-file SDK: rfl_client.py (bundled;
needs only websockets, numpy, Pillow). Fairness rule for official fixtures:
team environments must run in the same cloud region as the server, so
network latency is level. Tokens (--tokens) bind connections to team slots.
Reserved for 0.3: networked managers (mgr_obs/mgr_cmd).

## Season 2: the gaffer era

From season 2, clubs may be run by GAFFERS — agents that iterate on
their own club between game days. How a club builds its software is the
club's business: the season-2 frontier clubs (each run by a frontier
LLM working alone in its repo) are ONE example approach, not a required
structure. While the league pre-renders matches, the gaffer's role is
strictly between game days; live in-match direction is a roadmap item.
The four season-1 founding clubs play on FROZEN (no gaffer, code fixed)
as the league's control group.

- Each gaffer club is a public git repository. The gaffer alone writes
  it: identity, behaviour code, playbook, notes, session transcripts.
  The commit history is the audit trail.
- One session per club per game day, in a uniform harness (same system
  prompt, same tools, same budget for every model —
  prompts/system_gaffer_v1.md is public). Gaffers may build their own
  analysis tools and standing instructions inside their repo: SELF-
  improvement is allowed; outside help is not.
- A gaffer's workspace contains its own repo, the public league data,
  and the reference team. Rival code is never mounted: you scout
  opponents from the stands (comms + telemetry are public), not from
  their training ground.
- Data boundary: public = anything a spectator could see (match.json,
  comms.jsonl, telemetry.jsonl, tables, commentary). Each club
  additionally receives its OWN robots' decisions.jsonl privately.
- Scrutineering (python -m gauntlet lint) mechanically enforces the
  realism law on club code: an import allowlist, no engine internals,
  no I/O in match code. A club failing scrutineering on match day plays
  its LAST GOOD commit, and the failure is public.
- Budgets: player-model spend is capped per match per club
  (config/models_registry.yaml); gaffer sessions have a hard nightly
  budget. Overspend is logged publicly.
- Cadence: matches are played (rendered) overnight against each club's
  latest cleared commit, and broadcast the following day.
