# RFL Engine

The match engine of the **Robot Football League (RFL)** — 2v2 simulated
Unitree G1 humanoids playing football in MuJoCo, with LLM brains, produced
as a TV-style broadcast.

**Watch the league live:** https://twitch.tv/rfl_robot_football_league

## What this is

- A MuJoCo simulation of a walled 14 x 9 m pitch, two G1 robots per side,
  driven by a pretrained 12-DoF walking policy at 500 Hz.
- An onboard-software stack mirroring real humanoid-football practice
  (RoboCup stacks, Unitree's G1 competition SDK): camera detections in
  metres, a world model, A* navigation, and motion skills.
- A behaviour API: teams receive observations and reply with skills
  (`go_to_ball`, `kick_toward`, `walk_to`, `turn_to`, `hold`) plus a fully
  public natural-language radio to their teammate.
- A realtime match runner with fixed-length halves, goal replays, a
  broadcast overlay, and complete machine-readable match logs.

The engine, physics, and low-level walking are fixed and identical for
everyone. **A team supplies only decision-making.** The full participant
contract is in [docs/RFL_RULES.md](docs/RFL_RULES.md).

## Quickstart (no API keys needed)

Requires Python 3.10+ and `ffmpeg` on PATH.

```bash
git clone https://github.com/robot-football-league/rfl-engine
cd rfl-engine
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m gauntlet rfl examples/mock_united examples/mock_united \
    --time 60 --video match.mp4 --out runs/demo
```

That plays a 60-second exhibition between two copies of a scripted mock
team and writes the broadcast video to `match.mp4` plus full logs
(`match.json`, `decisions.jsonl`, `comms.jsonl`, `telemetry.jsonl`) to
`runs/demo/`.

## Building a real team

Start from the reference team — clone it, point the config at a model you
have keys for, and iterate:

- **Sample team:** https://github.com/robot-football-league/rfl-sample-team
- **League data (results, tables, every match's logs):**
  https://github.com/robot-football-league/rfl-league-data

LLM-backed players use the bundled adapter (`pip install -e ".[llm]"` for
the Anthropic / OpenAI / Google SDKs) with keys from the environment
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ZHIPU_API_KEY`),
or run entirely local models via Ollama. Teams may also ignore the adapter
and implement `decide()` any way they like — the only contract is the
observation/reply schema in the rules.

## Networked matches

For official fixtures the engine runs as a game server and each team
connects over a WebSocket from its own environment — your compute, your
keys, your code, never shared with the host:

```bash
python -m gauntlet rfl-serve --port 8800 --time 600 --halves 2 \
    --video match.mp4 --out runs/match
```

See `rfl_client.py` in the sample-team repo for the single-file client SDK.

## The realism law

Players perceive only what a real robot on a real pitch could: their
camera and the radio. Reaching into simulator internals from team code is
cheating; match logs are published and audited. Hair is cosmetic.

## Acknowledgements

The G1 robot description and the pretrained walking policy are from
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
(BSD-3-Clause, © Unitree Robotics) — vendored subset under
`third_party/unitree_rl_gym/` with license preserved. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT — see [LICENSE](LICENSE).
