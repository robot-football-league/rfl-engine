"""RFL 0.2 - the networked league: an authoritative game server.

Teams run WHEREVER THEY LIKE and connect over a WebSocket; the server owns
physics, rendering, rules, and the clock. A team receives observations (the
same published contract as in-process play, frames as base64 JPEG) and
returns velocity commands. Replies that arrive after the bridge deadline are
void — network misfortune is a missed decision, exactly like a real robot
dropping a packet. Team code can no longer touch simulator internals at all:
isolation is now physical, not procedural.

Wire protocol (JSON text frames, one team connection carries both players):
  server -> {"type":"hello","engine":...,"team_index":0|1,"settings":{...}}
  client -> {"type":"register","name":...,"code":...,"color":[r,g,b],
             "color_name":...}
  server -> {"type":"kickoff"}
  server -> {"type":"obs","player":0|1,"seq":N,"deadline_s":...,
             "obs":{... contract fields ..., "frames_jpeg":[b64, b64]}}
  client -> {"type":"cmd","player":0|1,"seq":N,
             "cmd":{"vx":...,"vy":...,"wz":...}}
  server -> {"type":"event","event":"goal"|...} (informational)
  server -> {"type":"full_time","fixture":{...}}
Reserved for 0.3: "mgr_obs"/"mgr_cmd" (networked managers).

Run:  python -m gauntlet rfl-serve --port 8800 --time 90 --out runs/net_match
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from pathlib import Path

import numpy as np

ENGINE_VERSION = "rfl-0.2"
REPLY_GRACE_S = 0.5  # wait deadline + grace before giving up on a reply


def _encode_obs(obs: dict) -> dict:
    """Wire form: numpy frames -> base64 JPEG, private keys dropped."""
    from PIL import Image
    out = {k: v for k, v in obs.items() if not k.startswith("_")}
    frames = obs.get("_frames")
    if frames is not None:
        enc = []
        for f in frames:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, format="JPEG", quality=80)
            enc.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        out["frames_jpeg"] = enc
    return out


class NetworkAgent:
    """Server-side proxy: decide() forwards over the team's WebSocket and
    blocks (in the engine's decider thread) until the reply or deadline."""

    def __init__(self, hub, team_index: int, player: int, deadline_s: float):
        self.hub = hub
        self.team_index = team_index
        self.player = player
        self.deadline_s = deadline_s
        self.name = f"net:t{team_index}p{player}"
        self.episode_usage = {}

    def begin_episode(self, log_dir=None):
        pass

    def decide(self, obs: dict):
        fut = asyncio.run_coroutine_threadsafe(
            self.hub.request(self.team_index, self.player, _encode_obs(obs),
                             self.deadline_s),
            self.hub.loop)
        try:
            return fut.result(timeout=self.deadline_s + REPLY_GRACE_S)
        except Exception:
            fut.cancel()
            return None  # engine treats as ignored_invalid -> command held


class TeamHub:
    """Owns the two team WebSocket connections and request/reply matching."""

    def __init__(self, settings: dict, tokens=None):
        self.settings = settings
        self.tokens = tokens  # optional [tokA, tokB]
        self.loop: asyncio.AbstractEventLoop | None = None
        self.conns: dict[int, object] = {}
        self.cards: dict[int, dict] = {}
        self.pending: dict[tuple, asyncio.Future] = {}
        self.seq = 0
        self.registered = asyncio.Event()

    async def handler(self, ws):
        team_index = None
        try:
            if self.tokens:
                token = (ws.request.path.split("token=")[-1]
                         if "token=" in ws.request.path else "")
                if token not in self.tokens:
                    await ws.close(code=4003, reason="bad token")
                    return
                team_index = self.tokens.index(token)
                if team_index in self.conns:
                    await ws.close(code=4001, reason="token already connected")
                    return
            else:
                free = [t for t in (0, 1) if t not in self.conns]
                if not free:
                    await ws.close(code=4001, reason="match full")
                    return
                team_index = free[0]
            # claim the slot BEFORE any await: two simultaneous connections
            # must not both see the same free slot (measured: they did)
            self.conns[team_index] = ws
            await ws.send(json.dumps({
                "type": "hello", "engine": ENGINE_VERSION,
                "team_index": team_index, "settings": self.settings}))
            raw = json.loads(await ws.recv())
            if raw.get("type") != "register":
                await ws.close(code=4002, reason="expected register")
                return
            self.cards[team_index] = raw
            print(f"[server] team {team_index} registered: "
                  f"{raw.get('name')} ({raw.get('code')})")
            if len(self.cards) == 2:
                self.registered.set()
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") == "cmd":
                    key = (team_index, int(m.get("player", -1)), int(m.get("seq", -1)))
                    fut = self.pending.pop(key, None)
                    if fut is not None and not fut.done():
                        fut.set_result(m.get("cmd"))
        except Exception as e:
            print(f"[server] connection error: {type(e).__name__}: {e}")
        finally:
            for t, c in list(self.conns.items()):
                if c is ws:
                    del self.conns[t]

    async def request(self, team_index: int, player: int, obs_wire: dict,
                      deadline_s: float):
        ws = self.conns.get(team_index)
        if ws is None:
            return None
        self.seq += 1
        seq = self.seq
        fut = self.loop.create_future()
        self.pending[(team_index, player, seq)] = fut
        await ws.send(json.dumps({
            "type": "obs", "player": player, "seq": seq,
            "deadline_s": deadline_s, "obs": obs_wire}))
        try:
            return await asyncio.wait_for(fut, timeout=deadline_s + REPLY_GRACE_S)
        except asyncio.TimeoutError:
            self.pending.pop((team_index, player, seq), None)
            return None

    async def broadcast(self, payload: dict):
        for ws in list(self.conns.values()):
            try:
                await ws.send(json.dumps(payload))
            except Exception:
                pass


def serve_match(port: int = 8800, match_time_s: float = 90.0,
                decision_deadline_s: float = 3.0, request_period_s: float = 2.0,
                video_path=None, log_dir=None, tokens=None):
    import websockets

    from .football import run_match

    settings = {"match_time_s": match_time_s,
                "decision_deadline_s": decision_deadline_s,
                "request_period_s": request_period_s,
                "obs_mode": "camera", "players_per_team": 2}
    hub = TeamHub(settings, tokens=tokens)
    result_box = {}

    async def main():
        hub.loop = asyncio.get_running_loop()
        async with websockets.serve(hub.handler, "0.0.0.0", port,
                                    max_size=8 * 1024 * 1024):
            print(f"[server] RFL {ENGINE_VERSION} listening on :{port} - "
                  f"waiting for two teams...")
            await hub.registered.wait()
            a, b = hub.cards[0], hub.cards[1]

            def color(card, fallback):
                c = card.get("color") or fallback
                return tuple([*[float(v) for v in c[:3]], 1.0])

            agents = [NetworkAgent(hub, t, p, decision_deadline_s)
                      for t in (0, 1) for p in (0, 1)]
            await hub.broadcast({"type": "kickoff"})

            def run():
                result_box["result"] = run_match(
                    agents, match_time_s=match_time_s, mode="realtime",
                    decision_deadline_s=decision_deadline_s,
                    request_period_s=request_period_s, obs_mode="camera",
                    team_colors=(color(a, [0.15, 0.4, 0.95]),
                                 color(b, [0.95, 0.3, 0.15])),
                    team_color_names=(a.get("color_name", "home"),
                                      b.get("color_name", "away")),
                    team_names=(a.get("name", "Home"), b.get("name", "Away")),
                    team_codes=(a.get("code", "HOM"), b.get("code", "AWY")),
                    video_path=video_path, log_dir=log_dir)

            th = threading.Thread(target=run, daemon=True)
            th.start()
            last_score = [0, 0]
            while th.is_alive():
                await asyncio.sleep(0.5)
                r = result_box.get("result")
            r = result_box["result"]
            fixture = {
                "engine": ENGINE_VERSION,
                "home": {"team": a.get("name"), "code": a.get("code"),
                         "goals": r.score[0]},
                "away": {"team": b.get("name"), "code": b.get("code"),
                         "goals": r.score[1]},
                "winner": (a.get("name") if r.winner == "A"
                           else b.get("name") if r.winner == "B" else "draw"),
                "goals": r.goals,
            }
            await hub.broadcast({"type": "full_time", "fixture": fixture})
            print(f"FULL TIME: {fixture['home']['code']} {r.score[0]} - "
                  f"{r.score[1]} {fixture['away']['code']} ({fixture['winner']})")
            if log_dir:
                (Path(log_dir) / "fixture.json").write_text(
                    json.dumps(fixture, indent=2))
            await asyncio.sleep(0.5)

    asyncio.run(main())
    return result_box.get("result")
