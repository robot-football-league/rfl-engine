"""RFL - the Robot Football League.

The league separation: this module is the ENGINE side. A TEAM is a directory
a participant builds in isolation:

    teams/<your_team>/
        team.yaml    # name, code, color, anything your own code wants
        team.py      # def build_team(ctx) -> {"players": [p0, p1],
                     #                          "manager": m_or_None}

Player objects implement begin_episode(log_dir=None) and
decide(obs) -> {"vx","vy","wz"}; managers decide(obs) -> {"message","move"}.
The observation and reply contracts are published in docs/RFL_RULES.md and
are the ONLY interface between a team and the match — players see camera
frames and the coach radio, managers see the data feed. How a team produces
decisions (LLM calls with their own keys, local models, plain code) is
entirely its own business.

Match day: `python -m gauntlet rfl teams/alpha teams/beta`.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ENGINE_VERSION = "rfl-0.1"


@dataclass
class Team:
    path: Path
    name: str
    code: str          # 3-letter scorebug code
    color: tuple       # RGBA 0..1
    color_name: str    # what the pocket paint is called over the radio
    hair: object       # team look: dict, or per-player list of dicts
    players: list      # [{"name": ..., "hair": {...}}, ...] roster (optional)
    module: object

    def build(self, ctx) -> dict:
        squad = self.module.build_team(ctx)
        players = squad.get("players") or []
        if len(players) != 2:
            raise ValueError(f"{self.name}: build_team must return exactly 2 players")
        return squad


def load_team(path: str | Path) -> Team:
    path = Path(path)
    cfg = yaml.safe_load((path / "team.yaml").read_text())
    spec = importlib.util.spec_from_file_location(
        f"rfl_team_{path.name}", path / "team.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_team"):
        raise ValueError(f"{path}/team.py must define build_team(ctx)")
    home = cfg.get("kit_home") or {}
    away = cfg.get("kit_away") or {}
    color = home.get("color") or cfg.get("color", [0.5, 0.5, 0.5])
    if len(color) == 3:
        color = [*color, 1.0]
    team_kit_away = None
    if away.get("color"):
        ac = list(away["color"])
        if len(ac) == 3:
            ac = [*ac, 1.0]
        team_kit_away = (tuple(float(v) for v in ac),
                         away.get("color_name", "away colors"))
    t = Team(path=path, name=cfg["name"], code=cfg.get("code", cfg["name"][:3].upper()),
                color=tuple(float(v) for v in color),
                color_name=home.get("color_name") or cfg.get("color_name", "colored"),
                hair=cfg.get("hair") or {},
                players=cfg.get("players") or [],
                module=module)
    t.kit_away = team_kit_away          # (rgba, name) or None
    return t


def run_rfl_match(team_a_dir, team_b_dir, match_time_s: float = 90.0,
                  mode: str = "realtime", halves: int = 1,
                  record_states: bool | None = None,
                  video_path=None, log_dir=None):
    from .football import run_match
    a, b = load_team(team_a_dir), load_team(team_b_dir)
    clash = sum((x - y) ** 2 for x, y in
                zip(a.color[:3], b.color[:3])) < 0.18
    b_kit_png = "kit_home.png"
    if clash and getattr(b, "kit_away", None):
        b.color, b.color_name = b.kit_away
        b_kit_png = "kit_away.png"
        print(f"  [kits] color clash: {b.name} wear their away kit "
              f"({b.color_name})")
    kit_textures, badges = {}, {}
    for tm, (team, png) in enumerate(((a, "kit_home.png"), (b, b_kit_png))):
        f = team.path / "identity" / png
        if f.exists():
            kit_textures[tm] = str(f.resolve())
        bf = team.path / "identity" / "badge.png"
        if bf.exists():
            badges[tm] = str(bf.resolve())
    print(f"RFL MATCH DAY: {a.name} ({a.code}) vs {b.name} ({b.code})")
    squads = []
    for tm, team in enumerate((a, b)):
        ctx = {"engine_version": ENGINE_VERSION, "team_index": tm,
               "config": yaml.safe_load((team.path / "team.yaml").read_text())}
        squads.append(team.build(ctx))
    agents = squads[0]["players"] + squads[1]["players"]
    managers = {tm: s["manager"] for tm, s in enumerate(squads)
                if s.get("manager") is not None}
    result = run_match(
        agents, match_time_s=match_time_s, mode=mode, halves=halves,
        decision_deadline_s=3.0, request_period_s=2.0,
        managers=managers, obs_mode="sdk",
        team_colors=(a.color, b.color),
        team_color_names=(a.color_name, b.color_name),
        team_names=(a.name, b.name), team_codes=(a.code, b.code),
        hair={0: ([pl.get("hair") or {} for pl in a.players] if a.players else a.hair),
              1: ([pl.get("hair") or {} for pl in b.players] if b.players else b.hair)},
        player_names={0: [pl.get("name", "") for pl in a.players],
                      1: [pl.get("name", "") for pl in b.players]},
        record_states=record_states, kit_textures=kit_textures, badges=badges,
        video_path=video_path, log_dir=log_dir)
    fixture = {"engine": ENGINE_VERSION,
               "home": {"team": a.name, "code": a.code, "goals": result.score[0]},
               "away": {"team": b.name, "code": b.code, "goals": result.score[1]},
               "winner": (a.name if result.winner == "A"
                          else b.name if result.winner == "B" else "draw")}
    print(f"FULL TIME: {a.code} {result.score[0]} - {result.score[1]} {b.code}"
          f"  ({fixture['winner']})")
    if log_dir:
        (Path(log_dir) / "fixture.json").write_text(json.dumps(fixture, indent=2))
    return result
