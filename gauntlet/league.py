"""RFL league driver: play fixtures from league.yaml, keep the table.

    python -m gauntlet league next      # play the next unplayed fixture
    python -m gauntlet league table     # print the standings

Each fixture lands in runs/league/s<season>/m<k>_<home>_<away>/ with the
broadcast video, then the sound pass produces match_tv.mp4 — the file a
stream schedule uploads. State lives in runs/league/s<season>/table.json:
one entry per played fixture, standings derived on demand. Designed so a
scheduler (cron / Twitch pipeline later) can just call `league next` once
per broadcast slot.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _load(league_path):
    cfg = yaml.safe_load(Path(league_path).read_text())
    root = Path("runs/league") / f"s{cfg.get('season', 1)}"
    root.mkdir(parents=True, exist_ok=True)
    state_p = root / "table.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"played": []}
    return cfg, root, state_p, state


def _team_name(team_dir):
    return yaml.safe_load((Path("teams") / team_dir / "team.yaml").read_text())


def _rollover(league_path, cfg, root):
    """Season N is done: archive its config, start season N+1 as a DOUBLE
    round robin (home/away reversed in the second half of the fixtures)."""
    import shutil
    season = cfg.get("season", 1)
    shutil.copy(league_path, root / "league.yaml")   # archive as played
    teams = cfg["teams"]
    single = [[teams[i], teams[j]] for i in range(len(teams))
              for j in range(i + 1, len(teams))]
    cfg2 = dict(cfg)
    cfg2["season"] = season + 1
    cfg2["fixtures"] = single + [[a, h] for h, a in single]
    Path(league_path).write_text(yaml.safe_dump(cfg2, sort_keys=False))
    print(f"  [league] season {season} archived -> season {season + 1} "
          f"begins: {len(cfg2['fixtures'])} matches (double round robin)")


def play_next(league_path="league.yaml", audio=True):
    cfg, root, state_p, state = _load(league_path)
    fixtures = cfg["fixtures"]
    k = len(state["played"])
    if k >= len(fixtures):
        print(f"season {cfg.get('season', 1)} complete "
              f"({len(fixtures)} matches played)")
        print_table(league_path)
        _rollover(league_path, cfg, root)
        cfg, root, state_p, state = _load(league_path)   # fresh season dir
        fixtures = cfg["fixtures"]
        k = 0
    home, away = fixtures[k]
    from .rfl_lint import check_team
    blocked = False
    for side in (home, away):
        problems = check_team(f"teams/{side}")
        if problems:
            print(f"  [league] {side} fails scrutineering — fixture not played:")
            for pr in problems:
                print(f"    - {pr}")
            blocked = True
    if blocked:
        return None
    out = root / f"m{k + 1}_{home}_{away}"
    out.mkdir(exist_ok=True)
    from .rfl import run_rfl_match
    res = run_rfl_match(
        f"teams/{home}", f"teams/{away}",
        match_time_s=float(cfg.get("match_time_s", 600)),
        halves=int(cfg.get("halves", 2)),
        video_path=str(out / "match.mp4"), log_dir=str(out))
    entry = {"fixture": k + 1, "home": home, "away": away,
             "score": list(res.score),
             "goals": res.goals, "est_cost_usd": res.est_cost_usd,
             "dir": str(out)}
    state["played"].append(entry)
    state_p.write_text(json.dumps(state, indent=2))
    if audio:
        try:                    # commentary first so the mix can embed it
            from .commentary import synthesize, write_script
            write_script(out, league=league_path)
            synthesize(out)
        except Exception as e:  # no key / no quota: crowd-only broadcast
            print(f"  [commentary] skipped: {e}")
        try:
            from .broadcast_audio import add_match_audio
            add_match_audio(out)
        except Exception as e:  # a silent match still counts
            print(f"  [audio] skipped: {e}")
    print_table(league_path)
    return entry


def compute_table(cfg, played):
    """Standings rows + order for a fixture list; shared with broadcast
    cards."""
    pts = cfg.get("points", {"win": 3, "draw": 1, "loss": 0})
    rows = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
            for t in cfg["teams"]}
    for m in played:
        h, a = m["home"], m["away"]
        gh, ga = m["score"]
        rows[h]["P"] += 1
        rows[a]["P"] += 1
        rows[h]["GF"] += gh
        rows[h]["GA"] += ga
        rows[a]["GF"] += ga
        rows[a]["GA"] += gh
        if gh > ga:
            rows[h]["W"] += 1
            rows[a]["L"] += 1
            rows[h]["Pts"] += pts["win"]
            rows[a]["Pts"] += pts["loss"]
        elif gh < ga:
            rows[a]["W"] += 1
            rows[h]["L"] += 1
            rows[a]["Pts"] += pts["win"]
            rows[h]["Pts"] += pts["loss"]
        else:
            rows[h]["D"] += 1
            rows[a]["D"] += 1
            rows[h]["Pts"] += pts["draw"]
            rows[a]["Pts"] += pts["draw"]
    order = sorted(rows, key=lambda t: (-rows[t]["Pts"],
                                        -(rows[t]["GF"] - rows[t]["GA"]),
                                        -rows[t]["GF"]))
    return rows, order


def print_table(league_path="league.yaml"):
    cfg, root, state_p, state = _load(league_path)
    rows, order = compute_table(cfg, state["played"])
    names = {t: _team_name(t)["name"] for t in cfg["teams"]}
    print(f"\n{cfg.get('name', 'league')} — season {cfg.get('season', 1)} "
          f"({len(state['played'])}/{len(cfg['fixtures'])} played)")
    print(f"  {'team':22} {'P':>2} {'W':>2} {'D':>2} {'L':>2} "
          f"{'GF':>3} {'GA':>3} {'Pts':>4}")
    for t in order:
        r = rows[t]
        print(f"  {names[t]:22} {r['P']:>2} {r['W']:>2} {r['D']:>2} "
              f"{r['L']:>2} {r['GF']:>3} {r['GA']:>3} {r['Pts']:>4}")
