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
        # A season is over when its matches have AIRED, not when they have
        # been rendered. Renders run days ahead of the broadcast, so rolling
        # on "all rendered" always fires while the previous season is still
        # on air — and `render_next.sh` then rsyncs the new league.yaml to
        # the box, which follows it and computes an EMPTY queue. That is
        # every remaining slot dead, silently.
        #
        # Seen for real on 2026-08-28: season 2 rolled with m25-m28 still to
        # air, including the finale. Refuse, and say what is holding it.
        bp = root / "broadcast.json"
        bs = json.loads(bp.read_text()) if bp.exists() else {"streamed": []}
        unaired = [i for i in range(1, len(fixtures) + 1)
                   if i not in bs.get("streamed", [])]
        if unaired:
            print(f"  NOT rolling over: {len(unaired)} match(es) rendered "
                  f"but not yet aired ({', '.join(f'm{i}' for i in unaired)})."
                  f"\n  The box follows league.yaml — rolling now would empty "
                  f"its queue and kill every remaining slot."
                  f"\n  Re-run once they have aired, or roll by hand.")
            return
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
        record_states=True,   # every fixture becomes volumetric-exportable
        video_path=str(out / "match.mp4"), log_dir=str(out))
    def roster(td):
        cfg_t = yaml.safe_load((Path("teams") / td / "team.yaml").read_text())
        return [p.get("name", f"#{j + 1}")
                for j, p in enumerate(cfg_t.get("players") or [])][:2]
    entry = {"fixture": k + 1, "home": home, "away": away,
             "score": list(res.score),
             "goals": res.goals, "est_cost_usd": res.est_cost_usd,
             "players": {"home": roster(home), "away": roster(away)},
             "dir": str(out)}
    state["played"].append(entry)
    state_p.write_text(json.dumps(state, indent=2))
    if audio:
        try:                    # commentary first so the mix can embed it
            from .commentary import (build_card_audio, synthesize,
                                     write_card_scripts, write_script)
            write_script(out, league=league_path)
            try:                # build-up + wrap-up for the programme cards;
                                # never let them cost the match commentary
                write_card_scripts(out, league=league_path)
            except Exception as e:
                print(f"  [commentary] card scripts skipped: {e}")
            synthesize(out)
            try:
                build_card_audio(out)
            except Exception as e:
                print(f"  [cards] audio skipped: {e}")
        except Exception as e:  # no key / no quota: crowd-only broadcast
            print(f"  [commentary] skipped: {e}")
        try:
            from .broadcast_audio import add_match_audio
            add_match_audio(out)
        except Exception as e:  # a silent match still counts
            print(f"  [audio] skipped: {e}")
    try:                        # volumetric replay bundle (after the audio
        from .volumetric import export_match      # mix, so it can carry it)
        export_match(out, fixture=k + 1)
    except Exception as e:      # a bundle-less match still counts
        print(f"  [volumetric] export skipped: {e}")
    try:                        # ...then queue it on 4dgsx.com (next free
        from .publish import publish_bundle       # broadcast slot)
        publish_bundle(out, season=int(cfg.get("season", 1)), fixture=k + 1)
    except Exception as e:      # an unpublished match still counts
        print(f"  [4dgsx] publish skipped: {e}")
    print_table(league_path)
    return entry


# --------------------------------------------------------------- movement
# Every table the league draws — the broadcast card, the Twitch panel, the
# website, the blog, the gaffer briefs, this CLI — shows the same up/down
# indicator, and it means the same thing on all of them: how a club's
# position has changed since the last COMPLETED round. That is the way a
# football table reads "since last week", and a round here is one fixture
# per club (len(teams) // 2 matches), so mid-round the arrows measure
# against the last table in which every club had played the same number.
#
# The baseline is derived from the SAME `played` list the caller passes,
# which is what keeps it spoiler-free: a public surface hands in aired
# fixtures only and gets arrows computed from aired fixtures only.

MOVE_UP, MOVE_DOWN, MOVE_SAME = "\u25b2", "\u25bc", "\u2013"


def move_label(move):
    """A club's movement as text: '\u25b22', '\u25bc1', '\u2013', or '' when there
    is no baseline to have moved from (nobody has moved in round 1)."""
    if move is None:
        return ""
    if move > 0:
        return f"{MOVE_UP}{move}"
    if move < 0:
        return f"{MOVE_DOWN}{-move}"
    return MOVE_SAME


def baseline_count(cfg, played):
    """How many matches deep the movement baseline sits — the end of the
    last completed round BEFORE the current one. 0 means no baseline: in
    round 1 there is no previous table, and comparing against an empty one
    would invent movement out of the config's team order."""
    per_round = max(1, len(cfg["teams"]) // 2)
    n = len(played)
    if n <= 0:
        return 0
    return per_round * (-(-n // per_round) - 1)      # ceil(n / per) - 1


def baseline_round(cfg, played):
    """The round number the arrows are measured from, or None if there is
    no baseline yet. Surfaces caption themselves with this so the reader
    knows what 'up 2' is up from."""
    per_round = max(1, len(cfg["teams"]) // 2)
    b = baseline_count(cfg, played)
    return (b // per_round) or None


def _standings(cfg, played):
    """Rows + order with no movement attached — also the baseline pass, so
    it must never call back into compute_table."""
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


def compute_table(cfg, played, baseline=None):
    """Standings rows + order for a fixture list; shared with the broadcast
    cards, the website feed, the blog posts and the gaffer briefs.

    Every row also carries `Prev` (its position at the movement baseline)
    and `Move` (places gained since — positive is up, negative is down,
    None when there is no baseline). See the movement note above.

    `baseline` overrides how many matches deep that baseline sits, for the
    callers asking something other than "how has the round gone so far":
    `len(played) - 1` asks what the LAST match alone changed, and 0 turns
    movement off altogether. Out of range it clamps, so a caller passing
    -1 for an empty list gets no movement rather than a table measured
    against itself.
    """
    rows, order = _standings(cfg, played)
    ordered = sorted(played, key=lambda m: m["fixture"])
    b = (baseline_count(cfg, ordered) if baseline is None
         else max(0, min(int(baseline), len(ordered))))
    prev_order = _standings(cfg, ordered[:b])[1] if b else None
    for pos, t in enumerate(order, 1):
        prev = prev_order.index(t) + 1 if prev_order else None
        rows[t]["Prev"] = prev
        rows[t]["Move"] = None if prev is None else prev - pos
    return rows, order


def print_table(league_path="league.yaml"):
    cfg, root, state_p, state = _load(league_path)
    rows, order = compute_table(cfg, state["played"])
    names = {t: _team_name(t)["name"] for t in cfg["teams"]}
    since = baseline_round(cfg, state["played"])
    print(f"\n{cfg.get('name', 'league')} — season {cfg.get('season', 1)} "
          f"({len(state['played'])}/{len(cfg['fixtures'])} played)")
    print(f"  {'#':>2} {'':2} {'team':22} {'P':>2} {'W':>2} {'D':>2} "
          f"{'L':>2} {'GF':>3} {'GA':>3} {'Pts':>4}")
    for pos, t in enumerate(order, 1):
        r = rows[t]
        print(f"  {pos:>2} {move_label(r['Move']):2} {names[t]:22} "
              f"{r['P']:>2} {r['W']:>2} {r['D']:>2} "
              f"{r['L']:>2} {r['GF']:>3} {r['GA']:>3} {r['Pts']:>4}")
    if since:
        print(f"  movement since round {since}")
