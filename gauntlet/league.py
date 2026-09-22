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
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(league_path):
    cfg = yaml.safe_load(Path(league_path).read_text())
    root = Path("runs/league") / f"s{cfg.get('season', 1)}"
    root.mkdir(parents=True, exist_ok=True)
    state_p = root / "table.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"played": []}
    return cfg, root, state_p, state


def _team_name(team_dir):
    return yaml.safe_load((Path("teams") / team_dir / "team.yaml").read_text())


# How many commits back the league will look for code that plays. Ten, since
# 2026-09-21 (NOTICES): the published rule is "your LAST GOOD commit", and
# three — enough to step over a bad night and the night before it — was
# reached and beaten by a club that shipped six broken nights in a row.
# frontier_muse's last good commit was HEAD~6 and every fixture of its from
# m34 on was skipped for it. Ten is still few enough that a club cannot drift
# a month into its own past unnoticed: the skip alert names the fixture, and
# `_last_good` prints how far back it went.
FALLBACK_DEPTH = 10


def _kickoff_check(team_dir, team_index: int):
    """Do to a club exactly what the match will do to it — load team.py,
    build_team(ctx), begin_episode on every player and the manager — in a
    throwaway log dir, and return the error string if any of it raises.

    The rule in every gaffer's prompt says code that fails to load on match
    day is replaced by the club's last good commit. Until 2026-09-05 nothing
    enforced that at match time: scrutineering is static (imports and
    config), so a wrapper with a bad kickoff signature cleared it, the
    render died 18 s in, and the fixture was simply not played. This is the
    dynamic half. It costs a second or two per club and no tokens.
    """
    from .rfl import ENGINE_VERSION, load_team
    from .util import call_begin_episode
    team_dir = Path(team_dir)
    try:
        team = load_team(team_dir)
        ctx = {"engine_version": ENGINE_VERSION, "team_index": team_index,
               "config": yaml.safe_load((team_dir / "team.yaml").read_text())}
        squad = team.build(ctx)
        with tempfile.TemporaryDirectory() as td:
            for i, agent in enumerate(squad["players"]):
                if hasattr(agent, "begin_episode"):
                    d = Path(td) / f"r{i}"
                    d.mkdir()
                    call_begin_episode(agent, d)
            mgr = squad.get("manager")
            if mgr is not None and hasattr(mgr, "begin_episode"):
                d = Path(td) / "manager"
                d.mkdir()
                call_begin_episode(mgr, d)
    except Exception as e:                       # anything: it is the match
        return f"{type(e).__name__}: {e}"
    return None


def kickoff_main(team_dir) -> int:
    """`python -m gauntlet kickoff teams/<club>`: what match day does to a
    club before a ball is kicked, in the order it does it — scrutineering,
    then build_team(ctx) and begin_episode for both team indexes. No tokens.

    Exists because the 2026-09-15 and 2026-09-21 notices sent clubs to
    `lint`, which is static and cleared six consecutive commits that raised
    at kickoff. A club could not run the dynamic half itself; now it can.
    """
    from .rfl_lint import check_team
    team_dir = Path(team_dir)
    problems = check_team(team_dir)
    if problems:
        print(f"SCRUTINEERING FAILED: {team_dir}")
        for pr in problems:
            print(f"  - {pr}")
        return 1
    print(f"scrutineering clear: {team_dir}")
    for idx in (0, 1):
        err = _kickoff_check(team_dir, idx)
        if err:
            print(f"KICKOFF FAILED (as team {idx}): {team_dir}\n  {err}\n"
                  f"  On match day this code would not play: the match-day "
                  f"rule looks up to {FALLBACK_DEPTH} commits back for one "
                  f"that starts.")
            return 1
    print(f"kickoff clear: {team_dir} (build_team + begin_episode ran as "
          f"home and as away)")
    return 0


def _last_good(side: str, team_index: int, root: Path, teams_dir=Path("teams")):
    """The directory to play `side` from, and a record of any fallback.

    The live checkout if it passes _kickoff_check. Otherwise the newest of
    its last FALLBACK_DEPTH commits that BOTH clears scrutineering AND
    passes the same check, exported with `git archive` into
    runs/league/s<N>/fallback/<club>/<sha>/ — the club's repository is never
    touched. Returns (path, None) when the live code is fine, (path, record)
    when a fallback was taken, and (path, record with played=None) when
    nothing within reach is playable, which the caller treats as "fixture
    not played", exactly as before.
    """
    from .rfl import _git_out
    from .rfl_lint import check_team
    live = teams_dir / side
    err = _kickoff_check(live, team_index)
    if err is None:
        return str(live), None
    head = _git_out(live, "rev-parse", "--short", "HEAD") or "(no git)"
    print(f"  [league] {side} @ {head} fails at kickoff: {err}")
    record = {"failed": head, "error": err, "played": None, "back": 0}
    if not (live / ".git").exists():
        return str(live), record
    for n in range(1, FALLBACK_DEPTH + 1):
        sha = _git_out(live, "rev-parse", "--short", f"HEAD~{n}")
        if not sha:
            break
        dest = root / "fallback" / side / sha
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        arc = subprocess.run(["git", "-C", str(live), "archive", f"HEAD~{n}"],
                             capture_output=True)
        if arc.returncode != 0:
            continue
        subprocess.run(["tar", "-x", "-C", str(dest)], input=arc.stdout,
                       check=True)
        problems = check_team(dest)
        if problems:
            print(f"  [league] {side} @ {sha} (HEAD~{n}) fails scrutineering "
                  f"— skipping: {problems[0]}")
            continue
        e2 = _kickoff_check(dest, team_index)
        if e2 is None:
            print(f"  [league] {side}: playing last good commit {sha} "
                  f"(HEAD~{n}) — the match-day rule")
            record.update({"played": sha, "back": n})
            return str(dest), record
        print(f"  [league] {side} @ {sha} (HEAD~{n}) also fails at kickoff: {e2}")
    print(f"  [league] {side}: no playable commit within {FALLBACK_DEPTH}")
    return str(live), record


def double_round_robin(teams):
    """Home/away reversed in the second half — same shape as _rollover.

    ROUND ORDER MATTERS, and the nested-loop version got it wrong. It
    produced every correct pairing, but in the order teams[0]-v-everyone,
    then teams[1]-v-everyone: slice that into rounds of five and "round 1"
    is one club playing all five matches while four clubs do not play at
    all. Every one of season 3's 18 rounds was like that. The league slices
    this list into rounds everywhere it matters — the round-boundary gaffer
    sessions, rounds_done in league_loop.sh, the standings the audience
    reads — so the order is not cosmetic.

    Circle method: fix one club, rotate the rest. Each round is then a set
    of simultaneous pairings using every club exactly once, which is what a
    round IS. The fixed club alternates home and away so it does not play
    every one of its first-half matches at home.
    """
    n = len(teams)
    fixed, rot = teams[0], list(teams[1:])
    single = []
    for r in range(n - 1):
        single.append([fixed, rot[0]] if r % 2 == 0 else [rot[0], fixed])
        for i in range(1, n // 2):
            single.append([rot[i], rot[-i]])
        rot = [rot[-1]] + rot[:-1]
    return single + [[a, h] for h, a in single]


def _rollover(league_path, cfg, root):
    """Season N is done: archive its config, start season N+1 as a DOUBLE
    round robin (home/away reversed in the second half of the fixtures)."""
    import shutil
    season = cfg.get("season", 1)
    shutil.copy(league_path, root / "league.yaml")   # archive as played
    teams = cfg["teams"]
    cfg2 = dict(cfg)
    cfg2["season"] = season + 1
    cfg2["fixtures"] = double_round_robin(teams)
    Path(league_path).write_text(yaml.safe_dump(cfg2, sort_keys=False))
    print(f"  [league] season {season} archived -> season {season + 1} "
          f"begins: {len(cfg2['fixtures'])} matches (double round robin)")


def _accounted(state) -> set:
    """Fixture numbers the league is finished with: played, or skipped.

    `k = len(state["played"])` was fine while every fixture either played or
    stopped the league. Now that an unplayable fixture is skipped, the played
    list has gaps in it, and the fixture NUMBER is the only honest key — which
    is what every consumer already uses (`volumetric.py`, `broadcast.py`
    filter on `m["fixture"]`, never on list position).
    """
    return ({m["fixture"] for m in state.get("played", [])}
            | set(state.get("skipped_fixtures", [])))


def _next_index(state, fixtures) -> int:
    """0-based index of the first fixture neither played nor skipped."""
    done = _accounted(state)
    for i in range(len(fixtures)):
        if i + 1 not in done:
            return i
    return len(fixtures)


def _write_state(path, state):
    """An interrupted result writer must leave the previous table readable."""
    path = Path(path)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=".table-", suffix=".tmp",
                                         delete=False) as f:
            name = f.name
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _skip_fixture(state, state_p, k, home, away, why):
    """Record fixture k+1 as not played, say so loudly, and keep the league
    moving. Until 2026-09-15 this path returned None instead, and the loop
    then retried the SAME fixture every hour forever — one club took the
    whole league off air for three slots (frontier_muse, nights 17-22, its
    build_team using ctx.player_model against a ctx that is a dict). The
    club-facing half is in config/NOTICES.md, 2026-09-15.
    """
    n = k + 1
    state.setdefault("skipped_fixtures", []).append(n)
    state.setdefault("skipped", []).append(
        {"fixture": n, "home": home, "away": away, "why": why,
         "at": _utc_now()})
    _write_state(state_p, state)
    print(f"  [league] m{n} {home} v {away} NOT PLAYED — {why}")
    print(f"  [league] skipped; moving to the next fixture")
    try:
        from . import alert as _alert
        _alert.alert(f"m{n} skipped: {home} v {away} — {why}")
    except Exception as e:
        print(f"  [league] alert failed: {type(e).__name__}: {e}")


def play_next(league_path="league.yaml", audio=True):
    # Also protects direct CLI/library use, before _load can create a season
    # or read a table. render_next descendants reuse a VERIFIED inherited FD.
    from .render_lock import render_lock
    from .render_capacity import require_render_capacity
    with render_lock():
        # Before _load, fallback exports or output creation, including direct use.
        require_render_capacity()
        return _play_next_locked(league_path, audio)


def _play_next_locked(league_path, audio):
    cfg, root, state_p, state = _load(league_path)
    fixtures = cfg["fixtures"]
    k = _next_index(state, fixtures)
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
        # A skipped fixture was never rendered and will never be streamed,
        # so it must not hold the season open for ever.
        skipped = set(state.get("skipped_fixtures", []))
        unaired = [i for i in range(1, len(fixtures) + 1)
                   if i not in bs.get("streamed", []) and i not in skipped]
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
    # Walk forward over fixtures nobody can play. Each one is recorded and
    # alerted on; the league does not stop for it (NOTICES, 2026-09-15).
    from .rfl_lint import check_team
    while True:
        if k >= len(fixtures):
            print("  [league] no playable fixture remains in this season")
            return None
        home, away = fixtures[k]
        out = root / f"m{k + 1}_{home}_{away}"
        # A stale/pulled table must never authorize overwriting a completed
        # match (or debris from an interrupted writer). Refuse before checks
        # can skip the fixture or otherwise mutate state. Recovery is explicit.
        if os.path.lexists(out):
            raise FileExistsError(f"refusing existing match output: {out}; "
                                  "inspect/recover it before retrying")
        why = None
        for side in (home, away):
            problems = check_team(f"teams/{side}")
            if problems:
                print(f"  [league] {side} fails scrutineering:")
                for pr in problems:
                    print(f"    - {pr}")
                why = why or f"{side} fails scrutineering: {problems[0]}"
        if why is None:
            # The match-day rule, enforced where it bites (see _kickoff_check).
            home_path, home_fb = _last_good(home, 0, root)
            away_path, away_fb = _last_good(away, 1, root)
            for side, fb in ((home, home_fb), (away, away_fb)):
                if fb and fb["played"] is None:
                    why = why or (f"{side} has no playable commit within "
                                  f"{FALLBACK_DEPTH}: {fb['error']}")
        if why is None:
            break
        _skip_fixture(state, state_p, k, home, away, why)
        k = _next_index(state, fixtures)
    # Exclusive creation is the final guard even against non-cooperating
    # writers. Leave partial output intact on failure; never retry in place.
    out.mkdir(exist_ok=False)
    if home_fb or away_fb:
        (out / "fallback.json").write_text(json.dumps(
            {"home": home_fb, "away": away_fb}, indent=1))
    from .rfl import run_rfl_match
    res = run_rfl_match(
        home_path, away_path,
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
    _write_state(state_p, state)
    # Seal the table at this result, before later renders can change it.
    # The exporter only consumes this sidecar; it never backfills an old
    # recording from whatever league.yaml/table.json happens to say today.
    try:
        from .broadcast import _teams_cfg
        from .fulltime_table import write_snapshot
        write_snapshot(out, cfg, state["played"], _teams_cfg(cfg), k + 1,
                       res.to_dict())
    except Exception as e:
        # A missing graphic must not cost the commentary/audio/export pass
        # of an already-played match. Never guess a replacement table.
        print(f"  [full-time table] skipped: {type(e).__name__}: {e}")
    if audio:
        try:                    # commentary first so the mix can embed it
            from .commentary import (build_card_audio, synthesize,
                                     write_card_scripts, write_script)
            write_script(out, league=league_path)
            # CARD COMMENTARY IS OFF UNLESS ASKED FOR. This is a budget
            # decision, not a judgement about the writing.
            #
            # The build-up and wrap-up are 4,322 of the 8,894 characters a
            # match sends to TTS — 48% of the bill — for speech over the
            # pre-roll countdown and the post-match result card, either side
            # of the football. Measured 2026-09-04: at ElevenLabs' Creator
            # allowance and the flash_v2_5 rate (0.275 credits/char, measured
            # in isolation, NOT the 1.0 its multiplier implies), the league
            # gets 49.5 matches a month with cards and 96.3 without, against
            # 90 needed. Shortening them does not close that: the budget
            # leaves 319 characters for cards once the match commentary has
            # had its share, and cutting far enough to matter drives the card
            # to ~44% speech — the density that left "a 16 s hole between
            # every line" and was heard on air. So it is all or nothing, and
            # this is the end that keeps the football.
            #
            # Everything downstream degrades correctly: synthesize() skips
            # files that do not exist, build_card_audio() skips them too, and
            # broadcast._card_video falls back to the plain crowd-hum card it
            # always built. Set RFL_CARD_COMMENTARY=1 to turn them back on.
            if os.environ.get("RFL_CARD_COMMENTARY", "").strip().lower() in (
                    "1", "true", "yes"):
                try:            # build-up + wrap-up for the programme cards;
                                # never let them cost the match commentary
                    write_card_scripts(out, league=league_path)
                except Exception as e:
                    print(f"  [commentary] card scripts skipped: {e}")
            else:
                print("  [cards] commentary OFF (RFL_CARD_COMMENTARY unset) — "
                      "cards keep their crowd hum; saves ~48% of the TTS bill")
            synthesize(out)
            try:
                build_card_audio(out)
            except Exception as e:
                print(f"  [cards] audio skipped: {e}")
        except Exception as e:  # crowd-only broadcast — say so out loud
            # This used to guess "no key / no quota" in a comment and print
            # one line. On 2026-09-08 it swallowed a Gemini MAX_TOKENS
            # truncation twice and two matches aired silent; the key and the
            # quota were both fine. It alerts now, and broadcast_audio
            # shouts again when it finds no script to place.
            print(f"  [commentary] skipped: {type(e).__name__}: {e}")
            try:
                from . import alert as _alert
                _alert.alert(
                    "Commentary script FAILED — match will air crowd-only",
                    f"{Path(out).name}: {type(e).__name__}: {e}",
                    severity="error")
            except Exception as ae:     # never fail a render over an alert
                print(f"  [commentary] (alert failed: {ae})")
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
