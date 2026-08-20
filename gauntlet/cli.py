"""Command-line entry points. Run as: python -m gauntlet <cmd> ..."""

from __future__ import annotations

import argparse
import json
import os


def _load_env_file():
    """Load KEY=value lines from the repo-root .env (API keys etc.).

    Variables already set in the shell win; comments, blank lines, optional
    'export ' prefixes, and quoted values are handled. No dependency needed.
    """
    from . import paths
    env_path = paths.ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if sep and key and value and key not in os.environ:
            os.environ[key] = value


def _load_course(name: str):
    from . import paths
    from .course import CourseSpec, generate_course, hand_courses

    if name.startswith("seed:"):
        return generate_course(int(name.split(":", 1)[1])), name.replace(":", "_")
    if name.startswith("takeshi:"):
        from .takeshi import generate_takeshi_course
        return (generate_takeshi_course(int(name.split(":", 1)[1])),
                name.replace(":", "_"))
    if name.startswith("hand_"):
        hc = hand_courses()
        if name in hc:
            return hc[name], name
    p = paths.COURSES / f"{name}.json"
    if p.exists():
        return CourseSpec.load(p), name
    raise SystemExit(f"unknown course: {name} (use seed:<n>, takeshi:<n>, "
                     "hand_a/b/c, or a courses/*.json name)")


def main(argv=None):
    _load_env_file()
    p = argparse.ArgumentParser(prog="gauntlet", description="G1 Gauntlet harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    fb = sub.add_parser("football", help="V4: 2v2 football match")
    fb.add_argument("--team-a", default="scripted", help="agent spec for both A players")
    fb.add_argument("--team-b", default="scripted", help="agent spec for both B players")
    fb.add_argument("--time", type=float, default=90.0, help="match length (s)")
    fb.add_argument("--mode", choices=["paused", "realtime"], default="paused")
    fb.add_argument("--video", default=None)
    fb.add_argument("--out", default=None, help="log dir")
    fb.add_argument("--seed", type=int, default=0)
    fb.add_argument("--manager-a", default=None, help="agent spec for team A's manager")
    fb.add_argument("--manager-b", default=None, help="agent spec for team B's manager")
    fb.add_argument("--obs", choices=["full", "camera"], default="full")

    rs = sub.add_parser("rfl-serve", help="RFL 0.2: networked game server")
    rs.add_argument("--port", type=int, default=8800)
    rs.add_argument("--time", type=float, default=90.0)
    rs.add_argument("--deadline", type=float, default=3.0)
    rs.add_argument("--period", type=float, default=2.0)
    rs.add_argument("--video", default=None)
    rs.add_argument("--out", default=None)
    rs.add_argument("--tokens", default=None,
                    help="comma-separated team tokens (team order); optional")

    rf = sub.add_parser("rfl", help="RFL match day: two team directories")
    rf.add_argument("team_a", help="path to home team directory")
    rf.add_argument("team_b", help="path to away team directory")
    rf.add_argument("--time", type=float, default=90.0)
    rf.add_argument("--halves", type=int, choices=[1, 2], default=1)
    rf.add_argument("--mode", choices=["paused", "realtime"], default="realtime")
    rf.add_argument("--video", default=None)
    rf.add_argument("--out", default=None)

    au = sub.add_parser("sound", help="broadcast audio: mix + mux a match dir")
    au.add_argument("match_dir", help="log dir holding match.json + the video")
    au.add_argument("--video", default=None, help="video path if not in dir")
    au.add_argument("--out", default=None, help="output mp4 (default *_tv.mp4)")

    lg = sub.add_parser("league", help="RFL league: play fixtures, standings")
    lg.add_argument("action", choices=["next", "table"])
    lg.add_argument("--league", default="league.yaml")
    lg.add_argument("--no-audio", action="store_true")

    bc = sub.add_parser("broadcast", help="Twitch: cards, program, stream")
    bc.add_argument("action", choices=["auth", "status", "preflight",
                                       "prepare", "build",
                                       "rehearse", "live", "schedule-week", "panels",
                                       "bio"])
    bc.add_argument("--league", default="league.yaml")
    bc.add_argument("--no-verify", action="store_true",
                    help="status: skip the live Twitch auth check")
    bc.add_argument("--text", default=None, help="bio: override the text")

    cm = sub.add_parser("commentary", help="LLM script + ElevenLabs voice")
    cm.add_argument("match_dir")
    cm.add_argument("--script-only", action="store_true",
                    help="write lines.json but skip TTS")
    cm.add_argument("--model", default="gemini-flash-latest")
    cm.add_argument("--team-a", default=None, help="fallback name for team A")
    cm.add_argument("--team-b", default=None, help="fallback name for team B")
    cm.add_argument("--players-a", default=None, help="comma: name1,name2")
    cm.add_argument("--players-b", default=None, help="comma: name1,name2")

    lt = sub.add_parser("lint", help="scrutineering: check a team dir against league law")
    lt.add_argument("team")

    gf = sub.add_parser("gaffer", help="run one gaffer night session for a club")
    gf.add_argument("--team", required=True)
    gf.add_argument("--model", required=True, help="llm:<provider>:<model>")
    gf.add_argument("--night", type=int, required=True)
    gf.add_argument("--budget", type=float, default=5.0)
    gf.add_argument("--data-dir", default=None)
    gf.add_argument("--ref-dir", default=None)

    args = p.parse_args(argv)

    if args.cmd in ("sound", "broadcast", "commentary"):
        try:                     # these need the RFL station, which is not
            from . import broadcast  # noqa: F401 -- part of the public engine
        except ImportError:
            import sys
            sys.exit(f"'{args.cmd}' is a station command (broadcast production"
                     " and Twitch); it is not included in the public engine.")

    if args.cmd == "lint":
        from .rfl_lint import main as lint_main
        import sys
        sys.exit(lint_main(args.team))

    if args.cmd == "gaffer":
        from .gaffer import run_night
        run_night(args.team, args.model, args.night, args.budget,
                  data_dir=args.data_dir, ref_dir=args.ref_dir)
        return

    if args.cmd == "football":
        import json as _json
        from .football import (make_football_agent, make_football_manager,
                               run_match)
        agents = [make_football_agent(args.team_a if i < 2 else args.team_b,
                                      i, seed=args.seed + i) for i in range(4)]
        managers = {}
        if args.manager_a:
            managers[0] = make_football_manager(args.manager_a, seed=args.seed + 10)
        if args.manager_b:
            managers[1] = make_football_manager(args.manager_b, seed=args.seed + 11)
        res = run_match(agents, match_time_s=args.time, mode=args.mode,
                        managers=managers, obs_mode=args.obs,
                        decision_deadline_s=3.0 if args.obs == "camera" else None,
                        request_period_s=2.0 if args.obs == "camera" else None,
                        video_path=args.video, log_dir=args.out)
        print(_json.dumps(res.to_dict(), indent=2))

    if args.cmd == "rfl":
        from .rfl import run_rfl_match
        run_rfl_match(args.team_a, args.team_b, match_time_s=args.time,
                      mode=args.mode, halves=args.halves,
                      video_path=args.video, log_dir=args.out)

    if args.cmd == "sound":
        from .broadcast_audio import add_match_audio
        add_match_audio(args.match_dir, video=args.video, out=args.out)

    if args.cmd == "league":
        from .league import play_next, print_table
        if args.action == "next":
            import sys
            if play_next(args.league, audio=not args.no_audio) is None:
                sys.exit(3)
        else:
            print_table(args.league)

    if args.cmd == "broadcast":
        if args.action == "auth":
            from .twitch import device_auth
            device_auth()
        elif args.action == "status":
            from .broadcast import status
            status(args.league, check_auth=not args.no_verify)
        elif args.action == "preflight":
            import sys as _sys

            from . import alert
            from .broadcast import preflight
            ok, problems = preflight(args.league)
            if not ok:
                alert.alert("preflight FAILED — next slot will not air",
                            "; ".join(problems), severity="error")
                _sys.exit(1)
        elif args.action == "prepare":
            from .league import play_next
            play_next(args.league, audio=True)
        elif args.action == "build":
            from .broadcast import build_program
            build_program(args.league)
        elif args.action == "rehearse":
            from .broadcast import push
            push(live=False, league_path=args.league)
        elif args.action == "live":
            from .broadcast import push
            push(live=True, league_path=args.league)
        elif args.action == "schedule-week":
            from .broadcast import schedule_week
            schedule_week(args.league)
        elif args.action == "bio":
            from .broadcast import set_bio
            set_bio(args.league, text=args.text)
        elif args.action == "panels":
            from .broadcast import render_panels
            render_panels(args.league)

    if args.cmd == "commentary":
        from .commentary import synthesize, write_script
        tn = pn = None
        if args.team_a or args.team_b:
            tn = {"A": args.team_a or "Team A", "B": args.team_b or "Team B"}
        if args.players_a or args.players_b:
            pn = {"A": (args.players_a or "#1,#2").split(","),
                  "B": (args.players_b or "#1,#2").split(",")}
        write_script(args.match_dir, model=args.model,
                     team_names=tn, player_names=pn)
        if not args.script_only:
            synthesize(args.match_dir)

    if args.cmd == "rfl-serve":
        from .rfl_server import serve_match
        serve_match(port=args.port, match_time_s=args.time,
                    decision_deadline_s=args.deadline,
                    request_period_s=args.period,
                    video_path=args.video, log_dir=args.out,
                    tokens=args.tokens.split(",") if args.tokens else None)
        return

if __name__ == "__main__":
    main()
