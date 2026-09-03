"""The gaffer harness: one frontier model improving one club, nightly.

    python -m gauntlet gaffer --team teams/frontier_fable \\
        --model llm:anthropic:claude-fable-5 --night 3 --budget 5

Uniform by construction: every gaffer gets the same system prompt
(prompts/system_gaffer_v1.md), the same tools, the same budget mechanics.
What differs is the model — and whatever the model has built for itself
inside its own repo (PLAYBOOK.md, tools/, notes). The session transcript
is written to club/sessions/ and everything is committed to the club repo,
so the full audit trail is public.

The harness gives a gaffer access to exactly three trees:
  club/       its own repo                      (read/write)
  data/       the published league archive      (read-only)
  reference/  the public sample team            (read-only)
Rival clubs' code is simply not in the workspace.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from string import Template

from . import paths
from .llm import estimate_cost
from .rfl_lint import check_team

MAX_TURNS = 28
MAX_PRACTICE = 2
PRACTICE_CAP_S = 120.0
READ_CAP = 24_000          # chars of a file served per read
RESULT_CAP = 6_000         # chars of a tool result echoed into the log

# The transcript is re-sent on every turn, so an un-windowed session pays
# for its own history over and over: 28 turns of a growing log is roughly
# quadratic in tokens, and it was the single biggest reason a premium
# gaffer could not afford a season. Recent turns stay verbatim because
# that is where the work is; older ones keep only their first line, which
# is enough to remember what was tried without re-reading its output.
# 12, not 8: entries come in pairs (the call and its result), so 8 was only
# four turns of real memory. A founding night has to hold the rules, the
# model registry and the reference implementation in mind while writing
# team.yaml, and at 8 it lost them and re-read.
WINDOW_VERBATIM = 12       # most recent log entries kept in full
WINDOW_HEADLINE = 220      # chars kept from each older entry

# One transient API failure must not throw away a whole session. The
# adapter has already exhausted its own retries by the time an error
# reaches us, so this is a second, slower chance rather than a duplicate
# of that. GLM's founding night died on a single APITimeoutError after
# eight productive turns and committed nothing but its transcript.
MAX_MODEL_ERRORS = 3

# Session transcripts are PUBLISHED. A tool error carries the exception's
# message verbatim, and a FileNotFoundError's message is an absolute path on
# the machine the league runs on — so `/Users/<someone>/...` went out in three
# published transcripts (fable night_005, sol night_005, glm night_000) before
# anyone noticed on 2026-09-02. It is not a credential, but it is the station's
# private filesystem layout and it belongs to nobody's football club.
#
# Scrubbed where it ENTERS the log, so a gaffer never sees the path and cannot
# quote it back — in glm's transcript the club had echoed the path into its own
# reasoning, which no amount of scrubbing the harness's side alone would catch.
# Applied again over the finished transcript as a backstop.
_HOME_PATH = re.compile(r"/(?:Users|home)/[^/\s'\"<>|]+/")


def scrub_paths(text: str) -> str:
    """Replace absolute home-directory paths with an elision marker."""
    return _HOME_PATH.sub(".../", text)


# A single gaffer reply's output ceiling, thinking tokens INCLUDED — they
# bill as output. This was 20_000 until 2026-09-02, when one AFC Fable
# session cost $9.51 of a $10 daily limit: six calls returned 12k-20k output
# tokens at ~$1.30 each. Measured against what a gaffer actually says, that
# ceiling bought nothing — the largest legitimate reply in the billing export
# was 1,492 output tokens, and the thinking budget is 2,048, so ~3.5k is the
# real worst case. 8_000 leaves room for a long file write while capping one
# call near $0.52 instead of $1.30.
#
# The ceiling is also a LATENCY control, which is the half that actually bit:
# a 20k-token generation can outrun REQUEST_TIMEOUT_S, and a timed-out
# generation is billed in full whether or not we ever see it.
GAFFER_MAX_OUTPUT = 8_000

# The smallest reply worth paying for. A thinking call needs max_tokens above
# the thinking budget by at least 1024 (see LLMAgent._call_aiml_anthropic), so
# under this there is no valid request to make and the session should stop.
MIN_TURN_OUTPUT = 3_072

# A session must also end in bounded TIME, not just bounded money and
# turns. GLM's founding night took 3h38m of wall clock — not hung, but
# sitting out repeated `retry_after` backoffs on a degraded endpoint. That
# is survivable for a founding night run by hand and impossible for a
# round: six clubs in sequence would outlast the gap between rounds, and
# the render queue is waiting behind them.
#
# The cap is fair the same way the purse is: every club gets the SAME
# clock, and a slow endpoint costs its club session time rather than
# costing the league its schedule. A club cut off keeps everything it
# committed, and its last good code plays on — exactly what happens when
# a club runs out of money or turns.
SESSION_WALL_CAP_S = 5400.0        # 90 minutes


def _fmt_dur(seconds):
    m, s = divmod(int(max(0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _extract_json(text: str, with_end: bool = False):
    """First balanced {...} object in a possibly chatty reply.

    With `with_end`, also returns the index just past that object, so the
    caller can discard whatever the model wrote afterwards.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': in_str = False
            elif c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return (obj, i + 1) if with_end else obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return (None, 0) if with_end else None


def _git(team_dir, *args, check=True):
    return subprocess.run(["git", "-C", str(team_dir), *args],
                          capture_output=True, text=True, check=check)


class GafferSession:
    def __init__(self, team_dir, model_spec, night, budget_usd=5.0,
                 data_dir=None, ref_dir=None, budget_note=None,
                 wall_cap_s=None, season=None):
        # An unpriced model makes the budget cap silently dead: run() only
        # adds to self.spent `if c`, and estimate_cost returns None for a
        # model with no PRICES row — so the session would run to MAX_TURNS
        # believing it had spent nothing. Refuse instead. This is the guard
        # that keeps a $50 season from becoming an unbounded one.
        if estimate_cost(model_spec, 1_000_000, 1_000_000, 0, 0) is None:
            raise ValueError(
                f"no PRICES row for {model_spec!r} — spend could not be "
                f"metered, so the budget cap would not exist. Add the model "
                f"to PRICES in gauntlet/llm.py (measured, not guessed) "
                f"before running it.")
        self.club = Path(team_dir).resolve()
        self.budget_note = budget_note
        self.data = Path(data_dir).resolve() if data_dir else None
        self.ref = Path(ref_dir).resolve() if ref_dir else None
        self.night = night
        self.budget = float(budget_usd)
        self.spent = 0.0
        self.practices = 0
        self.reasoning_turns = 0
        self.reasoning_tokens = 0
        self.reports = []             # issues this club filed against the league
        self.fabricated_chars = 0     # league-voice text the model invented
        # Set HERE, not by the caller afterwards: _system_prompt runs during
        # __init__ and has to name the current season, and season numbers are
        # not chronological -- the preseason is s0 and it comes AFTER s2.
        self.season = season          # stamped into reports
        self.turn = 0                 # for the turns-left line
        self.started_at = time.time()
        self.wall_cap = float(wall_cap_s or SESSION_WALL_CAP_S)
        self.timed_out = False
        self.log = []                 # (speaker, text) pairs
        _, provider, model = model_spec.split(":", 2)
        from .llm import LLMAgent
        self.adapter = LLMAgent(provider, model, prompt="football_v2")
        self.adapter.system_prompt = self._system_prompt(model_spec)
        self.adapter.max_output = GAFFER_MAX_OUTPUT
        self.adapter.effort = "high"
        # Ask the provider for its reasoning where one will give it (see
        # LLMAgent._call_aiml). A player brain never sets this: thinking
        # costs seconds it does not have inside the decision interval.
        self.adapter.want_reasoning = True
        self.model_spec = model_spec

    # ------------------------------------------------------------ prompt

    def _system_prompt(self, model_spec):
        tpl = Template((paths.PROMPTS / "system_gaffer_v1.md").read_text())
        founded = (self.club / "team.yaml").exists() and \
            "REPLACE" not in (self.club / "team.yaml").read_text()
        if founded:
            # Name the DIRECTORIES that exist, not a night number. The old
            # wording -- "results through night N are in data/" -- described
            # the data by a counter no directory is named after, and AFC
            # Fable twice reasoned from the league's season number to
            # data/seasons/s3/, found nothing, and filed a blocker report
            # instead of preparing. Its own friendly was sitting in
            # data/seasons/s0/ the whole time, which is where DeepSeek Rovers
            # found the digest that told it its player model was too slow.
            # Two sessions and $4.91 of one club's purse went on a sentence
            # that never said where to look.
            # Count the matches in each season and note when they were last
            # written. Recency has to come from the DATA: season numbers are
            # not in date order (the preseason is s0 and it ran after s2), so
            # neither the name nor the sort position tells you which football
            # happened most recently.
            seasons, counts, newest = [], {}, {}
            if self.data and (self.data / "seasons").is_dir():
                for d in sorted((self.data / "seasons").iterdir()):
                    if not d.is_dir():
                        continue
                    seasons.append(d.name)
                    ms = [m for m in d.glob("m*") if m.is_dir()]
                    counts[d.name] = len(ms)
                    newest[d.name] = max((m.stat().st_mtime for m in ms),
                                         default=0)
            # NEVER infer "most recent" from the sort order. Season numbers
            # are not chronological: the preseason is s0 and it runs AFTER
            # s2, so alphabetical order points a club at last season.
            cur = f"s{self.season}" if self.season is not None else None
            listing = ", ".join(f"{n} ({counts[n]} match"
                                f"{'' if counts[n] == 1 else 'es'})"
                                for n in seasons)
            digest = ("Each match directory has a digest.json beside the raw "
                      "match.json: read the digest first, it is the "
                      "counted-up version.")
            # A season that has just rolled over has an EMPTY directory. On
            # 2026-09-02 season 3 opened three minutes before AFC Fable sat
            # down; it was pointed at data/seasons/s3/, found nothing and
            # reported the archive missing for the third session running.
            # Every club would have hit this at the first round boundary.
            latest = max((n for n in seasons if counts[n]),
                         key=lambda n: newest[n], default=None)
            if cur in seasons and counts.get(cur):
                where = (f"data/seasons/ holds {listing}. The league is in "
                         f"season {self.season} right now, so your most recent "
                         f"matches are in data/seasons/{cur}/. Season numbers "
                         f"are not in date order. {digest}")
            elif cur in seasons and latest:
                where = (f"data/seasons/ holds {listing}. Season "
                         f"{self.season} has only just started and has no "
                         f"matches yet — that is expected, not a fault, so "
                         f"do not report it as missing data. The most "
                         f"recently played football is in "
                         f"data/seasons/{latest}/; review that. Season "
                         f"numbers are not in date order. {digest}")
            elif seasons:
                where = (f"data/seasons/ holds {listing}, one directory per "
                         f"season. {digest}")
            else:
                where = ("data/seasons/ is where match results live, one "
                         "directory per season.")
            brief = (f"Game-day results are in data/. {where} Review what "
                     "happened, scout the table, improve your club, and "
                     "commit.")
            state = "# Your club\n\n" + (self.club / "team.yaml").read_text()
        else:
            brief = (
                "FOUNDING NIGHT. This club does not exist yet — create it. "
                "The club is YOU, the model: name it after yourself, declare "
                "yourself in team.yaml (gaffer: model + maker), theme the "
                "players the same way, and use your maker's recognizable "
                "colors so spectators can tell models apart at a glance.\n"
                "1. Choose a club name and a unique 3-letter code.\n"
                "2. Name your two players and pick their hairstyles.\n"
                "3. Design your identity: kit_home and kit_away colors in "
                "team.yaml (away clearly distinct — worn on clashes), and "
                "a badge + kit designs in club/identity/ — as images if "
                "you can generate them, else club/identity/PROMPTS.md "
                "with one detailed image prompt per asset for the league "
                "to render.\n"
                "4. Write club/team.yaml (schema template is in the file) "
                "and club/team.py (start from reference/, make it yours).\n"
                "5. Choose your player_model from data/models_registry.yaml "
                "— you pay its per-match price out of the club's match cap.\n"
                "6. Write your first PLAYBOOK.md: how you intend to play "
                "and iterate.\n"
                "7. lint, practice if you wish, then done.")
            state = ("# Your club\n\nUnfounded. club/ contains only "
                     "scaffolding.")
        playbook = "(empty — write one)"
        pb = self.club / "PLAYBOOK.md"
        if pb.exists() and pb.read_text().strip():
            playbook = pb.read_text()[-8000:]
        notes = "(none yet)"
        nt = self.club / "NOTES.md"
        if nt.exists() and nt.read_text().strip():
            notes = nt.read_text()[-3000:]
        notices = "(none)"
        if self.data and (self.data / "NOTICES.md").exists():
            notices = (self.data / "NOTICES.md").read_text()[:4000]
        # What the league said back about issues clubs raised. Shown to
        # EVERY club, not just the one that filed: a defect one gaffer found
        # is usually one they all have, and a visible answer is what makes
        # the report tool worth using a second time.
        replies = "(nothing outstanding)"
        try:
            # NB: no `from . import paths` here — a function-local import of
            # a module already imported at module scope makes `paths` local
            # to the WHOLE function, so the earlier use above it raises
            # UnboundLocalError. It did.
            rp = Path(paths.ROOT) / "config" / "gaffer_replies.yaml"
            if rp.exists():
                import yaml
                data = yaml.safe_load(rp.read_text()) or {}
                items = list(data.get("all") or [])
                items += list((data.get(self.club.name) or []))
                if items:
                    replies = "\n".join(f"- {i}" for i in items)[:3000]
        except Exception:
            pass                      # a briefing must never fail on this
        return tpl.safe_substitute(
            model_label=model_spec.split(":", 2)[2],
            budget_usd=f"{self.budget:.2f}",
            season_purse=self.budget_note or
            (f"This session may spend up to ${self.budget:.2f}."),
            night_brief=brief, club_state=state,
            playbook=playbook, notes_tail=notes,
            league_notices=notices, league_replies=replies)

    # ------------------------------------------------------------ paths

    def _resolve(self, rel, write=False):
        rel = str(rel)
        roots = {"club": (self.club, True),
                 "data": (self.data, False),
                 "reference": (self.ref, False)}
        head, _, rest = rel.partition("/")
        if head not in roots or roots[head][0] is None:
            raise ValueError(f"path must start with one of {list(roots)}")
        root, writable = roots[head]
        p = (root / rest).resolve()
        if not str(p).startswith(str(root)):
            raise ValueError("path escapes its root")
        if write and not writable:
            raise ValueError(f"{head}/ is read-only")
        return p

    # ------------------------------------------------------------ tools

    def _tool_ls(self, call):
        out = []
        for label, root in (("club", self.club), ("data", self.data),
                            ("reference", self.ref)):
            if root is None:
                continue
            out.append(f"{label}/")
            for p in sorted(root.rglob("*")):
                if any(part.startswith(".") or part == "__pycache__"
                       for part in p.relative_to(root).parts):
                    continue
                if p.is_file():
                    kb = p.stat().st_size / 1024
                    out.append(f"  {label}/{p.relative_to(root)}  ({kb:.0f} KB)")
        return "\n".join(out)

    def _tool_read(self, call):
        """Read a file, optionally from an offset — a club's own decisions
        log runs to ~1.3 MB and READ_CAP is 24 KB, so without paging a
        gaffer could see the first 1.8% of its own match data and nothing
        else, for ever. Truncation at the head is the worst possible slice:
        it is the opening seconds of a match, every time.
        """
        p = self._resolve(call["path"])
        text = p.read_text(errors="replace")
        total = len(text)
        try:
            offset = max(0, int(call.get("offset") or 0))
        except (TypeError, ValueError):
            offset = 0
        if offset >= total and total:
            return (f"offset {offset} is past the end of {call['path']} "
                    f"({total} chars). Use a smaller offset.")
        chunk = text[offset:offset + READ_CAP]
        if total <= READ_CAP and not offset:
            return chunk
        end = offset + len(chunk)
        more = (f"\n...[showing {offset}-{end} of {total} chars. "
                f"Read on with {{\"tool\": \"read\", \"path\": "
                f"\"{call['path']}\", \"offset\": {end}}}]"
                if end < total else
                f"\n...[showing {offset}-{end} of {total} chars — end of file]")
        return chunk + more

    def _tool_write(self, call):
        p = self._resolve(call["path"], write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(call["content"])
        return f"wrote {call['path']} ({len(call['content'])} chars)"

    def _tool_replace(self, call):
        p = self._resolve(call["path"], write=True)
        s = p.read_text()
        n = s.count(call["old"])
        if n != 1:
            return f"ERROR: old occurs {n} times in {call['path']} (need exactly 1)"
        p.write_text(s.replace(call["old"], call["new"]))
        return f"replaced in {call['path']}"

    def _tool_lint(self, call):
        problems = check_team(self.club)
        return ("scrutineering CLEAR" if not problems else
                "scrutineering FAILED:\n" + "\n".join(f"- {p}" for p in problems))

    def _tool_report(self, call):
        """Tell the league something is wrong with the league.

        This exists because of a real failure: for an entire season,
        match.json's `robots` block sat past READ_CAP and decisions.jsonl
        was 1.3 MB against a 24 KB window, so no club could read its own
        match data — and no club had any way to SAY so. The league found out
        by auditing itself, which is luck, not process. The participants are
        the best-placed observers of the league's defects and they had no
        channel.

        Deliberately:
        - written OUTSIDE the club repo, so a scrutineering revert (which
          undoes the club's whole night) cannot erase a report;
        - never charged beyond the tokens it takes to write — a club must
          never weigh reporting a bug against affording a session;
        - answered. Replies come back in the next session's briefing
          (config/gaffer_replies.yaml). A report into silence teaches a club
          to stop reporting, and then the channel is worse than none.
        """
        sev = str(call.get("severity", "bug")).lower().strip()
        if sev not in ("blocker", "bug", "suggestion", "rules"):
            sev = "bug"
        subject = str(call.get("subject", "")).strip()[:200]
        detail = str(call.get("detail", "")).strip()[:4000]
        if not subject:
            return ("ERROR: a report needs a `subject`. Say what is wrong in "
                    "one line, and put the evidence in `detail`.")
        rec = {"season": self.season, "night": self.night, "club": self.club.name,
               "model": self.model_spec, "severity": sev,
               "subject": subject, "detail": detail, "status": "open"}
        try:
            p = Path(paths.ROOT) / "runs" / "league" / "gaffer_reports.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as e:
            return f"report could not be filed ({e}); tell us again next night"
        self.reports.append(rec)
        return (f"Report filed with the league ({sev}): {subject!r}. A human "
                f"reads these. If it is actionable you will see a reply in a "
                f"future briefing. Filing this costs you nothing beyond the "
                f"tokens you just spent.")

    def _tool_note(self, call):
        nt = self.club / "NOTES.md"
        stamp = f"\n## night {self.night}\n{call['text']}\n"
        nt.write_text((nt.read_text() if nt.exists() else "# Notes\n") + stamp)
        return "noted"

    def _tool_practice(self, call):
        if self.practices >= MAX_PRACTICE:
            return "ERROR: practice allowance used up for tonight"
        secs = min(float(call.get("seconds", 60)), PRACTICE_CAP_S)
        problems = check_team(self.club)
        if problems:
            return ("cannot practice, scrutineering fails:\n" +
                    "\n".join(f"- {p}" for p in problems))
        self.practices += 1
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            mirror = Path(td) / "mirror"
            shutil.copytree(self.club, mirror,
                            ignore=shutil.ignore_patterns(
                                ".git", "sessions", "runs", "__pycache__"))
            out = Path(td) / "run"
            try:
                from .rfl import run_rfl_match
                res = run_rfl_match(str(self.club), str(mirror),
                                    match_time_s=secs, log_dir=str(out))
            except Exception as e:
                return f"practice crashed: {type(e).__name__}: {e}"
            if res.est_cost_usd:
                self.spent += res.est_cost_usd
            events = [e.get("type") for e in getattr(res, "events", []) or []]
            from collections import Counter
            tape = dict(Counter(events))
            return (f"practice ({secs:.0f}s): score {res.score[0]}-"
                    f"{res.score[1]}, events {tape}, "
                    f"cost ${res.est_cost_usd or 0:.3f}")

    # ------------------------------------------------------------ loop

    def _render_transcript(self):
        lines = ["SESSION LOG (oldest first):"] if self.log else \
                ["SESSION LOG: (empty — your first move)"]
        # Reasoning is written to the transcript, never replayed into the
        # prompt (see run()), so the window is computed over what remains.
        visible = [(w, t) for w, t in self.log if w != "reasoning"]
        cut = max(0, len(visible) - WINDOW_VERBATIM)
        if cut:
            lines.append(
                f"[{cut} earlier entries condensed to one line each — the "
                f"full transcript is saved to club/sessions/. Re-read a file "
                f"if you need its contents again.]")
        for i, (who, text) in enumerate(visible):
            if i < cut:
                head = (text or "").strip().splitlines()
                head = head[0] if head else ""
                if len(head) > WINDOW_HEADLINE:
                    head = head[:WINDOW_HEADLINE] + "..."
                lines.append(f"[{who}, earlier] {head}")
                continue
            t = text if len(text) <= RESULT_CAP else \
                text[:RESULT_CAP] + "...[truncated]"
            lines.append(f"[{who}]\n{t}")
        lines.append(f"\n[budget] this session: spent ${self.spent:.2f} of "
                     f"${self.budget:.2f}")
        t_left = self.wall_cap - (time.time() - self.started_at)
        lines.append(f"[clock] {_fmt_dur(t_left)} of "
                     f"{_fmt_dur(self.wall_cap)} left. Every club gets the "
                     f"same clock; a slow reply spends yours.")
        left = MAX_TURNS - self.turn
        lines.append(f"[turns] {left} of {MAX_TURNS} left. When they run out "
                     f"the session ends wherever it stands, so make the "
                     f"changes you have decided on before you run low — an "
                     f"unwritten decision is worth nothing.")
        lines.append("Your next turn (a short paragraph if you have "
                     "something to say, then one JSON object):")
        return "\n\n".join(lines)

    def run(self):
        strikes = 0
        model_errors = 0
        summary = "(no summary given)"
        for turn in range(MAX_TURNS):
            self.turn = turn
            elapsed = time.time() - self.started_at
            if elapsed >= self.wall_cap:
                self.log.append((
                    "harness",
                    f"WALL-CLOCK CAP: {_fmt_dur(elapsed)} of "
                    f"{_fmt_dur(self.wall_cap)} used — session over. "
                    f"Everything you committed stands."))
                summary = (f"session ended at the {_fmt_dur(self.wall_cap)} "
                           f"wall-clock cap after {turn} turn(s)")
                self.timed_out = True
                break
            # A HARD cap, not a line the next turn steps over. Checking
            # `spent >= budget` between turns lets a turn that starts at
            # $2.49 of $2.50 spend another $1.30 and finish at $3.79 —
            # AFC Fable's night 3 overshot by 6.8c that way, and the same
            # check is what a season purse is enforced with.
            #
            # So size the REQUEST to what is left: the reply cannot cost more
            # than its output ceiling allows, so lower the ceiling until the
            # worst case fits the remaining budget. Below the floor a
            # thinking call cannot even be formed (the budget must be at
            # least 1024 and strictly under max_tokens), so the session ends
            # there rather than issuing a call it cannot pay for.
            remaining = self.budget - self.spent
            po = estimate_cost(self.model_spec, 0, 1_000_000, 0, 0) or 0.0
            pi = estimate_cost(self.model_spec, 1_000_000, 0, 0, 0) or 0.0
            est_in_tok = (len(self.adapter.system_prompt or "")
                          + len(self._render_transcript())) / 4
            afford = remaining - est_in_tok * pi / 1e6
            afford_out = int(afford * 1e6 / po) if po > 0 else GAFFER_MAX_OUTPUT
            if remaining <= 0 or afford_out < MIN_TURN_OUTPUT:
                self.log.append(("harness", "BUDGET EXHAUSTED — session over"))
                summary = "session ended at budget cap"
                break
            self.adapter.max_output = min(GAFFER_MAX_OUTPUT, afford_out)
            text, usage, err = self.adapter._call_with_retries(
                self._render_transcript())
            if err:
                model_errors += 1
                self.log.append((
                    "harness",
                    f"model error ({model_errors}/{MAX_MODEL_ERRORS}): {err}"))
                if model_errors >= MAX_MODEL_ERRORS:
                    summary = f"session ended after {model_errors} model errors"
                    break
                # A gateway that says how long to wait means it. AIMLAPI's
                # 504s carry `retry_after: 120`; backing off 15 s against
                # that just spends the remaining attempts on an origin that
                # is still overloaded, which is how GLM's founding night
                # burned three errors in a row and founded nothing.
                wait = 20.0 * model_errors
                m = re.search(r"'retry_after':\s*(\d+)", str(err))
                if m:
                    wait = max(wait, min(float(m.group(1)), 180.0))
                remaining = self.wall_cap - (time.time() - self.started_at)
                if wait >= remaining:
                    self.log.append((
                        "harness",
                        f"backoff of {wait:.0f}s would outlast the "
                        f"{_fmt_dur(self.wall_cap)} session cap — stopping "
                        f"here rather than sleeping through the end."))
                    summary = "session ended: backoff exceeded the wall clock"
                    self.timed_out = True
                    break
                self.log.append(("harness", f"backing off {wait:.0f}s"))
                time.sleep(wait)
                continue
            model_errors = 0        # a good reply clears the streak
            if usage:
                c = estimate_cost(self.model_spec,
                                  usage.get("input_tokens") or 0,
                                  usage.get("output_tokens") or 0,
                                  usage.get("cache_read_input_tokens") or 0,
                                  usage.get("cache_creation_input_tokens") or 0)
                if c:
                    self.spent += c
            # The gaffer's own working, when the model exposes it. Kept in
            # the log so ordering is automatic and _finish writes it to the
            # public transcript — but _render_transcript SKIPS this speaker,
            # so it is never fed back into the next prompt. Models re-reason
            # from scratch each turn anyway, and re-sending a session's worth
            # of chain-of-thought would multiply the input bill for nothing.
            reasoning = getattr(self.adapter, "last_reasoning", None)
            if reasoning:
                self.reasoning_turns += 1
                self.reasoning_tokens += getattr(
                    self.adapter, "last_reasoning_tokens", 0) or 0
                self.log.append(("reasoning", reasoning))
            # A model shown a transcript will sometimes carry on writing
            # it — inventing the league's reply, the budget line and the
            # clock, then answering itself. Measured on a real billing
            # export: 3 of 20 gemini-3.7-flash calls did this, one of them
            # spending 2,335 output tokens against a typical hundred.
            #
            # Two harms, and the second is the serious one. It is paid for
            # by the club. And the fabricated text lands in a transcript
            # the league PUBLISHES, where "[budget] ... [clock] ..." reads
            # as the league speaking — in a page whose whole design is that
            # a reader can tell the club's voice from the league's.
            #
            # So the reply is cut at the end of its first JSON object.
            # Everything after it is the model writing the league's lines,
            # and the league does not print words it did not write.
            call, end = _extract_json(text or "", with_end=True)
            said = (text or "")
            if call is not None and end:
                trimmed = said[end:].strip()
                said = said[:end]
                if trimmed:
                    self.fabricated_chars += len(trimmed)
            self.log.append(("gaffer", said))
            if not call or "tool" not in call:
                strikes += 1
                self.log.append(("harness",
                                 "could not parse a tool call; reply with "
                                 "exactly one JSON object"))
                if strikes >= 3:
                    break
                continue
            strikes = 0
            if call["tool"] == "done":
                summary = str(call.get("summary", summary))[:400]
                break
            handler = getattr(self, f"_tool_{call['tool']}", None)
            if handler is None:
                self.log.append(("harness", f"unknown tool {call['tool']!r}"))
                continue
            t0 = time.time()
            try:
                result = handler(call)
            except Exception as e:
                result = scrub_paths(f"tool error: {type(e).__name__}: {e}")
            # Label the result with the call it answers. Without this, a
            # condensed older entry shows the first line of a FILE's contents
            # and nothing about which file — so a gaffer cannot tell it has
            # already read something and reads it again. Observed on GLM's
            # founding night: RFL_RULES.md (20.6 KB) and models_registry.yaml
            # fetched twice each, burning turns and money on a session that
            # then ran out of time before founding the club.
            what = call.get("tool", "?")
            if call.get("path"):
                what += f" {call['path']}"
                if call.get("offset"):
                    what += f"@{call['offset']}"
            self.log.append(("harness", scrub_paths(
                f"[{what}] {result}\n({time.time()-t0:.1f}s)")))
        return self._finish(summary)

    def _finish(self, summary):
        sess = self.club / "sessions"
        sess.mkdir(exist_ok=True)
        tr = [f"# night {self.night} — {self.model_spec}",
              f"budget ${self.budget:.2f}, spent ${self.spent:.2f}"]
        if self.reasoning_turns:
            tr.append(f"reasoning captured on {self.reasoning_turns} turn(s), "
                      f"{self.reasoning_tokens} reasoning tokens")
        tr.append("")
        # Labels a reader will understand. "reasoning" is the gaffer's own
        # working, published because it is the most interesting thing here;
        # "harness" is the league answering a tool call.
        label = {"gaffer": "gaffer — says",
                 "reasoning": "gaffer — thinking",
                 "harness": "league"}
        for who, text in self.log:
            tr += [f"## {label.get(who, who)}", "", text, ""]
        # Backstop. The scrub at the tool boundary is the real fix; this
        # catches a path that reached the log by any other route — a gaffer
        # quoting one back, or a future caller appending to self.log without
        # going through the tool handler.
        (sess / f"night_{self.night:03d}.md").write_text(
            scrub_paths("\n".join(tr)))
        if not (self.club / ".git").exists():
            _git(self.club, "init")
        _git(self.club, "add", "-A")
        diff = _git(self.club, "diff", "--cached", "--quiet", check=False)
        committed = None
        if diff.returncode != 0:
            _git(self.club, "-c", "user.name=RFL Gaffer",
                 "-c", "user.email=gaffer@rfl.invalid",
                 "commit", "-m", f"night {self.night}: {summary}")
            committed = _git(self.club, "rev-parse", "--short",
                             "HEAD").stdout.strip()
        # A feed needs to know WHEN. Without this the site can only order
        # sessions by night number, which says nothing about how they
        # interleave with matches and other clubs' nights.
        from datetime import datetime, timezone
        report = {"night": self.night, "model": self.model_spec,
                  "finished_at": datetime.now(timezone.utc)
                  .isoformat(timespec="seconds"),
                  "turns": len([1 for w, _ in self.log if w == "gaffer"]),
                  "spent_usd": round(self.spent, 3),
                  "practices": self.practices,
                  "reasoning_turns": self.reasoning_turns,
                  "fabricated_chars": self.fabricated_chars,
                  "reasoning_tokens": self.reasoning_tokens,
                  "reports": len(self.reports),
                  "wall_s": round(time.time() - self.started_at, 1),
                  "timed_out": self.timed_out,
                  "commit": committed, "summary": summary}
        # The sidecar the publisher reads. Written HERE so that every
        # session is publishable however it was started: sessions run
        # through the direct CLI produced a transcript and no sidecar, the
        # gate withheld them for want of a stamp, and three clubs' work
        # was invisible on the site while looking perfectly fine on disk.
        # Hand-stamping them afterwards is what put a wrong "no reasoning"
        # claim on GLM FC's page, so the harness records what it knows.
        try:
            import yaml
            lg = yaml.safe_load((Path(paths.ROOT) / "league.yaml").read_text())
            season = self.season if self.season is not None else lg.get("season")
            # Highest fixture that EXISTS as a render: what this session
            # could have seen, and therefore what must have aired before
            # its transcript may be published.
            through = 0
            sdir = Path(paths.ROOT) / "runs" / "league" / f"s{season}"
            for m in sdir.glob("m*/") if sdir.exists() else []:
                if ".failed." in m.name:
                    continue
                try:
                    through = max(through, int(m.name[1:].split("_", 1)[0]))
                except ValueError:
                    pass
            meta = dict(report)
            meta.update({"season": season, "rendered_through": through,
                         "founding": self.night == 0})
            (sess / f"night_{self.night:03d}.json").write_text(
                json.dumps(meta, indent=1))
        except Exception as e:
            print(f"  [gaffer] (sidecar not written: {type(e).__name__}: {e})")

        print(f"  [gaffer] {json.dumps(report)}")
        return report


def run_night(team_dir, model_spec, night, budget_usd=5.0,
              data_dir=None, ref_dir=None, budget_note=None, ledger=None,
              club=None, season=None, wall_cap_s=None):
    """Run one session. With a ledger, the spend is recorded even on a crash.

    The ledger write is in a `finally` on purpose: a session that burned two
    dollars and then threw has still burned two dollars, and a purse that
    forgets those is not a cap. `_finish` is skipped on that path — the
    transcript is lost, the money is not.
    """
    sess = GafferSession(team_dir, model_spec, night, budget_usd,
                         data_dir=data_dir, ref_dir=ref_dir,
                         budget_note=budget_note, wall_cap_s=wall_cap_s,
                         season=season)
    try:
        return sess.run()
    finally:
        if ledger is not None and club:
            ledger.record(club, sess.spent, night=night)
