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


def _extract_json(text: str):
    """First balanced {...} object in a possibly chatty reply."""
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
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _git(team_dir, *args, check=True):
    return subprocess.run(["git", "-C", str(team_dir), *args],
                          capture_output=True, text=True, check=check)


class GafferSession:
    def __init__(self, team_dir, model_spec, night, budget_usd=5.0,
                 data_dir=None, ref_dir=None):
        self.club = Path(team_dir).resolve()
        self.data = Path(data_dir).resolve() if data_dir else None
        self.ref = Path(ref_dir).resolve() if ref_dir else None
        self.night = night
        self.budget = float(budget_usd)
        self.spent = 0.0
        self.practices = 0
        self.log = []                 # (speaker, text) pairs
        _, provider, model = model_spec.split(":", 2)
        from .llm import LLMAgent
        self.adapter = LLMAgent(provider, model, prompt="football_v2")
        self.adapter.system_prompt = self._system_prompt(model_spec)
        self.adapter.max_output = 20_000
        self.adapter.effort = "high"
        self.model_spec = model_spec

    # ------------------------------------------------------------ prompt

    def _system_prompt(self, model_spec):
        tpl = Template((paths.PROMPTS / "system_gaffer_v1.md").read_text())
        founded = (self.club / "team.yaml").exists() and \
            "REPLACE" not in (self.club / "team.yaml").read_text()
        if founded:
            brief = (f"Game-day results through night {self.night} are in "
                     "data/. Review what happened, scout the table, improve "
                     "your club, and commit.")
            state = "# Your club\n\n" + (self.club / "team.yaml").read_text()
        else:
            brief = (
                "FOUNDING NIGHT. This club does not exist yet — create it.\n"
                "1. Choose a club name and a 3-letter code (the league "
                "assigns kit colors later — leave the color field as is).\n"
                "2. Name your two players and pick their hairstyles.\n"
                "3. Write club/team.yaml and club/team.py (start from "
                "reference/, then make it yours).\n"
                "4. Choose your player_model from data/models_registry.yaml "
                "— you pay its per-match price out of the club's match cap.\n"
                "5. Write your first PLAYBOOK.md: how you intend to play "
                "and iterate.\n"
                "6. lint, practice if you wish, then done.")
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
        return tpl.safe_substitute(
            model_label=model_spec.split(":", 2)[2],
            budget_usd=f"{self.budget:.2f}",
            night_brief=brief, club_state=state,
            playbook=playbook, notes_tail=notes)

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
        p = self._resolve(call["path"])
        text = p.read_text(errors="replace")
        if len(text) > READ_CAP:
            return (text[:READ_CAP] +
                    f"\n...[truncated: {len(text)} chars total]")
        return text

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
        for who, text in self.log:
            t = text if len(text) <= RESULT_CAP else \
                text[:RESULT_CAP] + "...[truncated]"
            lines.append(f"[{who}]\n{t}")
        lines.append(f"\n[budget] spent ${self.spent:.2f} of "
                     f"${self.budget:.2f}")
        lines.append("Your next tool call (one JSON object only):")
        return "\n\n".join(lines)

    def run(self):
        strikes = 0
        summary = "(no summary given)"
        for turn in range(MAX_TURNS):
            if self.spent >= self.budget:
                self.log.append(("harness", "BUDGET EXHAUSTED — session over"))
                summary = "session ended at budget cap"
                break
            text, usage, err = self.adapter._call_with_retries(
                self._render_transcript())
            if err:
                self.log.append(("harness", f"model error: {err}"))
                break
            if usage:
                c = estimate_cost(self.model_spec,
                                  usage.get("input_tokens") or 0,
                                  usage.get("output_tokens") or 0,
                                  usage.get("cache_read_input_tokens") or 0,
                                  usage.get("cache_creation_input_tokens") or 0)
                if c:
                    self.spent += c
            call = _extract_json(text or "")
            self.log.append(("gaffer", text or ""))
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
                result = f"tool error: {type(e).__name__}: {e}"
            self.log.append(("harness", f"{result}\n({time.time()-t0:.1f}s)"))
        return self._finish(summary)

    def _finish(self, summary):
        sess = self.club / "sessions"
        sess.mkdir(exist_ok=True)
        tr = [f"# night {self.night} — {self.model_spec}",
              f"budget ${self.budget:.2f}, spent ${self.spent:.2f}", ""]
        for who, text in self.log:
            tr += [f"## {who}", "", text, ""]
        (sess / f"night_{self.night:03d}.md").write_text("\n".join(tr))
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
        report = {"night": self.night, "model": self.model_spec,
                  "turns": len([1 for w, _ in self.log if w == "gaffer"]),
                  "spent_usd": round(self.spent, 3),
                  "practices": self.practices,
                  "commit": committed, "summary": summary}
        print(f"  [gaffer] {json.dumps(report)}")
        return report


def run_night(team_dir, model_spec, night, budget_usd=5.0,
              data_dir=None, ref_dir=None):
    return GafferSession(team_dir, model_spec, night, budget_usd,
                         data_dir=data_dir, ref_dir=ref_dir).run()
