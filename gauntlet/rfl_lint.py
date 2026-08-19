"""Scrutineering: mechanical enforcement of the realism law on team code.

    python -m gauntlet lint teams/<club>

Checks every top-level .py in the team dir (tools/ is exempt: gaffer
analysis scripts never run at match time) against an import allowlist and
a short list of forbidden constructs, plus the team.yaml contract. A club
that fails scrutineering plays its last good commit instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

ALLOWED_MODULES = {
    # stdlib judged safe for a behaviour layer
    "math", "random", "json", "typing", "dataclasses", "collections",
    "itertools", "functools", "heapq", "statistics", "enum", "copy",
    "bisect", "array", "string", "re", "time",
    # numerics + the public engine surface
    "numpy", "gauntlet.football", "gauntlet.rfl_sdk",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open",
                   "breakpoint", "input", "vars", "globals"}


def _module_ok(name: str, local_stems: set[str]) -> bool:
    if name in ALLOWED_MODULES:
        return True
    root = name.split(".")[0]
    return root in local_stems           # sibling modules in the club repo


def check_team(team_dir) -> list[str]:
    """Return a list of violations; empty means the club is cleared."""
    team_dir = Path(team_dir)
    problems = []

    y = team_dir / "team.yaml"
    if not y.exists():
        return [f"missing {y.name}"]
    import yaml
    try:
        cfg = yaml.safe_load(y.read_text())
    except Exception as e:
        return [f"team.yaml unparseable: {e}"]
    if not isinstance(cfg, dict):
        return ["team.yaml is empty (club not founded yet)"]
    for key in ("name", "code", "player_model"):
        if not cfg.get(key):
            problems.append(f"team.yaml missing '{key}'")
    if not (cfg.get("color") or (cfg.get("kit_home") or {}).get("color")):
        problems.append("team.yaml needs kit_home.color (or legacy color)")
    if cfg.get("code") and len(str(cfg["code"])) != 3:
        problems.append("team.yaml 'code' must be exactly 3 letters")

    pys = sorted(team_dir.glob("*.py"))
    if not any(p.name == "team.py" for p in pys):
        problems.append("missing team.py")
    local_stems = {p.stem for p in pys}

    for p in pys:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as e:
            problems.append(f"{p.name}: syntax error: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if not _module_ok(a.name, local_stems):
                        problems.append(
                            f"{p.name}:{node.lineno}: import '{a.name}' "
                            "is outside the allowlist")
            elif isinstance(node, ast.ImportFrom):
                if node.level:            # relative sibling import: fine
                    continue
                if node.module and not _module_ok(node.module, local_stems):
                    problems.append(
                        f"{p.name}:{node.lineno}: import '{node.module}' "
                        "is outside the allowlist")
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else None
                if name in FORBIDDEN_CALLS:
                    problems.append(
                        f"{p.name}:{node.lineno}: call to '{name}' is "
                        "forbidden in match code")
    return problems


def main(team_dir) -> int:
    problems = check_team(team_dir)
    if problems:
        print(f"SCRUTINEERING FAILED: {team_dir}")
        for pr in problems:
            print(f"  - {pr}")
        return 1
    print(f"scrutineering clear: {team_dir}")
    return 0
