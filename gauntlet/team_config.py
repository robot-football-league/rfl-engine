"""One place that knows how a club may have written its kit.

Clubs author `team.yaml` themselves, and six of them produced four
different shapes for the same field:

    kit_home: {color: [r, g, b], color_name: terracotta}   # AFC Fable
    kit_home: [r, g, b]                                     # DeepSeek, Muse
    color_away: [r, g, b]                                   # GLM FC
    color: [r, g, b]                                        # the scaffold

Every one of them is a reasonable reading of "put your kit colours in
team.yaml", and scrutineering passed all four because it checks CODE, not
schema. The cost of that was three separate crashes on 2026-09-01, in
three different subsystems, from the identical assumption that `kit_home`
is a dict: the publisher, the match engine, and the broadcast card
renderer. Each was found by something breaking downstream of a club that
had already been declared eligible.

So the reading lives here once, and `rfl_lint` now REJECTS a shape it
cannot read — which is the half that actually helps a club, because it is
told while it can still act.
"""

from __future__ import annotations

DEFAULT = [0.5, 0.5, 0.5]


def _rgb(value):
    """An [r, g, b] from whatever the club wrote, or None."""
    if isinstance(value, dict):
        value = value.get("color")
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(c) for c in value[:3]]
        except (TypeError, ValueError):
            return None
    return None


def kit_color(cfg: dict, which: str = "home"):
    """The club's kit colour, however it chose to say so.

    `which` is "home" or "away". Home falls back to the top-level `color`
    (what the scaffold asks for); away falls back to `color_away`, which
    is what GLM FC wrote unprompted and is a perfectly sensible name.
    Returns None for an away kit the club never specified — a club with
    one kit is allowed, and the caller decides what to do about a clash.
    """
    cfg = cfg or {}
    if which == "home":
        return (_rgb(cfg.get("kit_home"))
                or _rgb(cfg.get("color"))
                or list(DEFAULT))
    return _rgb(cfg.get("kit_away")) or _rgb(cfg.get("color_away"))


def kit_name(cfg: dict, which: str = "home"):
    """The club's own word for the colour, when it gave one."""
    cfg = cfg or {}
    src = cfg.get("kit_home") if which == "home" else cfg.get("kit_away")
    if isinstance(src, dict) and src.get("color_name"):
        return src["color_name"]
    return cfg.get("color_name") if which == "home" else cfg.get("away_color_name")


def kit_problems(cfg: dict) -> list[str]:
    """Why this team.yaml's colours cannot be read. Empty means fine.

    Deliberately strict about the thing that bites and quiet about the
    rest: a club may write any of the four accepted shapes, but a kit that
    resolves to nothing at all is a club the engine cannot draw.
    """
    out = []
    for field in ("kit_home", "kit_away"):
        v = (cfg or {}).get(field)
        if v is None:
            continue
        if _rgb(v) is None:
            out.append(
                f"{field}: cannot read a colour from {v!r}. Write either "
                f"`{field}: [r, g, b]` with three numbers 0-1, or "
                f"`{field}: {{color: [r, g, b], color_name: ...}}`.")
    if kit_color(cfg, "home") is None:
        out.append("no home kit colour: set `color: [r, g, b]` or "
                   "`kit_home: [r, g, b]`.")
    return out


# Squared RGB distance below which two kits are too close to tell apart on
# a 720p stream. Matches the engine's long-standing threshold exactly —
# the point of sharing it is that a card can no longer show a club in a
# kit it did not wear.
CLASH_DISTANCE = 0.18


def kits_clash(home_rgb, away_rgb) -> bool:
    if not home_rgb or not away_rgb:
        return False
    return sum((x - y) ** 2 for x, y in zip(home_rgb[:3], away_rgb[:3])) < CLASH_DISTANCE


def worn_colors(home_cfg: dict, away_cfg: dict):
    """(home_rgb, away_rgb) as ACTUALLY WORN, applying the clash rule.

    The visiting club changes when the two home kits are too close, which
    is what the engine does at kick-off. The first preseason card drew two
    near-identical blues for a match the engine had already decided would
    be played in contrasting kits — the card was showing a fixture that
    did not happen.
    """
    h = kit_color(home_cfg, "home")
    a = kit_color(away_cfg, "home")
    if kits_clash(h, a):
        alt = kit_color(away_cfg, "away")
        if alt:
            return h, alt
    return h, a
