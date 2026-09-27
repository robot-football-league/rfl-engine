"""Pure, versioned future-match goal-celebration policy (no I/O or environment).

Public API: validate_goal_celebration(config, *, season) validates the whole
league config before any output/club work. resolve_goal_celebration(config,
*, season, match_index) returns an immutable GoalRelease for the actual
1-based match number AFTER skipped matches have been selected past.

GoalRelease.enabled is the effective bool. Its to_dict() is the expected
``goal_celebration`` provenance in fixture.json and the league played entry:
{"season": 3, "match_index": 48, "enabled": true,
 "policy": {"enabled": true, "version": 1, "season": 3, "first_match": 48}}.
An absent policy records null; explicit disabled records its supplied fields.
match.json records the engine's actual ``goal_explosion`` flag independently,
including when no goals occurred. Never infer old artifacts from today's policy.

Only a mapping is accepted: {enabled: false} disables, while enabling requires
all four fields. Supplied disabled fields still undergo type/version checks;
a disabled season pin may be stale. Enabled policies MUST match this season,
so rollover requires an explicit new decision rather than implicit activation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer (not bool)")


@dataclass(frozen=True)
class GoalCelebrationPolicy:
    enabled: bool
    version: int | None = None
    season: int | None = None
    first_match: int | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("goal_celebration.enabled must be a bool")
        for name in ("version", "season", "first_match"):
            value = getattr(self, name)
            if value is None:
                if self.enabled:
                    raise ValueError(f"enabled goal_celebration requires {name}")
            else:
                _positive_int(value, f"goal_celebration.{name}")
        if self.version is not None and self.version != 1:
            raise ValueError("unsupported goal_celebration.version (expected 1)")

    def to_dict(self):
        return {name: getattr(self, name)
                for name in ("enabled", "version", "season", "first_match")
                if getattr(self, name) is not None}


def validate_goal_celebration(config: Mapping, *, season: int) -> GoalCelebrationPolicy | None:
    """Return a validated policy (or None), before creating any artifacts."""
    _positive_int(season, "season")
    if not isinstance(config, Mapping):
        raise ValueError("league config must be a mapping")
    if "goal_celebration" not in config:
        return None
    raw = config["goal_celebration"]
    if not isinstance(raw, Mapping):
        raise ValueError("goal_celebration must be a mapping")
    if set(raw) - {"enabled", "version", "season", "first_match"}:
        raise ValueError("unknown goal_celebration fields")
    if "enabled" not in raw:
        raise ValueError("goal_celebration requires enabled")
    # YAML null is not a missing optional field, even in a disabled policy.
    for name in ("version", "season", "first_match"):
        if name in raw:
            _positive_int(raw[name], f"goal_celebration.{name}")
    policy = GoalCelebrationPolicy(**raw)
    if policy.enabled and policy.season != season:
        raise ValueError("enabled goal_celebration season does not match league season")
    return policy


@dataclass(frozen=True)
class GoalRelease:
    season: int
    match_index: int
    policy: GoalCelebrationPolicy | None = None

    def __post_init__(self):
        _positive_int(self.season, "season")
        _positive_int(self.match_index, "match_index")
        if self.policy is not None:
            if not isinstance(self.policy, GoalCelebrationPolicy):
                raise ValueError("goal release policy must be a GoalCelebrationPolicy")
            if self.policy.enabled and self.policy.season != self.season:
                raise ValueError("enabled goal_celebration season does not match league season")

    @property
    def enabled(self) -> bool:
        return (self.policy is not None and self.policy.enabled
                and self.match_index >= self.policy.first_match)

    def to_dict(self) -> dict:
        return {"season": self.season, "match_index": self.match_index,
                "enabled": self.enabled,
                "policy": self.policy.to_dict() if self.policy is not None else None}


def resolve_goal_celebration(config: Mapping, *, season: int,
                             match_index: int) -> GoalRelease:
    """Resolve the actual 1-based match index; no clock, environment or state."""
    return GoalRelease(season, match_index,
                       validate_goal_celebration(config, season=season))
