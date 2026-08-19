"""Safe command envelope measured in M0 (see docs/ENVELOPE.md)."""

from __future__ import annotations

import json

from . import paths

_DEFAULT = {"vx": [-0.8, 1.0], "vy": [-0.5, 0.5], "wz": [-1.0, 1.0]}


def load_envelope() -> dict:
    if paths.ENVELOPE_JSON.exists():
        return json.loads(paths.ENVELOPE_JSON.read_text())
    return dict(_DEFAULT)
