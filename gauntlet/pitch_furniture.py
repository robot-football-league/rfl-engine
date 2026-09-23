"""Shared pitch-furniture colours; safe to import without a renderer or physics."""

BOARD_BG_TOP = (0.075, 0.062, 0.105)
BOARD_BG_BOTTOM = (0.028, 0.022, 0.045)
RAM_DARK = tuple((top + bottom) / 2
                 for top, bottom in zip(BOARD_BG_TOP, BOARD_BG_BOTTOM))


def ram_face_rgba(charge_fraction: float, pushing: bool = False) -> tuple:
    """Whole face: board-dark at rest, white at full charge or during motion.

    Charge is normalised to [0, 1]; extension, hold and retraction all pass
    pushing=True. Clamp at the endpoints so decay/reset cannot overshoot.
    """
    fraction = 1.0 if pushing else max(0.0, min(1.0, float(charge_fraction)))
    return tuple(dark + (1.0 - dark) * fraction for dark in RAM_DARK) + (1.0,)
