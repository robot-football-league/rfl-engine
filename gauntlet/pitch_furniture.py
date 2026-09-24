"""Shared pitch-furniture palette and meter layout; no physics/renderer imports."""

BOARD_BG_TOP = (0.075, 0.062, 0.105)
BOARD_BG_BOTTOM = (0.028, 0.022, 0.045)
RAM_DARK = tuple((top + bottom) / 2
                 for top, bottom in zip(BOARD_BG_TOP, BOARD_BG_BOTTOM))

# Overall bezel dimensions. Both skins fill along body-local -X -> +X;
# front faces -Y at body z=.20, top faces +Z, centred across panel depth.
RAM_METER_WIDTH_M = 2.1
RAM_METER_HEIGHT_M = 0.16
RAM_METER_SEGMENTS = 20
RAM_METER_GREEN = (0.16, 0.88, 0.34)
RAM_METER_BEZEL = (0.014, 0.017, 0.022)
RAM_METER_INACTIVE = (0.030, 0.080, 0.045)
RAM_METER_EMISSION = 0.65
RAM_METER_FRONT_Z_M = 0.20
RAM_METER_BORDER_M = 0.008
RAM_METER_GAP_M = 0.008
RAM_METER_CELL_WIDTH_M = (
    RAM_METER_WIDTH_M - 2 * RAM_METER_BORDER_M
    - (RAM_METER_SEGMENTS - 1) * RAM_METER_GAP_M
) / RAM_METER_SEGMENTS
RAM_METER_CELL_HEIGHT_M = RAM_METER_HEIGHT_M - 2 * RAM_METER_BORDER_M
# Slab centre distances from the existing visual surface. Outermost face
# is .8 mm above it, not a second floating housing; each slab is .2 mm thick.
RAM_METER_SKIN_HALF_M = 0.0001
RAM_METER_BEZEL_OFFSET_M = 0.0001
RAM_METER_CELL_OFFSET_M = 0.0004
RAM_METER_LIT_OFFSET_M = 0.0007


def ram_face_rgba(charge_fraction: float, pushing: bool = False) -> tuple:
    """Compatibility API: housing stays board-dark at every charge/phase."""
    return RAM_DARK + (1.0,)


def ram_meter_fraction(charge_fraction: float, pushing: bool = False) -> float:
    """Clamp normalized charge; extension, hold and retraction remain full."""
    return 1.0 if pushing else max(0.0, min(1.0, float(charge_fraction)))


def ram_meter_cells(charge_fraction: float, pushing: bool = False) -> tuple:
    """Return (body-local left X, lit width) for each cell, in metres.

    Partial leading cells grow left-to-right, not brighter. Gaps/borders are
    never illuminated: total lit width is exactly the fraction of the total
    available cell width. Zero widths mean hidden, NOT zero-sized native geoms.
    """
    progress = ram_meter_fraction(charge_fraction, pushing) * RAM_METER_SEGMENTS
    return tuple((
        -RAM_METER_WIDTH_M / 2 + RAM_METER_BORDER_M
        + i * (RAM_METER_CELL_WIDTH_M + RAM_METER_GAP_M),
        RAM_METER_CELL_WIDTH_M * max(0.0, min(1.0, progress - i)),
    ) for i in range(RAM_METER_SEGMENTS))
