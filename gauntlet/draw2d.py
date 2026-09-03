"""Author broadcast graphics at one resolution, draw them at another.

The scorebug, name plates, speech bubbles and programme cards were all laid
out against an 854x480 frame, in absolute pixels — `bug_r = 384`, a 26 px
badge, a 15 px font, a bubble that wraps at 190 px. Re-tuning forty numbers
to move to 720p is how a layout quietly stops matching itself.

So do not re-tune them. `Scaled` wraps `ImageDraw`, multiplies every
coordinate, radius and stroke width on the way through, and loads fonts at
the scaled size while `textlength` still answers in BASE units — so the
caller's arithmetic stays in the coordinate system it was written for.

At scale 1.0 it is a pass-through, which is the point: the old resolution
keeps its exact pixels, and that is checkable rather than hoped for.

    sd = Scaled(ImageDraw.Draw(img, "RGBA"), img.height / 480.0)
    f = sd.font(15)
    sd.rounded_rectangle([10, 8, 384, 38], radius=7, fill=(12, 12, 18, 215))
    sd.text((20, 23), "RMA", font=f, anchor="lm", fill=(255, 255, 255, 255))
"""

from __future__ import annotations

# THE BROADCAST FRAME, and the resolution the graphics are AUTHORED at.
# The match render and the programme cards are concatenated with `-c copy`,
# which demands identical parameters, and the same file is what gets pushed
# to air — so these are ONE set of numbers, imported by football.py and
# broadcast.py rather than repeated in either. 16:9 is a hard constraint:
# 4DGSX size their XR screen off the video's own dimensions.
#
# They live HERE, in the module that does the scaling, because both callers
# already import it and it imports nothing itself. broadcast.py cannot take
# them from football.py: that would pull mujoco and the policy net into the
# streaming box, which only ever assembles cards.
TV_W, TV_H, TV_FPS = 1280, 720, 50
TV_CRF = 23              # match render; the static cards can afford 21
BASE_W, BASE_H = 854, 480


# Overlay text is drawn in the best REAL sans this machine has, never in
# PIL's bundled default. That default has no em dash: `getmask("—")` returns
# the same full-height .notdef box as any absent glyph, and it aired that
# way — s3 m4's AFC Fable bubble read "rolling at our net [box] mine [box]
# no naps". The clubs are language models, so this is not an edge case: 696
# em dashes across 41 matches, in 11.9% of every shout line ever spoken.
# En dash, bullet, arrows and every accented letter draw the same box.
#
# A ladder, not a hard dependency. The station renders on macOS, the box is
# Debian, and a machine with none of these still gets its match — typography
# degrades to the bitmap font rather than failing a render. Same shape as
# football._board_font, which draws the hoardings.
#
# Measured, not guessed. Against the default over 914 distinct s3 shout
# strings, Arial runs 0.96-1.02x mean width (p95 <= 1.08x) and Helvetica the
# same, so the 854x480 layout these graphics are authored in still fits: no
# bubble grew a row, 49 lost one, max bubble width unchanged at 206 px.
# Verdana (1.12-1.19x) would not have fitted and is deliberately absent.
#
# ORDER MATTERS, because the match and the cards are rendered on DIFFERENT
# MACHINES: match_tv.mp4 on the Mac, the pre/post-roll cards on the box at
# slot time. Liberation Sans is metrically identical to Arial and is one
# apt package away on Debian (`fonts-liberation`) — it is listed first among
# the Linux rungs so that installing it makes both halves of the programme
# one typeface. Until then the box falls to DejaVu, which is 7-10% wider
# (max 17%): checked by rendering a real post-roll card through it, and the
# table still clears its columns.
_FONT_LADDER = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
_font_cache: dict[int, object] = {}


def load_font(px: int):
    """The best available real sans at `px` pixels, cached per size.

    Cached because `truetype` re-reads the file on every call and the panels
    ask for a font inside a per-club loop."""
    px = max(1, int(px))
    f = _font_cache.get(px)
    if f is None:
        from PIL import ImageFont
        for cand in _FONT_LADDER:
            try:
                f = ImageFont.truetype(cand, px)
                break
            except OSError:
                continue
        else:
            try:
                f = ImageFont.load_default(size=px)
            except TypeError:      # older Pillow: no size kwarg
                f = ImageFont.load_default()
        _font_cache[px] = f
    return f


class Font:
    """A font that remembers the size it was ASKED for, not the one it got."""

    __slots__ = ("base", "pil")

    def __init__(self, base: float, pil):
        self.base = base
        self.pil = pil


class Scaled:
    def __init__(self, draw, scale: float):
        self.d = draw
        self.s = float(scale)
        self._fonts: dict[int, Font] = {}

    # ---------------------------------------------------------------- fonts
    def font(self, size: float) -> Font:
        key = int(round(size * 100))
        f = self._fonts.get(key)
        if f is None:
            px = max(1, int(round(size * self.s)))
            f = self._fonts[key] = Font(size, load_font(px))
        return f

    def textlength(self, text, font=None) -> float:
        """Length in BASE units, so callers can keep laying out in base space."""
        f = font or self.font(10)
        return self.d.textlength(text, font=f.pil) / self.s

    # ----------------------------------------------------------- primitives
    def _xy(self, xy):
        s = self.s
        if xy is None:
            return None
        first = xy[0] if len(xy) else None
        if isinstance(first, (list, tuple)):
            return [tuple(v * s for v in p) for p in xy]
        return [v * s for v in xy]

    def _n(self, v):
        return None if v is None else v * self.s

    def _w(self, v):
        """Stroke width: scaled, but never rounded away to nothing."""
        return None if v is None else max(1, int(round(v * self.s)))

    def text(self, xy, text, font=None, **kw):
        f = font or self.font(10)
        self.d.text(tuple(self._xy(xy)), text, font=f.pil, **kw)

    def rectangle(self, xy, **kw):
        if "width" in kw:
            kw["width"] = self._w(kw["width"])
        self.d.rectangle(self._xy(xy), **kw)

    def rounded_rectangle(self, xy, radius=0, **kw):
        if "width" in kw:
            kw["width"] = self._w(kw["width"])
        self.d.rounded_rectangle(self._xy(xy), radius=self._n(radius), **kw)

    def ellipse(self, xy, **kw):
        if "width" in kw:
            kw["width"] = self._w(kw["width"])
        self.d.ellipse(self._xy(xy), **kw)

    def line(self, xy, **kw):
        if "width" in kw:
            kw["width"] = self._w(kw["width"])
        self.d.line(self._xy(xy), **kw)

    def polygon(self, xy, **kw):
        """Drawn, not typed: the default PIL font has no geometric-shape
        glyphs, so a table's up/down triangles are polygons or they are
        tofu boxes on air."""
        if "width" in kw:
            kw["width"] = self._w(kw["width"])
        self.d.polygon(self._xy(xy), **kw)

    # -------------------------------------------------------------- images
    def paste(self, canvas, im, box, mask=None):
        """Paste `im` at a BASE-space box, resizing it by the scale factor."""
        from PIL import Image as _I
        if self.s != 1.0:
            w = max(1, int(round(im.width * self.s)))
            h = max(1, int(round(im.height * self.s)))
            im = im.resize((w, h), _I.LANCZOS)
            mask = im if mask is not None else None
        canvas.paste(im, (int(round(box[0] * self.s)),
                          int(round(box[1] * self.s))), mask)

    def sized(self, im, w, h):
        """Resize a source image to a BASE-space width/height."""
        from PIL import Image as _I
        return im.resize((max(1, int(round(w * self.s))),
                          max(1, int(round(h * self.s)))), _I.LANCZOS)
