"""Offline, frame-indexed inserts into a silent constant-frame-rate MP4.

``assemble(base_path, [(base_frame_index, clip_path), ...], fps)`` replaces
base_path atomically, and returns that Path, only after complete validation.
An index means *before* that original base frame (0 prepends, frame count
appends); equal indices retain caller order. Indices never include inserts.
No inserts is a validated, byte-for-byte no-op. All inputs must contain exactly
one video stream and no audio/other streams: assemble before soundtrack muxing.

One base decoder, one insert decoder at a time, and one H.264 encoder stream
RGB frames through bounded pipes. Memory is O(frame size), not goal count or
match length; there is no growing filter graph or whole-match raw temp file.
Input/output ffprobe -count_frames scans add offline CPU cost. The assembled
video gets ONE additional lossy encode (libx264 CRF 18, yuv420p), not a packet
copy; original input clips are untouched. ffmpeg and ffprobe must be on PATH.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import os
from pathlib import Path
import subprocess
import tempfile
import json


@dataclass(frozen=True)
class _Video:
    width: int
    height: int
    frames: int


def _rate(value):
    if isinstance(value, bool):
        raise ValueError("fps must be a positive finite frame rate")
    try:
        rate = Fraction(str(value)).limit_denominator(1_000_000)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("fps must be a positive finite frame rate") from exc
    if rate <= 0:
        raise ValueError("fps must be a positive finite frame rate")
    return rate


def _errors(log):
    """Keep failure reporting bounded even if a damaged input logs at length."""
    log.flush()
    log.seek(0, os.SEEK_END)
    size = log.tell()
    log.seek(max(0, size - 8192))
    return log.read().decode("utf-8", errors="replace").strip()


def _probe(path, rate, work):
    with tempfile.TemporaryFile(dir=work) as log:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
             "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
             "-of", "json", str(path)],
            stdout=subprocess.PIPE, stderr=log, check=False,
        )
        errors = _errors(log)
    if result.returncode or errors:
        raise RuntimeError(f"ffprobe failed for {path}: {errors}")
    try:
        streams = json.loads(result.stdout)["streams"]
        if len(streams) != 1 or streams[0].get("codec_type") != "video":
            raise ValueError("expected exactly one video stream and no audio/other streams")
        stream = streams[0]
        if any(Fraction(stream[key]) != rate
               for key in ("avg_frame_rate", "r_frame_rate")):
            raise ValueError(f"frame rate does not match {rate}")
        video = _Video(int(stream["width"]), int(stream["height"]),
                       int(stream["nb_read_frames"]))
        if min(video.width, video.height, video.frames) <= 0:
            raise ValueError("video must have positive dimensions and decoded frame count")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid video {path}: {exc}") from exc
    return video


@contextmanager
def _process(args, work, *, stdin=None, stdout=None):
    """Reap only our child on every path; stderr cannot block a media pipe."""
    with tempfile.TemporaryFile(dir=work) as log:
        proc = subprocess.Popen(args, stdin=stdin, stdout=stdout, stderr=log)
        try:
            yield proc
            if proc.stdin is not None:
                proc.stdin.close()
            code = proc.wait()
            errors = _errors(log)
            if code or errors:
                raise RuntimeError(f"ffmpeg failed (exit {code}): {errors}")
        except BrokenPipeError as exc:
            # The consumer died; retrieve its diagnostic before closing up.
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            raise RuntimeError(f"ffmpeg pipe failed: {_errors(log)}") from exc
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None and not pipe.closed:
                    try:
                        pipe.close()
                    except BrokenPipeError:
                        pass


def _decoder(path, work):
    return _process(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
         "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
         "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-threads", "1",
         "-f", "rawvideo", "pipe:1"], work,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
    )


def _copy_frames(source, destination, count, frame_bytes):
    for _ in range(count):
        # BufferedReader.read normally fills this; explicitly handle short reads.
        remaining = frame_bytes
        while remaining:
            chunk = source.read(remaining)
            if not chunk:
                raise RuntimeError("ffmpeg decoded fewer frames than ffprobe counted")
            destination.write(chunk)
            remaining -= len(chunk)


def _expect_eof(source):
    if source.read(1):
        raise RuntimeError("ffmpeg decoded more frames than ffprobe counted")


def assemble(base_path, inserts, fps):
    """Validate and atomically insert clips; ValueError/OS errors/RuntimeError
    leave the original base bytes intact. Call only after all writers close.
    Inputs must not be concurrently modified while assembly is running.
    """
    rate = _rate(fps)
    base = Path(base_path).absolute()
    pending = []
    previous = 0
    for index, clip in inserts:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("insert frame indices must be integers (not bools)")
        if index < previous:
            raise ValueError("insert frame indices must be nonnegative and sorted")
        pending.append((int(index), Path(clip).absolute()))
        previous = index

    with tempfile.TemporaryDirectory(prefix=".goal-video-", dir=base.parent) as work:
        original = _probe(base, rate, work)
        planned = []
        for index, clip in pending:
            if index > original.frames:
                raise ValueError(f"insert index {index} exceeds base frame count {original.frames}")
            video = _probe(clip, rate, work)
            if (video.width, video.height) != (original.width, original.height):
                raise ValueError(f"insert resolution does not match base: {clip}")
            planned.append((index, clip, video.frames))
        if not planned:
            return base

        output = Path(work) / "assembled.mp4"
        frame_bytes = original.width * original.height * 3
        encoder = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
            f"{original.width}x{original.height}", "-framerate", str(rate),
            "-i", "pipe:0", "-map", "0:v:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-threads", "1", "-fps_mode", "passthrough", "-movflags", "+faststart",
            "-y", str(output),
        ]
        with _process(encoder, work, stdin=subprocess.PIPE,
                      stdout=subprocess.DEVNULL) as enc:
            with _decoder(base, work) as dec:
                cursor = 0
                for index, clip, frames in planned:
                    _copy_frames(dec.stdout, enc.stdin, index - cursor, frame_bytes)
                    with _decoder(clip, work) as insert:
                        _copy_frames(insert.stdout, enc.stdin, frames, frame_bytes)
                        _expect_eof(insert.stdout)
                    cursor = index
                _copy_frames(dec.stdout, enc.stdin, original.frames - cursor, frame_bytes)
                _expect_eof(dec.stdout)

        verified = _probe(output, rate, work)
        expected = _Video(original.width, original.height,
                          original.frames + sum(frames for _, _, frames in planned))
        if verified != expected:
            raise RuntimeError(f"assembled video failed validation: {verified} != {expected}")
        os.replace(output, base)
    return base
