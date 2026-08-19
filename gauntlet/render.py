"""Offscreen replay rendering: MuJoCo renderer piped to the system ffmpeg as MP4."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np


class VideoWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: int):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; required for MP4 replays")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def add(self, frame: np.ndarray):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()


class EpisodeRenderer:
    """Offscreen renderer writing an MP4 at fixed sim-time fps.

    track_body: body name for a tracking camera, or None for a free camera
    (set `lookat`; callers may mutate `self.cam.lookat` between frames, e.g.
    to follow the midpoint of two robots).
    """

    def __init__(self, model: mujoco.MjModel, path: str | Path,
                 width=854, height=480, fps=25,
                 distance=4.5, azimuth=135.0, elevation=-28.0,
                 track_body: str | None = "pelvis", lookat=(0.0, 0.0, 0.6)):
        self.model = model
        self.fps = fps
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        if track_body is not None:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, track_body)
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat = lookat
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.writer = VideoWriter(path, width, height, fps)
        self._frame_period = 1.0 / fps
        self._next_frame_t = 0.0

    overlay_fn = None  # optional: fn(frame_array, t) -> frame_array (subtitles etc.)

    def maybe_frame(self, data: mujoco.MjData, t: float):
        if t + 1e-9 >= self._next_frame_t:
            self.renderer.update_scene(data, camera=self.cam)
            frame = self.renderer.render()
            if self.overlay_fn is not None:
                frame = self.overlay_fn(frame, t)
            self.writer.add(frame)
            self._next_frame_t += self._frame_period

    def close(self):
        self.writer.close()
        self.renderer.close()
