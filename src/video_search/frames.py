"""Extract frames from a video at a fixed time interval."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
from PIL import Image


@dataclass
class Frame:
    """A sampled video frame and the timestamp (seconds) it was taken at."""
    timestamp: float
    image: Image.Image


def iter_frames(video_path: str | Path, sample_every_sec: float = 1.0) -> Iterator[Frame]:
    """Yield one Frame roughly every `sample_every_sec` seconds of the video.

    Uses the video's reported FPS to convert the interval into a frame stride.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0  # fallback for containers that don't report FPS

    stride = max(1, int(round(fps * sample_every_sec)))

    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                timestamp = idx / fps
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield Frame(timestamp=timestamp, image=Image.fromarray(frame_rgb))
            idx += 1
    finally:
        cap.release()
