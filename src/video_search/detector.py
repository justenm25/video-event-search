"""Run YOLO detection + ByteTrack tracking over a video.

Every kept box carries a persistent ``track_id`` (thanks to ByteTrack), so "the
same car" keeps one ID across frames. That ID is what lets us later fold the
per-frame boxes into start/end time intervals (see ``intervals.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

import config


@dataclass
class Detection:
    track_id: int
    cls: int
    name: str
    conf: float
    xyxy: List[float]  # [x1, y1, x2, y2] in original-frame pixels


@dataclass
class Frame:
    frame_idx: int
    t: float                       # timestamp in seconds
    dets: List[Detection] = field(default_factory=list)


@dataclass
class TrackingResult:
    fps: float
    width: int
    height: int
    total_frames: int
    duration: float
    frames: List[Frame]


def _video_meta(video_path: Path) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "width": width, "height": height, "total": total}


def run_tracking(
    video_path: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> TrackingResult:
    """Track objects through ``video_path``.

    ``progress_cb(done, total)`` is called after each processed frame so a UI
    can show a progress bar.
    """
    # Imported lazily so the module (and CLI --help) load without torch present.
    from ultralytics import YOLO

    meta = _video_meta(video_path)
    fps = meta["fps"]
    stride = max(1, config.FRAME_STRIDE)
    total_to_process = max(1, meta["total"] // stride) if meta["total"] else 0

    model = YOLO(config.YOLO_MODEL)
    results = model.track(
        source=str(video_path),
        stream=True,             # generator: one Results per processed frame
        tracker=config.TRACKER,
        conf=config.CONF_THRESHOLD,
        vid_stride=stride,
        verbose=False,
    )

    frames: List[Frame] = []
    for i, r in enumerate(results):
        frame_idx = i * stride
        t = frame_idx / fps if fps else float(i)
        frame = Frame(frame_idx=frame_idx, t=round(t, 3))

        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            names = r.names
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxys = boxes.xyxy.tolist()
            for tid, c, cf, box in zip(ids, clss, confs, xyxys):
                frame.dets.append(
                    Detection(
                        track_id=int(tid),
                        cls=int(c),
                        name=str(names[int(c)]),
                        conf=round(float(cf), 3),
                        xyxy=[round(float(v), 1) for v in box],
                    )
                )
        frames.append(frame)
        if progress_cb:
            progress_cb(i + 1, total_to_process)

    duration = frames[-1].t if frames else 0.0
    return TrackingResult(
        fps=round(fps, 3),
        width=meta["width"],
        height=meta["height"],
        total_frames=meta["total"],
        duration=round(duration, 3),
        frames=frames,
    )
