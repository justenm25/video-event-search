"""End-to-end: a video path in, saved artifacts out.

    video -> YOLO detection + ByteTrack tracking -> per-frame boxes
          -> fold tracks into intervals -> detections.json + intervals.csv + summary.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Optional

from . import storage
from .detector import run_tracking
from .intervals import build_intervals, summarize, write_csv


def _detections_payload(result) -> Dict:
    return {
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "total_frames": result.total_frames,
        "duration": result.duration,
        "frames": [
            {
                "t": fr.t,
                "dets": [
                    {"id": d.track_id, "cls": d.name, "conf": d.conf, "box": d.xyxy}
                    for d in fr.dets
                ],
            }
            for fr in result.frames
        ],
    }


def process_video(
    video_path: Path,
    video_id: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    result = run_tracking(video_path, progress_cb=progress_cb)
    intervals = build_intervals(result)
    per_class = summarize(intervals)

    storage.detections_path(video_id).write_text(
        json.dumps(_detections_payload(result)), encoding="utf-8"
    )
    write_csv(intervals, storage.intervals_path(video_id))

    summary = {
        "video_id": video_id,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "duration": result.duration,
        "total_frames": result.total_frames,
        "num_intervals": len(intervals),
        "classes": per_class,
        "intervals": [asdict(iv) for iv in intervals],
    }
    storage.summary_path(video_id).write_text(json.dumps(summary), encoding="utf-8")
    return summary
