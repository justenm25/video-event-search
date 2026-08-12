"""End-to-end: a video path in, saved artifacts out.

    video -> YOLO detection + ByteTrack tracking -> per-frame boxes
          -> fold tracks into intervals -> detections.json + intervals.csv + summary.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Optional

import config

from . import analytics, storage
from .detector import run_tracking
from .intervals import build_intervals, summarize, write_csv


def _det_dict(d) -> Dict:
    out = {"id": d.track_id, "cls": d.name, "conf": d.conf, "box": d.xyxy}
    # Optional robustness/CV extras, only emitted when a pass produced them.
    mask = getattr(d, "mask", None)
    kpts = getattr(d, "kpts", None)
    action = getattr(d, "action", None)
    depth = getattr(d, "depth", None)
    attrs = getattr(d, "attrs", None)
    if mask:
        out["mask"] = mask
    if kpts:
        out["kpts"] = kpts
    if action:
        out["action"] = action
    if depth is not None:
        out["depth"] = depth
    if attrs:
        out["attrs"] = attrs
    return out


def _detections_payload(result) -> Dict:
    return {
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "total_frames": result.total_frames,
        "duration": result.duration,
        "frames": [
            {"t": fr.t, "dets": [_det_dict(d) for d in fr.dets]}
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
    motion = analytics.build(result)

    storage.detections_path(video_id).write_text(
        json.dumps(_detections_payload(result)), encoding="utf-8"
    )
    write_csv(intervals, storage.intervals_path(video_id))

    # Occupancy heatmap image (best-effort; never fail the whole job over it).
    try:
        import cv2
        cv2.imwrite(str(storage.heatmap_path(video_id)), analytics.occupancy_heatmap(result))
    except Exception:
        pass

    # Depth preview of a representative frame (best-effort).
    if config.DEPTH_ENABLED:
        try:
            import cv2
            from . import depth as depthmod
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, (result.total_frames or 2) // 2))
            ok, frame_img = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(storage.depth_path(video_id)), depthmod.colorize(frame_img))
        except Exception:
            pass

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
        "analytics": motion,
    }
    storage.summary_path(video_id).write_text(json.dumps(summary), encoding="utf-8")
    return summary
