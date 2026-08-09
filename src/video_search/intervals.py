"""Fold per-frame tracked detections into time intervals.

This implements the core idea: instead of storing every frame an object appears
in, keep one row per *appearance* — a ``track_id`` with a start time and an end
time. As we scan frames in order, each track keeps a running end time that we
push forward while it stays visible; once it's been gone for longer than the gap
tolerance, we finalize the interval and write it out.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import config
from .detector import TrackingResult


@dataclass
class Interval:
    track_id: int
    cls: str
    start_sec: float
    end_sec: float
    start_time: str
    end_time: str
    duration_sec: float
    max_conf: float


def _fmt(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def build_intervals(result: TrackingResult) -> List[Interval]:
    gap = config.INTERVAL_GAP_SEC
    min_len = config.MIN_INTERVAL_SEC

    # open[track_id] = running interval state
    open_tracks: Dict[int, Dict] = {}
    finished: List[Interval] = []

    def close(tid: int) -> None:
        st = open_tracks.pop(tid)
        duration = st["last_t"] - st["start_t"]
        if duration < min_len:
            return
        finished.append(
            Interval(
                track_id=tid,
                cls=st["name"],
                start_sec=round(st["start_t"], 3),
                end_sec=round(st["last_t"], 3),
                start_time=_fmt(st["start_t"]),
                end_time=_fmt(st["last_t"]),
                duration_sec=round(duration, 3),
                max_conf=round(st["max_conf"], 3),
            )
        )

    for frame in result.frames:
        t = frame.t
        present = set()
        for det in frame.dets:
            present.add(det.track_id)
            st = open_tracks.get(det.track_id)
            if st is None:
                open_tracks[det.track_id] = {
                    "name": det.name,
                    "start_t": t,
                    "last_t": t,
                    "max_conf": det.conf,
                }
            else:
                st["last_t"] = t
                st["max_conf"] = max(st["max_conf"], det.conf)

        # Close any track absent for longer than the gap tolerance.
        for tid in [tid for tid, st in open_tracks.items()
                    if tid not in present and (t - st["last_t"]) > gap]:
            close(tid)

    for tid in list(open_tracks):
        close(tid)

    finished.sort(key=lambda iv: (iv.start_sec, iv.track_id))
    return finished


def write_csv(intervals: List[Interval], path: Path) -> None:
    fields = ["track_id", "cls", "start_time", "end_time",
              "start_sec", "end_sec", "duration_sec", "max_conf"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for iv in intervals:
            writer.writerow({k: asdict(iv)[k] for k in fields})


def summarize(intervals: List[Interval]) -> Dict:
    """Per-class rollup: how many distinct objects and total appearances."""
    per_class: Dict[str, Dict] = {}
    for iv in intervals:
        c = per_class.setdefault(iv.cls, {"objects": set(), "appearances": 0})
        c["objects"].add(iv.track_id)
        c["appearances"] += 1
    return {
        cls: {"count": len(v["objects"]), "appearances": v["appearances"]}
        for cls, v in sorted(per_class.items(), key=lambda kv: -len(kv[1]["objects"]))
    }
