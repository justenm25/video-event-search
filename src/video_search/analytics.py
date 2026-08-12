"""Motion analytics derived purely from the tracks (no extra model needed).

Once every object has a persistent ``track_id`` and a box per frame, a lot of
classic video-analytics falls out for free:

* **Trajectories & motion** — the path each object traces, how far it travelled,
  its average speed and net heading.
* **Occupancy heatmap** — where in the frame activity concentrated, as an image.
* **Line crossings** — how many objects crossed a reference line and in which
  direction (the basis of people-counting / traffic-counting).

Everything here is geometry over ``TrackingResult.frames``; the per-frame boxes
the front-end already has are enough to *draw* trajectories, so we only compute
the numeric summaries + the heatmap image on the backend.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

import config
from .detector import TrackingResult


def _box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _foot_point(box: List[float]) -> Tuple[float, float]:
    """Bottom-center of the box — where the object meets the ground plane."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def _collect_tracks(result: TrackingResult) -> Dict[int, Dict]:
    """Per track_id: its class and time-ordered center/foot points."""
    tracks: Dict[int, Dict] = {}
    for fr in result.frames:
        for d in fr.dets:
            tr = tracks.setdefault(d.track_id, {"cls": d.name, "pts": [], "foot": [],
                                                "actions": []})
            tr["cls"] = d.name  # voted class is stable; last wins harmlessly
            cx, cy = _box_center(d.xyxy)
            tr["pts"].append((fr.t, cx, cy))
            tr["foot"].append(_foot_point(d.xyxy))
            if getattr(d, "action", None):
                tr["actions"].append(d.action)
            if getattr(d, "attrs", None):
                tr["attrs"] = d.attrs
    return tracks


def _track_stats(pts: List[Tuple[float, float, float]]) -> Dict:
    """Distance travelled, net displacement, average speed, heading."""
    if len(pts) < 2:
        return {"distance_px": 0.0, "displacement_px": 0.0,
                "avg_speed_px_s": 0.0, "direction_deg": 0.0, "duration_sec": 0.0}
    dist = 0.0
    for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
        dist += math.hypot(x1 - x0, y1 - y0)
    (ts, xs, ys), (te, xe, ye) = pts[0], pts[-1]
    disp = math.hypot(xe - xs, ye - ys)
    dur = max(1e-6, te - ts)
    # Screen coords: y grows downward, so negate dy for an intuitive heading
    # (0deg = moving right, 90deg = moving up).
    direction = math.degrees(math.atan2(-(ye - ys), xe - xs))
    return {
        "distance_px": round(dist, 1),
        "displacement_px": round(disp, 1),
        "avg_speed_px_s": round(dist / dur, 1),
        "direction_deg": round(direction, 1),
        "duration_sec": round(te - ts, 3),
    }


def _line_crossings(tracks: Dict[int, Dict], x_line: float) -> Dict:
    """Count tracks crossing a vertical line at ``x_line``, by direction + class."""
    l2r = r2l = 0
    by_class: Dict[str, Dict[str, int]] = {}
    for tr in tracks.values():
        xs = [cx for _, cx, _ in tr["pts"]]
        if len(xs) < 2:
            continue
        # side sequence relative to the line; ignore points sitting exactly on it
        sides = [1 if x >= x_line else -1 for x in xs]
        crossed_l2r = any(a == -1 and b == 1 for a, b in zip(sides, sides[1:]))
        crossed_r2l = any(a == 1 and b == -1 for a, b in zip(sides, sides[1:]))
        cls = tr["cls"]
        bc = by_class.setdefault(cls, {"l2r": 0, "r2l": 0})
        if crossed_l2r:
            l2r += 1; bc["l2r"] += 1
        if crossed_r2l:
            r2l += 1; bc["r2l"] += 1
    return {"l2r": l2r, "r2l": r2l, "total": l2r + r2l, "by_class": by_class}


def occupancy_heatmap(result: TrackingResult) -> np.ndarray:
    """A JET-colormapped image of where activity concentrated (foot points)."""
    W = max(1, result.width)
    H = max(1, result.height)
    out_w = min(W, 640)
    scale = out_w / W
    out_h = max(1, int(round(H * scale)))

    acc = np.zeros((out_h, out_w), dtype=np.float32)
    import cv2
    for fr in result.frames:
        for d in fr.dets:
            fx, fy = _foot_point(d.xyxy)
            px, py = int(fx * scale), int(fy * scale)
            if 0 <= px < out_w and 0 <= py < out_h:
                cv2.circle(acc, (px, py), 6, 1.0, -1)
    if acc.max() > 0:
        acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=9)
        acc = acc / acc.max()
    heat = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Damp the cold (zero-activity) regions toward black so it reads as a heatmap.
    mask = (acc > 0.02)[..., None]
    heat = (heat * mask).astype(np.uint8)
    return heat


def build(result: TrackingResult) -> Dict:
    """Numeric analytics summary (trajectories are drawn client-side from boxes)."""
    tracks = _collect_tracks(result)
    x_line = result.width * config.ZONE_LINE_FRAC

    track_stats = {}
    for tid, tr in tracks.items():
        s = _track_stats(tr["pts"])
        s["cls"] = tr["cls"]
        if tr["actions"]:
            # dominant (most frequent) action over the track's lifetime
            s["action"] = max(set(tr["actions"]), key=tr["actions"].count)
        if tr.get("attrs"):
            s["attrs"] = tr["attrs"]
        track_stats[str(tid)] = s

    return {
        "line": {"orientation": "vertical",
                 "pos_frac": config.ZONE_LINE_FRAC,
                 "x": round(x_line, 1)},
        "crossings": _line_crossings(tracks, x_line),
        "tracks": track_stats,
    }
