"""Where each processed video's artifacts live on disk.

For a given ``video_id`` we store three files under ``data/processed/<video_id>/``:

* ``detections.json`` — per-frame tracked boxes (drives the live HUD overlay)
* ``intervals.csv``    — one row per object appearance (start/end timestamps)
* ``summary.json``     — video metadata + per-class counts (drives the dashboard)
"""
from __future__ import annotations

from pathlib import Path

import config


def processed_dir(video_id: str) -> Path:
    d = config.PROCESSED_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def detections_path(video_id: str) -> Path:
    return processed_dir(video_id) / "detections.json"


def intervals_path(video_id: str) -> Path:
    return processed_dir(video_id) / "intervals.csv"


def summary_path(video_id: str) -> Path:
    return processed_dir(video_id) / "summary.json"


def is_processed(video_id: str) -> bool:
    return summary_path(video_id).exists() and detections_path(video_id).exists()
