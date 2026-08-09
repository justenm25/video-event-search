"""Central configuration for the Intelligent Video Event Search Engine.

The system detects objects (COCO-80 classes) in a video, tracks each one across
frames to give it a persistent ID, and folds those tracks into time intervals:
one row per object appearance, with a start and end timestamp.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"        # uploaded / source videos
PROCESSED_DIR = DATA_DIR / "processed"  # per-video artifacts (detections + intervals)

# --- Detection + tracking model (Ultralytics YOLO) ---
# yolo11n < yolo11s < yolo11m < yolo11l < yolo11x  (speed -> accuracy).
# Weights auto-download on first run. yolo11s is a good speed/accuracy balance;
# bump to yolo11m/l for a nicer demo on the RTX 5060.
YOLO_MODEL = "yolo11s.pt"

# Built-in ByteTrack tracker config shipped with Ultralytics.
TRACKER = "bytetrack.yaml"

# Minimum detection confidence to keep a box.
CONF_THRESHOLD = 0.35

# Process every Nth frame. 1 = every frame (best tracking, slowest).
# 2-3 is a good balance for most footage.
FRAME_STRIDE = 2

# --- Interval building ---
# When an object disappears, wait this many SECONDS of continuous absence before
# closing its interval. Absorbs brief occlusions / missed detections so a single
# car behind a pole doesn't get split into many rows. This is the "gap tolerance".
INTERVAL_GAP_SEC = 0.6

# Drop appearances shorter than this (seconds) as noise.
MIN_INTERVAL_SEC = 0.3

# --- Server ---
HOST = "127.0.0.1"
PORT = 8000
