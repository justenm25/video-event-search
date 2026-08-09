# Intelligent Video Event Search Engine

CS 5330: Pattern Recognition and Computer Vision

**Team:** Nikhil Singh Shekhawat, Nimish Vivek Poonekar

## What this is
Drop in any video and the system scans it, detects objects (people, cars, chairs,
bicycles… the 80 COCO classes), tracks each one across frames, and tells you
**when** each object was on screen — e.g. *"car #7 from 00:03 → 00:11."* A
Jarvis-style HUD draws live tracking reticles over the video and lets you search
by object type to jump straight to those moments.

## How it works
```
video ──▶ YOLO11 detection ──▶ ByteTrack tracking ──▶ interval builder ──▶ CSV + JSON ──▶ HUD
          (what's in frame)     (persistent per-       (start/end per
                                 object IDs)            object appearance)
```

1. **Detect** objects in each processed frame (`src/video_search/detector.py`).
2. **Track** with ByteTrack so "the same car" keeps one `track_id` across frames.
3. **Build intervals** (`src/video_search/intervals.py`): as frames are scanned in
   order, each track keeps a running end time that's pushed forward while it stays
   visible. When it's been gone longer than a small **gap tolerance**
   (`INTERVAL_GAP_SEC`), the interval is finalized. Result: **one row per object
   appearance**, not per frame:

   ```
   track_id, cls,    start_time,   end_time,     duration_sec, max_conf
   7,        car,    00:00:03.000, 00:00:11.000, 8.0,          0.94
   12,       person, 00:00:05.500, 00:00:29.000, 23.5,         0.97
   ```

4. **Search** the intervals by class and jump to the matching timestamps in the HUD.

## Project layout
```
config.py                    # model, thresholds, frame stride, gap tolerance, server
src/video_search/
  detector.py                # YOLO11 + ByteTrack -> per-frame tracked boxes
  intervals.py               # fold tracks into start/end intervals + write CSV
  pipeline.py                # video -> detections.json + intervals.csv + summary.json
  storage.py                 # artifact paths per video
backend/
  app.py                     # FastAPI: upload / process / status / search / serve
  static/                    # the HUD (index.html, style.css, app.js)
scripts/
  process_video.py           # CLI: process a video headless (no UI)
data/
  videos/                    # source / uploaded videos
  processed/<id>/            # detections.json, intervals.csv, summary.json
```

## Setup

> Your machine has an **RTX 5060 (Blackwell)** GPU. It needs a recent CUDA build
> of PyTorch (`cu128`). Install PyTorch *first*, separately from the rest.

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell

# 2. Install PyTorch with the Blackwell-compatible CUDA build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install the rest
pip install -r requirements.txt
```

Verify the GPU is visible (Ultralytics will then run detection on it automatically):
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The YOLO11 weights download automatically the first time you process a video.

## Usage

**Web HUD (main experience):**
```bash
python backend/app.py
# open http://127.0.0.1:8000  -> drop a video -> watch it scan -> search "car"
```

**Headless (for testing the pipeline):**
```bash
python scripts/process_video.py data/videos/sample.mp4
# -> data/processed/<id>/intervals.csv
```

Tune detection in `config.py`: `YOLO_MODEL` (accuracy vs speed), `FRAME_STRIDE`
(process every Nth frame), `CONF_THRESHOLD`, and `INTERVAL_GAP_SEC` (how long an
object can vanish before its interval is closed).

## Roadmap
- [x] Detection + tracking → per-object time intervals *(current)*
- [x] Jarvis-style HUD with live tracking reticles + searchable timeline
- [ ] Object counting overlays (live per-class counts) — partially in the HUD
- [ ] Open-vocabulary queries (YOLO-World / CLIP) for objects outside COCO-80
- [ ] Export matched **clips** (not just timestamps) to disk
- [ ] Multi-video search across a whole library
- [ ] Evaluation on a labeled dataset (subset of ActivityNet / Kinetics)
