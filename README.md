# Intelligent Video Event Search Engine

**Course:** CS 5330 — Pattern Recognition and Computer Vision

**Team members:** Nikhil Singh Shekhawat, Nimish Vivek Poonekar

## Project description
Drop in any video and the system scans it, detects objects (people, cars, boats,
chairs, bicycles… the 80 COCO classes), tracks each one across frames so it keeps
a persistent identity, and folds those tracks into **time intervals** — one row
per appearance with a start and end timestamp (e.g. *"car #7 from 00:03 →
00:11."*). An editor-style web interface draws live tracking overlays on the video
and lets you **search by object type** to jump straight to the moments an object
is on screen.

Around this core the system adds robustness and recognition passes: adaptive
low-light (CLAHE) enhancement, per-class confidence floors, ghost-track removal,
and confidence-weighted class voting clean up the detections; an optional
**face-recognition** pass relabels the generic "person" class with an enrolled
name; and optional passes add segmentation masks, pose-based action cues,
monocular depth, appearance re-identification, and age/gender attributes. A
quantitative evaluation harness reports precision/recall/F1, mAP@0.5, MOTA, IDF1,
and identity accuracy against labeled ground truth.

## Links
- **Demo video (Google Drive):** https://drive.google.com/file/d/1-yKu1CdBvjg-rMZh3NFh8-Cxryevda8N/view?usp=sharing
- **Report (IEEE format):** [`report/Video_Event_Search_Report.pdf`](report/Video_Event_Search_Report.pdf)

> Per the assignment, videos and datasets are **not** submitted to Gradescope —
> the demo video is hosted on Google Drive and linked above.

## How it works
```
video ──▶ YOLO11 detection ──▶ ByteTrack tracking ──▶ interval builder ──▶ CSV + JSON ──▶ web UI
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

4. **Search** the intervals by class and jump to the matching timestamps in the UI.

## Project layout
```
config.py                    # model, thresholds, frame stride, gap tolerance, server
src/video_search/
  detector.py                # YOLO11 + ByteTrack -> per-frame tracked boxes
  intervals.py               # fold tracks into start/end intervals + write CSV
  pipeline.py                # video -> detections.json + intervals.csv + summary.json
  faces.py / identities.py   # face embeddings + enrolled-identity gallery + matching
  pose.py / depth.py         # optional: pose->action, MiDaS relative depth
  reid.py / attributes.py    # optional: appearance re-ID, age/gender
  analytics.py               # trajectories, occupancy heatmap, line crossings
  evaluation.py              # precision/recall/mAP, MOTA/IDF1, identity accuracy
  storage.py                 # artifact paths per video
backend/
  app.py                     # FastAPI: upload / process / status / search / serve
  static/                    # the web UI (index.html, style.css, app.js)
scripts/
  process_video.py           # CLI: process a video headless (no UI)
report/
  main.tex, figs/            # IEEE-format report source
  Video_Event_Search_Report.pdf
data/
  videos/                    # source / uploaded videos (not submitted)
  processed/<id>/            # detections.json, intervals.csv, summary.json
```

## Setup

> Our machine has an **RTX 5060 (Blackwell)** GPU. It needs a recent CUDA build
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

**Web UI (main experience):**
```bash
python backend/app.py
# open http://127.0.0.1:8000  -> import a video -> watch it scan -> search "car"
```

**Headless (for testing the pipeline):**
```bash
python scripts/process_video.py data/videos/sample.mp4
# -> data/processed/<id>/intervals.csv
```

Tune detection in `config.py`: `YOLO_MODEL` (accuracy vs speed), `FRAME_STRIDE`
(process every Nth frame), `CONF_THRESHOLD`, and `INTERVAL_GAP_SEC` (how long an
object can vanish before its interval is closed). The optional recognition passes
(`FACE_RECOGNITION`, `POSE_ENABLED`, `DEPTH_ENABLED`, `REID_ENABLED`,
`ATTRIBUTES_ENABLED`, `SEGMENTATION`) are also toggled there.

## Building the report
The report is written in LaTeX (IEEE `IEEEtran` conference class). With any LaTeX
engine, e.g. Tectonic:
```bash
tectonic report/main.tex
```
The figures in `report/figs/` are generated from the processed corpus.

## Roadmap
- [x] Detection + tracking → per-object time intervals
- [x] Editor-style web UI with live tracking overlays + searchable timeline
- [x] Custom identities via face recognition; optional pose / depth / re-ID / attributes
- [x] Quantitative evaluation harness (detection / tracking / identity metrics)
- [ ] Evaluation on a labeled benchmark (subset of ActivityNet / Kinetics)
- [ ] Export matched **clips** (not just timestamps); multi-video library search
- [ ] Open-vocabulary queries (YOLO-World / CLIP) for objects outside COCO-80
