"""FastAPI backend for the Video Event Search HUD.

Endpoints
    POST /api/upload              upload a video, get back a video_id
    POST /api/process/{id}        kick off detection+tracking in the background
    GET  /api/status/{id}         progress of the processing job
    GET  /api/summary/{id}        metadata + per-class counts + intervals
    GET  /api/search/{id}?q=car   intervals whose class matches the query
    GET  /api/detections/{id}     per-frame boxes (drives the live overlay)
    GET  /api/video/{id}          the source video (with HTTP range support)
    GET  /api/videos              list processed videos
"""
from __future__ import annotations

import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

# --- make `config` and the `video_search` package importable ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from video_search import pipeline, storage, identities  # noqa: E402

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from typing import List  # noqa: E402
import json  # noqa: E402

app = FastAPI(title="Video Event Search Engine")

config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
config.IDENTITIES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry: video_id -> progress state.
JOBS: Dict[str, Dict] = {}


def _find_video_file(video_id: str) -> Optional[Path]:
    matches = list(config.VIDEO_DIR.glob(f"{video_id}.*"))
    return matches[0] if matches else None


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_id = uuid.uuid4().hex[:12]
    dest = config.VIDEO_DIR / f"{video_id}{ext}"
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    JOBS[video_id] = {"state": "uploaded", "done": 0, "total": 0,
                      "filename": file.filename}
    return {"video_id": video_id, "filename": file.filename}


def _run_job(video_id: str, video_path: Path) -> None:
    job = JOBS[video_id]
    job.update(state="processing", done=0, total=0)

    def progress(done: int, total: int) -> None:
        job["done"], job["total"] = done, total

    try:
        summary = pipeline.process_video(video_path, video_id, progress_cb=progress)
        job.update(state="done", num_intervals=summary["num_intervals"])
    except Exception as exc:  # surface the error to the UI instead of dying silently
        job.update(state="error", error=str(exc))


@app.post("/api/process/{video_id}")
def process(video_id: str):
    video_path = _find_video_file(video_id)
    if video_path is None:
        raise HTTPException(404, "video not found")
    if JOBS.get(video_id, {}).get("state") == "processing":
        return {"started": False, "reason": "already processing"}
    JOBS.setdefault(video_id, {})
    threading.Thread(target=_run_job, args=(video_id, video_path), daemon=True).start()
    return {"started": True}


@app.get("/api/status/{video_id}")
def status(video_id: str):
    if storage.is_processed(video_id) and JOBS.get(video_id, {}).get("state") != "processing":
        return {"state": "done", **JOBS.get(video_id, {})}
    job = JOBS.get(video_id)
    if job is None:
        raise HTTPException(404, "unknown video")
    return job


@app.get("/api/summary/{video_id}")
def summary(video_id: str):
    path = storage.summary_path(video_id)
    if not path.exists():
        raise HTTPException(404, "not processed yet")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/search/{video_id}")
def search(video_id: str, q: str):
    path = storage.summary_path(video_id)
    if not path.exists():
        raise HTTPException(404, "not processed yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    q = q.strip().lower()
    matches = [iv for iv in data["intervals"] if q in iv["cls"].lower()] if q else []
    return {"query": q, "count": len(matches), "matches": matches}


@app.get("/api/detections/{video_id}")
def detections(video_id: str):
    path = storage.detections_path(video_id)
    if not path.exists():
        raise HTTPException(404, "not processed yet")
    return FileResponse(path, media_type="application/json")


@app.get("/api/heatmap/{video_id}")
def heatmap(video_id: str):
    path = storage.heatmap_path(video_id)
    if not path.exists():
        raise HTTPException(404, "no heatmap")
    return FileResponse(path, media_type="image/png")


@app.get("/api/depth/{video_id}")
def depth(video_id: str):
    path = storage.depth_path(video_id)
    if not path.exists():
        raise HTTPException(404, "no depth map")
    return FileResponse(path, media_type="image/png")


@app.post("/api/evaluate/{video_id}")
async def evaluate_video(video_id: str, file: UploadFile = File(...)):
    """Score stored predictions against an uploaded ground-truth JSON file."""
    det_path = storage.detections_path(video_id)
    if not det_path.exists():
        raise HTTPException(404, "not processed yet")
    try:
        gt = json.loads(await file.read())
    except Exception:
        raise HTTPException(400, "ground-truth file must be valid JSON")
    from video_search import evaluation
    pred = json.loads(det_path.read_text(encoding="utf-8"))
    try:
        return evaluation.evaluate(pred, gt)
    except Exception as exc:
        raise HTTPException(400, f"evaluation failed: {exc}")


@app.get("/api/info")
def info():
    """System/model info for the HUD status readouts."""
    device = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "model": config.YOLO_MODEL,
        "imgsz": config.IMG_SIZE,
        "conf": config.CONF_THRESHOLD,
        "stride": config.FRAME_STRIDE,
        "device": device,
    }


@app.get("/api/videos")
def videos():
    out = []
    for summ in config.PROCESSED_DIR.glob("*/summary.json"):
        data = json.loads(summ.read_text(encoding="utf-8"))
        out.append({"video_id": data["video_id"], "duration": data["duration"],
                    "num_intervals": data["num_intervals"],
                    "classes": list(data["classes"].keys())})
    return {"videos": out}


@app.get("/api/video/{video_id}")
def video(video_id: str, request: Request):
    path = _find_video_file(video_id)
    if path is None:
        raise HTTPException(404, "video not found")
    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    if range_header is None:
        return FileResponse(path, media_type="video/mp4")

    # Serve a byte range so the <video> element can seek.
    start_s, _, end_s = range_header.replace("bytes=", "").partition("-")
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else file_size - 1
    end = min(end, file_size - 1)
    length = end - start + 1
    with path.open("rb") as f:
        f.seek(start)
        chunk = f.read(length)
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return Response(chunk, status_code=206, headers=headers, media_type="video/mp4")


# --- Custom identities (relabel "person" as an enrolled name) ---

@app.post("/api/identities")
async def create_identity(name: str = Form(...), files: List[UploadFile] = File(...)):
    """Enroll a person: a name + a batch of photos of their face."""
    images = [await f.read() for f in files]
    try:
        meta = identities.create_identity(name, images)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return meta


@app.get("/api/identities")
def list_identities():
    return {"identities": identities.list_identities()}


@app.delete("/api/identities/{identity_id}")
def delete_identity(identity_id: str):
    if not identities.delete_identity(identity_id):
        raise HTTPException(404, "identity not found")
    return {"deleted": True}


@app.get("/api/identities/{identity_id}/thumb")
def identity_thumb(identity_id: str):
    path = identities.thumb_path(identity_id)
    if path is None:
        raise HTTPException(404, "no thumbnail")
    return FileResponse(path, media_type="image/jpeg")


# Static front-end (index.html at /). Mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
          name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=False)
