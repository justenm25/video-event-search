"""Run YOLO detection + tracking over a video, with robustness passes.

Every kept box carries a persistent ``track_id`` (from the tracker), so "the same
car" keeps one ID across frames. That ID is what lets us later fold the per-frame
boxes into start/end time intervals (see ``intervals.py``) — and it's also what
makes the robustness passes here possible:

* **Low-light enhancement** — dark/back-lit frames are CLAHE-equalized before
  detection so silhouettes become recognizable (fixes "dog silhouette -> person").
* **Per-class confidence floors** — error-prone classes (``person``) must clear a
  higher bar; marginal low-confidence guesses are dropped.
* **Ghost-track removal** — a track seen in only a frame or two is discarded.
* **Dominant-class voting** — each track is relabeled to the class it was called
  most often (confidence-weighted), so a momentary flicker to the wrong class is
  corrected across the whole appearance.
* **Face identity naming** — person tracks matched to an enrolled face are renamed.

Because we preprocess frames, detection runs through a manual capture loop with
``model.track(..., persist=True)`` rather than handing the file straight to
Ultralytics.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

import config


@dataclass
class Detection:
    track_id: int
    cls: int
    name: str
    conf: float
    xyxy: List[float]  # [x1, y1, x2, y2] in original-frame pixels
    mask: Optional[List] = None    # simplified polygon [[x,y],...] (segmentation)
    kpts: Optional[List] = None    # [[x,y,conf]*17] (pose)
    action: Optional[str] = None   # coarse action label (pose)
    depth: Optional[float] = None  # relative distance 0=far .. 1=near (MiDaS)
    attrs: Optional[dict] = None   # {"age":int,"gender":"M"|"F"} (InsightFace)


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


# --- Perception: low-light / back-light enhancement ---
_CLAHE = None


def _enhance(image):
    """CLAHE-equalize the luminance of dark frames; pass bright frames through.

    A silhouette is a contrast failure — the object is there but crushed into
    shadow. Adaptive histogram equalization on the L channel pulls that structure
    back so the detector classifies it correctly. Auto-gated on mean brightness so
    well-lit footage is left untouched.
    """
    global _CLAHE
    if float(image.mean()) >= config.ENHANCE_LUMA_THRESH:
        return image
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _passes_floor(name: str, conf: float) -> bool:
    """Whether ``conf`` clears the confidence floor configured for ``name``."""
    return conf >= config.CLASS_CONF_FLOORS.get(name, config.CONF_FLOOR_DEFAULT)


# --- Segmentation + pose helpers ---

def _simplify_poly(poly, max_pts: int = 32) -> Optional[List[List[int]]]:
    """Downsample a mask contour (Nx2) to at most ``max_pts`` integer points."""
    n = len(poly)
    if n == 0:
        return None
    step = max(1, n // max_pts)
    pts = [[int(x), int(y)] for x, y in poly[::step]]
    return pts if len(pts) >= 3 else None


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _attach_pose(pose_model, image, frame: "Frame") -> None:
    """Run the pose model on ``image`` and attach keypoints + action to person dets."""
    persons = [d for d in frame.dets if d.name == "person"]
    if not persons:
        return
    rp = pose_model.predict(image, imgsz=config.IMG_SIZE, conf=0.3, verbose=False)[0]
    if rp.keypoints is None or rp.boxes is None or len(rp.boxes) == 0:
        return
    from .pose import classify_action

    pboxes = rp.boxes.xyxy.tolist()
    kxy = rp.keypoints.xy.tolist()          # P x 17 x 2
    kcf = rp.keypoints.conf
    kcf = kcf.tolist() if kcf is not None else [[1.0] * 17 for _ in kxy]

    used = set()
    for d in persons:
        best, bi = 0.3, -1  # require IoU > 0.3 to accept an association
        for i, pb in enumerate(pboxes):
            if i in used:
                continue
            iou = _iou(d.xyxy, pb)
            if iou > best:
                best, bi = iou, i
        if bi >= 0:
            used.add(bi)
            kp = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                  for (x, y), c in zip(kxy[bi], kcf[bi])]
            d.kpts = kp
            d.action = classify_action(kp, d.xyxy)


# --- Custom identity recognition (relabel "person" -> an enrolled name) ---
# We collect a few of the largest, most confident face-bearing crops per person
# track during the main loop, then embed + match + majority-vote *once per track*
# afterward. This keeps face inference cost bounded (a handful of runs per person,
# never per frame) and gives a single stable name for the whole appearance.

def _collect_crop(
    store: Dict[int, List[Tuple[float, object]]],
    tid: int,
    img,
    box: List[float],
    conf: float,
) -> None:
    """Keep the top ``FACE_CROPS_PER_TRACK`` crops (by area*conf) for track ``tid``."""
    if img is None:
        return
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1 = max(0, min(x1, w_img - 1)); x2 = max(0, min(x2, w_img))
    y1 = max(0, min(y1, h_img - 1)); y2 = max(0, min(y2, h_img))
    w, h = x2 - x1, y2 - y1
    if w < 40 or h < 60:  # too small for a reliable face embedding
        return
    lst = store.setdefault(tid, [])
    lst.append((w * h * conf, img[y1:y2, x1:x2].copy()))
    if len(lst) > config.FACE_CROPS_PER_TRACK:
        lst.sort(key=lambda p: p[0], reverse=True)
        del lst[config.FACE_CROPS_PER_TRACK:]


def _name_tracks(store: Dict[int, List[Tuple[float, object]]], gallery) -> Dict[int, str]:
    """Embed each track's crops, match to the gallery, and majority-vote a name."""
    from . import identities
    from .faces import embed_face

    names: Dict[int, str] = {}
    for tid, crops in store.items():
        votes: Counter = Counter()
        for _, crop in sorted(crops, key=lambda p: p[0], reverse=True):
            emb = embed_face(crop)
            if emb is None:
                continue
            name, _score = identities.match(emb, gallery)
            if name:
                votes[name] += 1
        if votes:
            name, count = votes.most_common(1)[0]
            if count >= config.FACE_MIN_VOTES:
                names[tid] = name
    return names


def _reid_propagate(person_crops, valid, track_cls, track_names) -> None:
    """Spread face-derived names to unnamed person tracks via appearance similarity."""
    from . import reid

    ptids = [tid for tid in valid
             if track_cls.get(tid) == "person" and person_crops.get(tid)]
    embs = {}
    for tid in ptids:
        e = reid.mean_embedding([c for _, c in person_crops[tid]])
        if e is not None:
            embs[tid] = e
    named = [(tid, embs[tid]) for tid in ptids if tid in track_names and tid in embs]
    if not named:
        return
    for tid in ptids:
        if tid in track_names or tid not in embs:
            continue
        best, best_name = 0.0, None
        for ntid, ne in named:
            s = float(embs[tid] @ ne)
            if s > best:
                best, best_name = s, track_names[ntid]
        if best_name and best >= config.REID_THRESHOLD:
            track_names[tid] = best_name


def _track_attributes(person_crops, valid, track_cls) -> Dict[int, dict]:
    """Estimate age/gender once per person track from its best crops."""
    from . import attributes

    out: Dict[int, dict] = {}
    for tid in valid:
        if track_cls.get(tid) != "person" or not person_crops.get(tid):
            continue
        best = [c for _, c in sorted(person_crops[tid], key=lambda p: p[0], reverse=True)[:3]]
        res = attributes.analyze_best(best)
        if res:
            out[tid] = res
    return out


def run_tracking(
    video_path: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> TrackingResult:
    """Track objects through ``video_path`` with the robustness passes applied.

    ``progress_cb(done, total)`` is called after each processed frame so a UI
    can show a progress bar.
    """
    # Imported lazily so the module (and CLI --help) load without torch present.
    from ultralytics import YOLO

    meta = _video_meta(video_path)
    fps = meta["fps"] or 30.0
    stride = max(1, config.FRAME_STRIDE)
    total_to_process = max(1, meta["total"] // stride) if meta["total"] else 0

    # Face recognition is active only when enabled AND someone is enrolled.
    gallery = None
    face_enabled = False
    if config.FACE_RECOGNITION:
        from . import identities
        gallery = identities.load_gallery()
        face_enabled = gallery.embeddings.shape[0] > 0

    # Person crops feed face recognition, Re-ID, and attribute estimation.
    need_person_crops = face_enabled or config.REID_ENABLED or config.ATTRIBUTES_ENABLED

    # Segmentation swaps the main model for the seg variant (boxes + masks in one
    # pass). Pose runs as an optional second model, associated to person tracks.
    model = YOLO(config.SEG_MODEL if config.SEGMENTATION else config.YOLO_MODEL)
    pose_model = YOLO(config.POSE_MODEL) if config.POSE_ENABLED else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    person_crops: Dict[int, List[Tuple[float, object]]] = {}
    cls_votes: Dict[int, Dict[str, float]] = {}  # track -> {class: summed conf}
    cls_ids: Dict[str, int] = {}                 # class name -> its COCO id
    track_count: Dict[int, int] = {}             # track -> kept detection count

    # Depth runs on a handful of sampled frames, not every frame.
    depth_stash: List[Tuple["Frame", object]] = []
    depth_every = max(1, (total_to_process or 1) // max(1, config.DEPTH_SAMPLES))

    frames: List[Frame] = []
    frame_idx = -1
    processed = 0
    try:
        while True:
            if not cap.grab():
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue
            ok, image = cap.retrieve()
            if not ok:
                break
            if config.ENHANCE_LOWLIGHT:
                image = _enhance(image)

            r = model.track(
                image,
                persist=True,            # keep tracker state across our manual calls
                tracker=config.TRACKER,
                conf=config.CONF_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                imgsz=config.IMG_SIZE,
                augment=config.USE_TTA,
                verbose=False,
            )[0]

            frame = Frame(frame_idx=frame_idx, t=round(frame_idx / fps, 3))
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                names_map = r.names
                ids = boxes.id.int().tolist()
                clss = boxes.cls.int().tolist()
                confs = boxes.conf.tolist()
                xyxys = boxes.xyxy.tolist()
                masks_xy = r.masks.xy if (config.SEGMENTATION and r.masks is not None) else None
                for idx, (tid, c, cf, box) in enumerate(zip(ids, clss, confs, xyxys)):
                    name = str(names_map[int(c)])
                    cf = round(float(cf), 3)
                    if not _passes_floor(name, cf):
                        continue
                    tid = int(tid)
                    det = Detection(
                        track_id=tid,
                        cls=int(c),
                        name=name,
                        conf=cf,
                        xyxy=[round(float(v), 1) for v in box],
                    )
                    if masks_xy is not None and idx < len(masks_xy):
                        det.mask = _simplify_poly(masks_xy[idx])
                    frame.dets.append(det)
                    track_count[tid] = track_count.get(tid, 0) + 1
                    votes = cls_votes.setdefault(tid, {})
                    votes[name] = votes.get(name, 0.0) + cf
                    cls_ids[name] = int(c)
                    if need_person_crops and name == "person":
                        _collect_crop(person_crops, tid, image, box, cf)
            if pose_model is not None:
                _attach_pose(pose_model, image, frame)
            if config.DEPTH_ENABLED and processed % depth_every == 0 \
                    and len(depth_stash) < config.DEPTH_SAMPLES + 2:
                depth_stash.append((frame, image.copy()))
            frames.append(frame)
            processed += 1
            if progress_cb:
                progress_cb(processed, total_to_process or processed)
    finally:
        cap.release()

    # --- ghost-track removal + dominant-class voting ---
    valid = {tid for tid, n in track_count.items() if n >= config.MIN_TRACK_DETECTIONS}
    track_cls = {
        tid: max(cls_votes[tid].items(), key=lambda kv: kv[1])[0]
        for tid in valid if cls_votes.get(tid)
    }

    # --- face naming (only for valid, person-dominant tracks) ---
    track_names: Dict[int, str] = {}
    if face_enabled and person_crops:
        person_tracks = {
            tid: crops for tid, crops in person_crops.items()
            if tid in valid and track_cls.get(tid) == "person"
        }
        if person_tracks:
            track_names = _name_tracks(person_tracks, gallery)

    # --- appearance Re-ID: carry names onto unnamed person tracks ---
    if config.REID_ENABLED and track_names and person_crops:
        _reid_propagate(person_crops, valid, track_cls, track_names)

    # --- face attributes (age/gender) per person track ---
    track_attrs: Dict[int, dict] = {}
    if config.ATTRIBUTES_ENABLED and person_crops:
        track_attrs = _track_attributes(person_crops, valid, track_cls)

    # --- relative depth per track (median over sampled frames) ---
    track_depth: Dict[int, float] = {}
    if config.DEPTH_ENABLED and depth_stash:
        import numpy as np
        from . import depth as depthmod
        samples: Dict[int, List[float]] = {}
        for fr, img in depth_stash:
            dm = depthmod.infer(img)
            for det in fr.dets:
                v = depthmod.box_median(dm, det.xyxy)
                if v is not None:
                    samples.setdefault(det.track_id, []).append(v)
        track_depth = {tid: round(float(np.median(vs)), 3) for tid, vs in samples.items()}

    # --- rewrite every detection: drop ghosts, apply voted class or identity ---
    for frame in frames:
        kept: List[Detection] = []
        for det in frame.dets:
            if det.track_id not in valid:
                continue
            label = track_names.get(det.track_id) or track_cls.get(det.track_id, det.name)
            det.name = label
            if label in cls_ids:
                det.cls = cls_ids[label]
            if det.track_id in track_depth:
                det.depth = track_depth[det.track_id]
            if det.track_id in track_attrs:
                det.attrs = track_attrs[det.track_id]
            kept.append(det)
        frame.dets = kept

    duration = frames[-1].t if frames else 0.0
    return TrackingResult(
        fps=round(fps, 3),
        width=meta["width"],
        height=meta["height"],
        total_frames=meta["total"],
        duration=round(duration, 3),
        frames=frames,
    )
