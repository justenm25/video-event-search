"""Enrolled identities: the on-disk gallery of named faces + matching.

An *identity* is a name plus one or more face embeddings (see ``faces.py``),
built from photos the user uploads. Each identity lives in its own directory
under ``config.IDENTITIES_DIR``:

    data/identities/<id>/
        meta.json        {id, name, created, num_photos, num_faces}
        embeddings.npy   float32 array, num_faces x 512 (L2-normalized)
        thumb.jpg        small preview of one enrolled photo (for the UI)

At analysis time ``detector.py`` embeds a person crop and calls ``match`` to find
the best identity; if the cosine similarity clears ``FACE_MATCH_THRESHOLD`` the
person's track is relabeled with that name.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from .faces import embed_face


@dataclass
class Gallery:
    embeddings: np.ndarray  # (M, 512) float32, L2-normalized; empty (0, 512) if none
    names: List[str]        # length M, name for each row


# Cached gallery, invalidated when IDENTITIES_DIR's mtime changes (add/remove of
# an identity) or explicitly after a write below.
_GALLERY_CACHE: Optional[Tuple[float, Gallery]] = None


def _identity_dir(identity_id: str) -> Path:
    return config.IDENTITIES_DIR / identity_id


def _read_meta(d: Path) -> Optional[Dict]:
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _decode_image(data: bytes) -> Optional[np.ndarray]:
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR, or None if not an image


def create_identity(name: str, images: List[bytes]) -> Dict:
    """Enroll a new identity ``name`` from raw image bytes.

    Embeds each photo, keeps the ones with a detectable face, and persists the
    embeddings + a thumbnail. Raises ``ValueError`` if the name is blank or no
    face could be found in any photo.
    """
    import cv2

    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    if not images:
        raise ValueError("no photos provided")

    embeddings: List[np.ndarray] = []
    thumb_bgr: Optional[np.ndarray] = None
    for data in images:
        img = _decode_image(data)
        if img is None:
            continue
        emb = embed_face(img)
        if emb is None:
            continue
        embeddings.append(emb)
        if thumb_bgr is None:
            thumb_bgr = img

    if not embeddings:
        raise ValueError(
            "No faces detected in the uploaded photos. Use clear, well-lit "
            "photos where the face is visible and reasonably large."
        )

    identity_id = uuid.uuid4().hex[:12]
    d = _identity_dir(identity_id)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embeddings.npy", np.stack(embeddings).astype(np.float32))

    # Thumbnail: longest side capped at 160px, saved as jpg.
    if thumb_bgr is not None:
        h, w = thumb_bgr.shape[:2]
        scale = 160.0 / max(h, w)
        if scale < 1.0:
            thumb_bgr = cv2.resize(thumb_bgr, (int(w * scale), int(h * scale)))
        cv2.imwrite(str(d / "thumb.jpg"), thumb_bgr)

    meta = {
        "id": identity_id,
        "name": name,
        "created": time.time(),
        "num_photos": len(images),
        "num_faces": len(embeddings),
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    _invalidate()
    return meta


def list_identities() -> List[Dict]:
    """All enrolled identities, newest first, each with a ``has_thumb`` flag."""
    if not config.IDENTITIES_DIR.exists():
        return []
    out: List[Dict] = []
    for d in config.IDENTITIES_DIR.iterdir():
        if not d.is_dir():
            continue
        meta = _read_meta(d)
        if meta is None:
            continue
        meta = dict(meta)
        meta["has_thumb"] = (d / "thumb.jpg").exists()
        out.append(meta)
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


def thumb_path(identity_id: str) -> Optional[Path]:
    p = _identity_dir(identity_id) / "thumb.jpg"
    return p if p.exists() else None


def delete_identity(identity_id: str) -> bool:
    import shutil

    d = _identity_dir(identity_id)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    _invalidate()
    return True


def _invalidate() -> None:
    global _GALLERY_CACHE
    _GALLERY_CACHE = None


def load_gallery() -> Gallery:
    """All identities' embeddings stacked into one matrix (cached)."""
    global _GALLERY_CACHE
    mtime = config.IDENTITIES_DIR.stat().st_mtime if config.IDENTITIES_DIR.exists() else 0.0
    if _GALLERY_CACHE is not None and _GALLERY_CACHE[0] == mtime:
        return _GALLERY_CACHE[1]

    rows: List[np.ndarray] = []
    names: List[str] = []
    if config.IDENTITIES_DIR.exists():
        for d in config.IDENTITIES_DIR.iterdir():
            if not d.is_dir():
                continue
            meta = _read_meta(d)
            emb_path = d / "embeddings.npy"
            if meta is None or not emb_path.exists():
                continue
            try:
                embs = np.load(emb_path)
            except Exception:
                continue
            for row in embs:
                rows.append(row.astype(np.float32))
                names.append(meta["name"])

    embeddings = np.stack(rows) if rows else np.zeros((0, 512), dtype=np.float32)
    gallery = Gallery(embeddings=embeddings, names=names)
    _GALLERY_CACHE = (mtime, gallery)
    return gallery


def match(
    embedding: np.ndarray, gallery: Optional[Gallery] = None
) -> Tuple[Optional[str], float]:
    """Best-matching enrolled name for ``embedding`` (a normalized 512-vector).

    Returns ``(name, score)`` where ``score`` is cosine similarity, or
    ``(None, score)`` if the best score is below ``FACE_MATCH_THRESHOLD`` (or the
    gallery is empty).
    """
    if gallery is None:
        gallery = load_gallery()
    if gallery.embeddings.shape[0] == 0:
        return None, 0.0
    sims = gallery.embeddings @ embedding  # both L2-normalized -> cosine

    # Best cosine per enrolled name, then rank names.
    best_by_name: Dict[str, float] = {}
    for name, s in zip(gallery.names, sims):
        s = float(s)
        if s > best_by_name.get(name, -1.0):
            best_by_name[name] = s
    ranked = sorted(best_by_name.items(), key=lambda kv: kv[1], reverse=True)

    name, score = ranked[0]
    if score < config.FACE_MATCH_THRESHOLD:
        return None, score
    # Ambiguous if the runner-up (a different person) is within the margin.
    if len(ranked) > 1 and (score - ranked[1][1]) < config.FACE_MATCH_MARGIN:
        return None, score
    return name, score
