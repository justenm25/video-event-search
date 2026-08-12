"""Soft-biometric face attributes (age + gender) via InsightFace.

InsightFace's ``buffalo_l`` pack bundles a face detector plus a small age/gender
network. Given a person crop, we detect the largest face inside it and return an
estimated age and gender. (The pack does not include an emotion/expression model,
so we surface age + gender only — the reliable signals.)

The model is loaded lazily and reused; it runs on CPU via onnxruntime, which is
plenty for the handful of crops we score (once per person track).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_APP = None


def _get():
    global _APP
    if _APP is None:
        import warnings
        warnings.filterwarnings("ignore")
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=["detection", "genderage"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _APP = app
    return _APP


def analyze(crop_bgr: np.ndarray) -> Optional[Dict]:
    """Return ``{'age': int, 'gender': 'M'|'F'}`` for the largest face in ``crop_bgr``."""
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 32:
        return None
    app = _get()
    try:
        faces = app.get(crop_bgr)
    except Exception:
        return None
    if not faces:
        return None
    # largest detected face
    f = max(faces, key=lambda ff: (ff.bbox[2] - ff.bbox[0]) * (ff.bbox[3] - ff.bbox[1]))
    return {"age": int(f.age), "gender": str(f.sex)}


def analyze_best(crops: List[np.ndarray]) -> Optional[Dict]:
    """Score up to a few crops and return the first successful age/gender read."""
    for c in crops:
        res = analyze(c)
        if res is not None:
            return res
    return None
