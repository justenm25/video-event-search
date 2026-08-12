"""Face detection + embedding for custom identity recognition.

Wraps ``facenet-pytorch`` (MTCNN face detector + InceptionResnetV1 trained on
VGGFace2). Given an image, we find the most prominent face and return a 512-d
**L2-normalized** embedding — a numeric "fingerprint" of that face. Two faces of
the same person have embeddings with high cosine similarity; different people
score low. Matching happens in ``identities.py``.

The heavy torch model is imported and built lazily (mirroring how ``detector.py``
defers importing ultralytics) so the CLI/tests load without torch present and we
pay the model-load cost only once, on first use.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import config


# Cached singletons: (device, mtcnn, resnet). Built on first embed_face call.
_MODELS = None


def _get_models():
    global _MODELS
    if _MODELS is None:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # keep_all=False + select_largest=True -> return the single biggest face,
        # already aligned/cropped to 3x160x160 and normalized for the embedder.
        mtcnn = MTCNN(
            image_size=160,
            keep_all=False,
            select_largest=True,
            post_process=True,
            device=device,
        )
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        _MODELS = (device, mtcnn, resnet)
    return _MODELS


def embed_face(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return a 512-d L2-normalized embedding for the largest face in ``img_bgr``.

    ``img_bgr`` is an OpenCV-style HxWx3 uint8 array (BGR channel order), which is
    what both ``cv2.imread`` and YOLO's ``r.orig_img`` produce. Returns ``None`` if
    no face is detected (e.g. a back-turned or too-small person crop).
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    import torch
    from PIL import Image

    device, mtcnn, resnet = _get_models()

    # MTCNN expects RGB. cv2/YOLO give BGR -> flip channels.
    rgb = img_bgr[:, :, ::-1]
    try:
        face, prob = mtcnn(Image.fromarray(np.ascontiguousarray(rgb)), return_prob=True)
    except Exception:
        return None
    # Reject weak detections (blurry / partial / not-really-a-face crops).
    if face is None or prob is None or float(prob) < config.FACE_MIN_PROB:
        return None

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    if norm == 0:
        return None
    return (emb / norm).astype(np.float32)
