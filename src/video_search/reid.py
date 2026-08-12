"""Appearance-based person Re-Identification (Re-ID).

Face recognition only works when a face is visible. Re-ID instead describes a
person by their overall *appearance* — build, clothing, colors — so the same
person can be linked even when turned away or after their track breaks and
restarts with a new id.

We use a ResNet-50 (ImageNet) as a generic appearance encoder: the 2048-d
global-average-pooled feature, L2-normalized, is a robust descriptor for
"same person, same clothes" matching within a clip. (A ReID-specialized backbone
like OSNet could be dropped in here later for higher accuracy.)
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

_MODEL = None  # (device, encoder)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get():
    global _MODEL
    if _MODEL is None:
        import torch
        from torchvision.models import resnet50, ResNet50_Weights

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        encoder = torch.nn.Sequential(*list(net.children())[:-1]).eval().to(device)
        _MODEL = (device, encoder)
    return _MODEL


def embed(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalized 2048-d appearance embedding for a person crop (BGR)."""
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 16:
        return None
    import torch

    device, encoder = _get()
    x = cv2.resize(crop_bgr, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        f = encoder(torch.from_numpy(x).float().to(device)).flatten().cpu().numpy()
    n = np.linalg.norm(f)
    return (f / n).astype(np.float32) if n > 0 else None


def mean_embedding(crops: List[np.ndarray]) -> Optional[np.ndarray]:
    """Average appearance embedding over several crops of one track (renormalized)."""
    embs = [e for e in (embed(c) for c in crops) if e is not None]
    if not embs:
        return None
    m = np.mean(embs, axis=0)
    n = np.linalg.norm(m)
    return (m / n).astype(np.float32) if n > 0 else None
