"""Monocular depth estimation (MiDaS) for relative distance cues.

A single RGB frame carries no true scale, but MiDaS predicts *relative* depth —
which pixels are nearer vs. farther. We use it two ways:

* tag each track with a relative distance (0 = far, 1 = near), so objects can be
  ordered front-to-back;
* render a colorized depth preview of one frame for the HUD.

The model (``MiDaS_small``) is pulled from ``torch.hub`` on first use and reused
across the video. Depth is *not* run every frame — the caller samples a handful
of frames — because it's a second network on top of detection.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

_MODEL = None  # (device, midas, transform)


def _get():
    global _MODEL
    if _MODEL is None:
        import torch
        # MiDaS_small loads its backbone via a nested torch.hub call that would
        # otherwise hit an interactive trust prompt; neutralize it for headless use.
        torch.hub._check_repo_is_trusted = lambda *a, **k: None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        midas.eval().to(device)
        _MODEL = (device, midas, transform)
    return _MODEL


def infer(image_bgr: np.ndarray) -> np.ndarray:
    """Return a normalized relative-depth map (HxW, 0=far .. 1=near)."""
    import torch

    device, midas, transform = _get()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return (d - lo) / (hi - lo)  # MiDaS outputs inverse depth -> higher already = nearer


def box_median(depth_map: np.ndarray, box: List[float]) -> Optional[float]:
    """Median relative depth inside ``box`` (a single distance value for the object)."""
    H, W = depth_map.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return round(float(np.median(region)), 3)


def colorize(image_bgr: np.ndarray) -> np.ndarray:
    """A MAGMA-colormapped depth preview image (BGR)."""
    u8 = (infer(image_bgr) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
