"""Turn a person's 17 COCO keypoints into a coarse action label.

The YOLO pose model gives us, per person, 17 body keypoints (nose, shoulders,
elbows, wrists, hips, knees, ankles). That's enough for simple, explainable
rule-based action recognition — no temporal model needed for a first pass:

    raising hand  — a wrist is above the shoulders
    fallen        — the body is wider than tall / torso is horizontal
    sitting       — knees are drawn up close to the hips (compressed lower body)
    standing      — default

Each rule is a geometric relationship between keypoints, normalized by the
person's box height so it's scale-invariant. Keypoints below ``KP_CONF`` are
treated as unknown.
"""
from __future__ import annotations

from typing import List, Optional

# COCO-17 keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

KP_CONF = 0.3  # minimum keypoint confidence to trust a joint


def _pt(kpts: List[List[float]], i: int):
    """Return (x, y) if keypoint ``i`` is confident enough, else None."""
    if i >= len(kpts):
        return None
    x, y, c = kpts[i]
    return (x, y) if c >= KP_CONF else None


def _avg_y(*pts):
    ys = [p[1] for p in pts if p is not None]
    return sum(ys) / len(ys) if ys else None


def classify_action(kpts: List[List[float]], box: List[float]) -> Optional[str]:
    """Best-effort action for one person given keypoints ``[[x,y,conf]*17]``."""
    if not kpts:
        return None
    x1, y1, x2, y2 = box
    box_h = max(1.0, y2 - y1)
    box_w = max(1.0, x2 - x1)

    sh_y = _avg_y(_pt(kpts, L_SHOULDER), _pt(kpts, R_SHOULDER))
    hip_y = _avg_y(_pt(kpts, L_HIP), _pt(kpts, R_HIP))
    knee_y = _avg_y(_pt(kpts, L_KNEE), _pt(kpts, R_KNEE))

    # Fallen: lying down reads as a wide, short box, or a near-horizontal torso.
    if box_w / box_h > 1.3:
        return "fallen"
    if sh_y is not None and hip_y is not None and abs(hip_y - sh_y) < 0.15 * box_h:
        return "fallen"

    # Raising hand: a wrist sits above the shoulder line.
    if sh_y is not None:
        for w in (_pt(kpts, L_WRIST), _pt(kpts, R_WRIST)):
            if w is not None and w[1] < sh_y:
                return "raising hand"

    # Sitting: knees pulled up close to the hips (short hip->knee span).
    if hip_y is not None and knee_y is not None and (knee_y - hip_y) < 0.2 * box_h:
        return "sitting"

    return "standing"
