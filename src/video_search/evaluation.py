"""Quantitative evaluation of predictions against ground truth.

For a pattern-recognition project it isn't enough to *demo* detections — you have
to *measure* them. Given a hand-labeled ground-truth (GT) file for a video, this
module scores the stored predictions:

* **Detection** — precision / recall / F1 and **AP@0.5** (mean over classes),
  using greedy IoU matching at 0.5.
* **Tracking** — **MOTA** and **IDF1** (via ``motmetrics``) when the GT carries
  per-object track ids.
* **Identity** — fraction of GT person boxes whose matched prediction carries the
  correct enrolled name (when GT boxes are labeled with names).

GT format (JSON), aligned to predictions by nearest timestamp::

    {"frames": [{"t": 0.0, "dets": [{"cls": "person", "box": [x1,y1,x2,y2],
                                     "id": 1, "name": "Nimish"}, ...]}, ...]}

``id`` and ``name`` are optional (needed only for tracking / identity metrics).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# motmetrics 1.4 still calls np.asfarray, removed in NumPy 2.0 — restore the alias.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _align(pred_frames: List[Dict], gt_frames: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Pair each GT frame with the prediction frame nearest in time."""
    if not pred_frames or not gt_frames:
        return []
    p_times = [f.get("t", i) for i, f in enumerate(pred_frames)]
    pairs = []
    for g in gt_frames:
        gt_t = g.get("t", 0.0)
        j = min(range(len(p_times)), key=lambda k: abs(p_times[k] - gt_t))
        pairs.append((pred_frames[j], g))
    return pairs


def _ap_from_pr(scores: List[float], matches: List[int], n_gt: int) -> float:
    """Average precision (area under PR curve) for one class at IoU 0.5."""
    if n_gt == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for i in order:
        if matches[i]:
            tp += 1
        else:
            fp += 1
        recall = tp / n_gt
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)  # right-hand Riemann step
        prev_recall = recall
    return ap


def evaluate(pred_payload: Dict, gt: Dict, iou_thr: float = 0.5) -> Dict:
    """Score ``pred_payload`` (a detections.json dict) against ground truth ``gt``."""
    pairs = _align(pred_payload.get("frames", []), gt.get("frames", []))

    tp = fp = fn = 0
    per_class: Dict[str, Dict] = {}   # cls -> {scores, matches, n_gt}
    id_total = id_correct = 0
    gt_has_ids = False

    # motmetrics accumulator (only used if GT has ids)
    try:
        import motmetrics as mm
        acc = mm.MOTAccumulator(auto_id=True)
        have_mm = True
    except Exception:
        acc = None
        have_mm = False

    for pred_fr, gt_fr in pairs:
        preds = list(pred_fr.get("dets", []))
        gts = list(gt_fr.get("dets", []))
        for g in gts:
            per_class.setdefault(g["cls"], {"scores": [], "matches": [], "n_gt": 0})["n_gt"] += 1

        # greedy match preds -> gts by descending confidence, same class, IoU>=thr
        preds_sorted = sorted(preds, key=lambda d: d.get("conf", 1.0), reverse=True)
        used = set()
        for p in preds_sorted:
            pc = per_class.setdefault(p["cls"], {"scores": [], "matches": [], "n_gt": 0})
            best, bj = iou_thr, -1
            for j, g in enumerate(gts):
                if j in used or g["cls"] != p["cls"]:
                    continue
                v = _iou(p["box"], g["box"])
                if v >= best:
                    best, bj = v, j
            matched = bj >= 0
            pc["scores"].append(p.get("conf", 1.0))
            pc["matches"].append(1 if matched else 0)
            if matched:
                used.add(bj)
                tp += 1
                g = gts[bj]
                if "name" in g:
                    id_total += 1
                    if p.get("cls") == g["name"]:
                        id_correct += 1
            else:
                fp += 1
        fn += len(gts) - len(used)

        # tracking accumulator (class-agnostic, needs ids on both sides)
        if have_mm and any("id" in g for g in gts):
            gt_has_ids = True
            gt_ids = [g.get("id") for g in gts if "id" in g]
            gt_boxes = [_xywh(g["box"]) for g in gts if "id" in g]
            hyp_ids = [p.get("id") for p in preds if p.get("id") is not None]
            hyp_boxes = [_xywh(p["box"]) for p in preds if p.get("id") is not None]
            dists = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=1 - iou_thr) \
                if gt_boxes and hyp_boxes else _empty(len(gt_boxes), len(hyp_boxes))
            acc.update(gt_ids, hyp_ids, dists)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    aps = {c: _ap_from_pr(v["scores"], v["matches"], v["n_gt"]) for c, v in per_class.items()}
    mAP = sum(aps.values()) / len(aps) if aps else 0.0

    result = {
        "frames_evaluated": len(pairs),
        "detection": {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "mAP@0.5": round(mAP, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "per_class_ap": {c: round(a, 4) for c, a in aps.items()},
        },
        "identity": ({"labeled_gt": id_total, "correct": id_correct,
                      "accuracy": round(id_correct / id_total, 4)} if id_total else None),
        "tracking": None,
    }

    if have_mm and gt_has_ids:
        try:
            import motmetrics as mm
            mh = mm.metrics.create()
            summ = mh.compute(acc, metrics=["mota", "idf1", "num_switches",
                                            "num_false_positives", "num_misses"], name="acc")
            row = summ.loc["acc"]
            result["tracking"] = {
                "MOTA": round(float(row["mota"]), 4),
                "IDF1": round(float(row["idf1"]), 4),
                "id_switches": int(row["num_switches"]),
                "false_positives": int(row["num_false_positives"]),
                "misses": int(row["num_misses"]),
            }
        except Exception as exc:
            result["tracking"] = {"error": str(exc)}

    return result


def _xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _empty(n, m):
    import numpy as np
    return np.full((n, m), np.nan)
