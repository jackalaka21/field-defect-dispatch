"""Detector evaluation helpers: IoU matching and recall-targeted operating thresholds.

Hazard classes are tuned for *recall* (a missed exposed-rebar photo is a missed ticket), cosmetic
classes for a balance. Thresholds are chosen on pooled out-of-fold predictions so they are not fit
to the images the detector trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vision import iou

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Record:
    cls: int
    conf: float
    tp: bool


def read_yolo_labels(path: str | Path) -> list[tuple[int, Box]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[tuple[int, Box]] = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        out.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def match_image(preds: list[tuple[int, float, Box]], gts: list[tuple[int, Box]],
                iou_thr: float = 0.5) -> tuple[list[Record], dict[int, int]]:
    """Greedy per-class matching by descending confidence. Returns records and GT counts per class."""
    records: list[Record] = []
    n_gt: dict[int, int] = {}
    for c, _ in gts:
        n_gt[c] = n_gt.get(c, 0) + 1
    for c in {p[0] for p in preds}:
        gt_boxes = [b for g, b in gts if g == c]
        used = [False] * len(gt_boxes)
        for _, conf, box in sorted((p for p in preds if p[0] == c), key=lambda p: -p[1]):
            best, best_i = 0.0, -1
            for i, gb in enumerate(gt_boxes):
                if not used[i]:
                    v = iou(box, gb)
                    if v > best:
                        best, best_i = v, i
            hit = best >= iou_thr
            if hit:
                used[best_i] = True
            records.append(Record(c, conf, hit))
    return records, n_gt


def choose_threshold(records: list[Record], n_gt: int, recall_target: float) -> dict[str, float]:
    """Highest-precision threshold that still reaches `recall_target`; else the best-F2 threshold."""
    if not records or n_gt == 0:
        return {"threshold": 0.5, "precision": float("nan"), "recall": 0.0, "reached_target": False}
    recs = sorted(records, key=lambda r: -r.conf)
    tp = np.cumsum([r.tp for r in recs])
    fp = np.cumsum([not r.tp for r in recs])
    precision, recall = tp / np.maximum(tp + fp, 1), tp / n_gt
    ok = np.where(recall >= recall_target)[0]
    if len(ok):
        i = int(ok[np.argmax(precision[ok])])
        reached = True
    else:
        f2 = 5 * precision * recall / np.maximum(4 * precision + recall, 1e-9)
        i, reached = int(np.argmax(f2)), False
    return {"threshold": float(recs[i].conf), "precision": float(precision[i]), "recall": float(recall[i]),
            "reached_target": reached}
