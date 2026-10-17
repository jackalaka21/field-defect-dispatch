"""YOLOv8 defect detection with sliced inference and per-class operating thresholds.

Small defects (hairline cracks, early corrosion) shrink to a few pixels when a 12 MP photo is
resized to 640, so large images are also run as overlapping tiles and the detections are merged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schemas import Detection
from .taxonomy import DETECTOR_CLASSES, DefectClass

Box = tuple[float, float, float, float]


def tile_windows(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping square-ish windows (x0, y0, x1, y1) covering the whole image."""
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile, step))
        s.append(size - tile)  # the last window ends exactly at the border
        return s

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


def iou(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_detections(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Class-wise greedy NMS."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.conf, reverse=True):
        if all(k.cls is not d.cls or iou(k.box, d.box) < iou_thr for k in kept):
            kept.append(d)
    return kept


def load_thresholds(path: str | Path, default: float = 0.25) -> dict[DefectClass, float]:
    raw = json.loads(Path(path).read_text())
    return {c: float(raw.get(c.value, default)) for c in DETECTOR_CLASSES}


class DefectDetector:
    def __init__(self, weights: str, thresholds: dict[DefectClass, float] | None = None, *, tile: int | None = 640,
                 overlap: float = 0.2, imgsz: int = 640, device: str | int = 0, floor_conf: float = 0.02):
        from ultralytics import YOLO  # imported lazily so the rest of the package needs no vision stack

        self.model = YOLO(weights)
        self.thresholds = thresholds or {}
        self.tile, self.overlap, self.imgsz, self.device, self.floor_conf = tile, overlap, imgsz, device, floor_conf
        self.names = self.model.names

    def _run(self, img: np.ndarray, x0: int, y0: int, W: int, H: int) -> list[Detection]:
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.floor_conf, device=self.device, verbose=False)[0]
        out: list[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy()):
            try:
                cls = DefectClass(self.names[int(k)])
            except ValueError:
                continue
            nx1, ny1 = max(0.0, (x1 + x0) / W), max(0.0, (y1 + y0) / H)
            nx2, ny2 = min(1.0, (x2 + x0) / W), min(1.0, (y2 + y0) / H)
            if nx2 > nx1 and ny2 > ny1:
                out.append(Detection(cls=cls, conf=float(np.clip(c, 0, 1)), box=(nx1, ny1, nx2, ny2)))
        return out

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:
        if not isinstance(image, np.ndarray):
            from PIL import Image

            image = np.asarray(Image.open(image).convert("RGB"))
        H, W = image.shape[:2]
        dets = self._run(image, 0, 0, W, H)
        if self.tile and max(H, W) > self.tile * 1.25:
            for x0, y0, x1, y1 in tile_windows(W, H, self.tile, self.overlap):
                dets += self._run(image[y0:y1, x0:x1], x0, y0, W, H)
        dets = merge_detections(dets)
        return [d for d in dets if d.conf >= self.thresholds.get(d.cls, 0.25)]
