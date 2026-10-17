"""Per-class calibration of detector scores.

Raw YOLO confidences are not probabilities: a class can be present in most images where its score is only 0.1.
The fusion stage thresholds and weighs confidences as if they were probabilities, so scores are mapped through a
monotone (isotonic) curve fitted on pooled out-of-fold image-level scores.

Prior shift warning: the curve reflects the class prevalence of the data it was fitted on (20-40% of photos here
contain each class). In deployment with a different defect rate the calibrated value over- or under-states the
true probability; refit on a sample of production photos before trusting absolute levels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .taxonomy import DefectClass


class Calibrator:
    def __init__(self, curves: dict[str, tuple[list[float], list[float]]] | None = None):
        self.curves = curves or {}

    @classmethod
    def fit(cls, scores: np.ndarray, present: np.ndarray, classes: list[DefectClass]) -> "Calibrator":
        curves: dict[str, tuple[list[float], list[float]]] = {}
        for j, c in enumerate(classes):
            y = present[:, j]
            if y.sum() == 0 or y.all():
                continue  # no labels for this class: leave it uncalibrated
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores[:, j], y.astype(float))
            xs, ys = iso.X_thresholds_, iso.y_thresholds_
            curves[c.value] = ([float(v) for v in xs], [float(v) for v in ys])
        return cls(curves)

    def __call__(self, cls: DefectClass, conf: float) -> float:
        curve = self.curves.get(cls.value)
        if curve is None:
            return float(conf)
        return float(np.interp(conf, curve[0], curve[1]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.curves))

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        return cls({k: (v[0], v[1]) for k, v in json.loads(Path(path).read_text()).items()})
