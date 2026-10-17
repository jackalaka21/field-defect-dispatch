"""Replay of REAL out-of-fold detector behaviour inside synthetic cases.

For a case whose true defect is class c, the bank draws a real CODEBRIM photo that contains c (or a clean photo for
`no_defect`) and returns what the cross-validated detector actually reported for it: its false positives, its
confusions and its misses, at the tuned operating thresholds, with calibrated confidences. Voice, severity and the
priority label stay synthetic, so this is a hybrid fixture: the vision noise is empirical, the rest is not.
Classes with no labels in CODEBRIM (hairline_crack, water_leak) are not covered and fall back to the simulator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..calibration import Calibrator
from ..schemas import Detection
from ..taxonomy import DETECTOR_CLASSES, DefectClass


class DetectorBank:
    def __init__(self, npz: str | Path, thresholds_json: str | Path, calibration_json: str | Path):
        z = np.load(npz)
        self.scores, self.present, self.areas = z["scores"], z["present"], z["areas"]
        raw = json.loads(Path(thresholds_json).read_text())
        self.thresholds = {c: float(raw[c.value]) for c in DETECTOR_CLASSES}
        self.calibrator = Calibrator.load(calibration_json)
        self.fold_sizes = z["fold_sizes"]
        self._index(np.arange(len(self.scores)))

    def _index(self, rows: np.ndarray) -> None:
        keep = np.zeros(len(self.scores), bool)
        keep[rows] = True
        self.clean = np.where(~self.present.any(axis=1) & keep)[0]
        self.by_class = {c: np.where(self.present[:, j] & keep)[0] for j, c in enumerate(DETECTOR_CLASSES)}

    def view(self, part: str) -> "DetectorBank":
        """'dev' = first 3 CV folds' photos, 'holdout' = last 2. Rows are stored fold by fold, so the two parts share
        no photo (and, because folds are block-grouped, no neighbouring photos either)."""
        import copy

        cut = int(np.sum(self.fold_sizes[:3]))
        v = copy.copy(self)
        v._index(np.arange(cut) if part == "dev" else np.arange(cut, len(self.scores)))
        return v

    def covers(self, cls: DefectClass) -> bool:
        if cls is DefectClass.NO_DEFECT:
            return len(self.clean) > 0
        return len(self.by_class.get(cls, [])) > 0

    def sample(self, cls: DefectClass, rng: np.random.Generator) -> list[Detection]:
        pool = self.clean if cls is DefectClass.NO_DEFECT else self.by_class[cls]
        i = int(rng.choice(pool))
        dets: list[Detection] = []
        for j, c in enumerate(DETECTOR_CLASSES):
            raw = float(self.scores[i, j])
            if raw < self.thresholds[c] or raw <= 0.0:
                continue
            side = float(np.clip(np.sqrt(max(self.areas[i, j], 1e-4)), 0.02, 0.98))
            x1 = float(rng.uniform(0.0, 1.0 - side))
            y1 = float(rng.uniform(0.0, 1.0 - side))
            conf = float(np.clip(self.calibrator(c, raw), 0.0, 1.0))
            dets.append(Detection(cls=c, conf=conf, box=(x1, y1, x1 + side, y1 + side)))
        return dets
