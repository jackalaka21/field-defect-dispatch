"""Fit and validate per-class score calibration from the pooled out-of-fold image-level scores.

    python scripts/fit_calibration.py --scores reports/detector_image_level_v2.npz --eval reports/detector_image_level_v2.json

Writes reports/detector_calibration.json (curves) and reports/detector_thresholds.json (raw-score thresholds).
Calibration quality is measured by cross-fitting (fit on 4/5 of the scores, score the held-out 1/5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch.calibration import Calibrator  # noqa: E402
from dispatch.taxonomy import DETECTOR_CLASSES  # noqa: E402


def binary_ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True, help="JSON written by eval_detector_oof.py (thresholds)")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    a = ap.parse_args()

    z = np.load(a.scores)
    S, P = z["scores"], z["present"]
    cal = Calibrator.fit(S, P, DETECTOR_CLASSES)
    cross = np.zeros_like(S)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(S):
        c = Calibrator.fit(S[tr], P[tr], DETECTOR_CLASSES)
        for j, cls in enumerate(DETECTOR_CLASSES):
            cross[te, j] = [c(cls, v) for v in S[te, j]]
    report = {}
    for j, cls in enumerate(DETECTOR_CLASSES):
        if cls.value not in cal.curves:
            continue
        y = P[:, j].astype(float)
        raw = np.clip(S[:, j], 0, 1)
        report[cls.value] = {"ece_raw": binary_ece(raw, y), "ece_calibrated_crossfit": binary_ece(cross[:, j], y),
                             "brier_raw": float(np.mean((raw - y) ** 2)),
                             "brier_calibrated_crossfit": float(np.mean((cross[:, j] - y) ** 2))}
        print(f"{cls.value:18s} ECE {report[cls.value]['ece_raw']:.3f} -> {report[cls.value]['ece_calibrated_crossfit']:.3f}   "
              f"Brier {report[cls.value]['brier_raw']:.3f} -> {report[cls.value]['brier_calibrated_crossfit']:.3f}")

    a.out_dir.mkdir(exist_ok=True)
    cal.save(a.out_dir / "detector_calibration.json")
    ev = json.loads(a.eval.read_text())
    thresholds = {c.value: float(ev["per_class"][c.value].get("threshold", 0.5)) for c in DETECTOR_CLASSES}
    (a.out_dir / "detector_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (a.out_dir / "detector_calibration_quality.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
