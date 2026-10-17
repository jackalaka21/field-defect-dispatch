"""Image-level scoring: per-class box statistics -> calibrated presence probabilities.

The detector emits many boxes; the pipeline needs "is class c in this photo, how sure". The statistics of ALL
classes' boxes are summarised per class (max conf, sum of the top-3 confs, number of boxes >= 0.1, area of the top
box) and a small model per class maps them to a probability. Using every class's statistics lets it exploit
co-occurrence (exposed bars usually come with spalling) and box counts that a plain max-confidence ignores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STAT_NAMES = ("max_conf", "top3_sum", "n_ge_0.1", "top_area")


def class_stats(cls_ids: np.ndarray, confs: np.ndarray, areas: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class box statistics for one image -> (n_classes, 4). Inputs are all boxes of the image."""
    f = np.zeros((n_classes, len(STAT_NAMES)))
    for c in range(n_classes):
        sel = cls_ids == c
        if not sel.any():
            continue
        cc, aa = confs[sel], areas[sel]
        order = np.argsort(-cc)
        f[c] = [cc[order[0]], cc[order[:3]].sum(), float((cc >= 0.1).sum()), aa[order[0]]]
    return f


def design(F: np.ndarray) -> np.ndarray:
    """(n, K, 4) statistics -> (n, 4K + 1) features, compressing the heavy-tailed ones."""
    maxc, top3, cnt, area = F[..., 0], F[..., 1], F[..., 2], F[..., 3]
    X = np.concatenate([maxc, np.log1p(10 * top3), np.log1p(cnt), np.sqrt(area)], axis=1)
    return np.concatenate([X, maxc.max(axis=1, keepdims=True)], axis=1)


class ImageScorer:
    def __init__(self, models: dict[int, object] | None = None, n_classes: int = 0):
        self.models = models or {}
        self.n_classes = n_classes

    @classmethod
    def fit(cls, F: np.ndarray, present: np.ndarray, C: float = 1.0) -> "ImageScorer":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, models = design(F), {}
        for j in range(present.shape[1]):
            if present[:, j].sum() == 0 or present[:, j].all():
                continue  # no labels for this class: it is never reported
            models[j] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, present[:, j])
        return cls(models, present.shape[1])

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        X = design(F)
        out = np.zeros((len(F), self.n_classes))
        for j, m in self.models.items():
            out[:, j] = m.predict_proba(X)[:, 1]
        return out

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"models": self.models, "n_classes": self.n_classes}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ImageScorer":
        import joblib

        d = joblib.load(path)
        return cls(d["models"], d["n_classes"])
