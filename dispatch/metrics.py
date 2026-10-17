"""Metrics that reflect how dispatch errors actually hurt: missed criticals matter most."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from .fusion import N_PRIORITIES, CostMatrix


def priority_metrics(y_true: np.ndarray, y_pred: np.ndarray, cost: CostMatrix = CostMatrix()) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    m = cost.matrix()
    p1_true = y_true == 0
    p1_pred = y_pred == 0
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "within_one": float((np.abs(y_true - y_pred) <= 1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(N_PRIORITIES)), zero_division=0)),
        "under_rate": float((y_pred > y_true).mean()),
        "critical_miss_rate": float((p1_true & ~p1_pred).sum() / max(1, p1_true.sum())),
        "p1_precision": float((p1_true & p1_pred).sum() / max(1, p1_pred.sum())),
        "mean_cost": float(m[y_true, y_pred].mean()),
    }


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    hit = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(hit[sel].mean() - conf[sel].mean())
    return float(ece)


# --------------------------------------------------------------------------- text metrics
def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()


def word_error_rate(ref: str, hyp: str) -> float:
    r = _norm(ref)
    return _edit_distance(r, _norm(hyp)) / max(1, len(r))


def keyword_error_rate(ref: str, hyp: str, keywords: set[str]) -> tuple[int, int]:
    """(missed keywords, keywords in reference). A keyword counts as found if it appears in hyp."""
    r = [w for w in _norm(ref) if w in keywords]
    h = set(_norm(hyp))
    return sum(1 for w in r if w not in h), len(r)
