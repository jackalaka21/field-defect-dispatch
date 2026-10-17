"""Learned image-level aggregator over the detector's per-class box statistics.

The raw image-level score is the max box confidence of one class. A small model over all classes' statistics
(max conf, sum of top-3 confs, box count, top-box area) can use co-occurrence (exposed bars usually come with
spalling) and box counts. It is trained on out-of-fold detector features and evaluated by leave-one-fold-out over
the ORIGINAL detector CV folds (rows are stored fold by fold), so the aggregator never sees the photos it scores.

    python scripts/fit_stacker.py --scores reports/exp_v2_tta.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dispatch.det_eval import Record, choose_threshold  # noqa: E402
from dispatch.stacking import ImageScorer, design  # noqa: E402
from dispatch.taxonomy import DETECTOR_CLASSES  # noqa: E402
from train_detector_cv import RECALL_TARGET  # noqa: E402


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                                          random_state=0)


def evaluate(score: np.ndarray, y: np.ndarray, clean: np.ndarray, target: float) -> dict[str, float]:
    t = choose_threshold([Record(0, float(s), bool(v)) for s, v in zip(score, y)], int(y.sum()), target)
    return {"AP": float(average_precision_score(y, score)), "AUROC": float(roc_auc_score(y, score)),
            "precision_at_target": t["precision"], "recall": t["recall"], "threshold": t["threshold"],
            "FPR_clean": float((score[clean] >= t["threshold"]).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/stacker_eval.json"))
    a = ap.parse_args()

    z = np.load(a.scores)
    S, P, F, sizes = z["scores"], z["present"], z["feats"], z["fold_sizes"]
    X = design(F)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    clean = ~P.any(axis=1)
    out: dict[str, dict] = {}
    for kind in ("logreg", "hgb"):
        oof = np.zeros_like(S)
        for k in range(len(sizes)):
            te = np.arange(bounds[k], bounds[k + 1])
            tr = np.setdiff1d(np.arange(len(S)), te)
            for j, cls in enumerate(DETECTOR_CLASSES):
                if P[:, j].sum() == 0:
                    continue
                m = make_model(kind).fit(X[tr], P[tr, j])
                oof[te, j] = m.predict_proba(X[te])[:, 1]
        out[kind] = {}
        for j, cls in enumerate(DETECTOR_CLASSES):
            if P[:, j].sum() == 0:
                continue
            out[kind][cls.value] = evaluate(oof[:, j], P[:, j], clean, RECALL_TARGET[cls.value])
        np.save(a.out.with_name(f"stacker_oof_{kind}.npy"), oof)
    out["raw_max_conf"] = {c.value: evaluate(S[:, j], P[:, j], clean, RECALL_TARGET[c.value])
                           for j, c in enumerate(DETECTOR_CLASSES) if P[:, j].sum() > 0}
    # deployable scorer: fit on ALL out-of-fold features; thresholds come from the leave-one-fold-out probabilities
    oof = np.load(a.out.with_name("stacker_oof_logreg.npy"))
    thresholds = {}
    for j, cls in enumerate(DETECTOR_CLASSES):
        thresholds[cls.value] = out["logreg"][cls.value]["threshold"] if cls.value in out["logreg"] else 0.5
    Path("weights").mkdir(exist_ok=True)
    ImageScorer.fit(F, P).save("weights/stacker.joblib")
    (a.out.parent / "stacker_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    np.savez_compressed(a.out.with_name("detector_final_oof.npz"), scores=oof, present=P, areas=z["areas"],
                        fold_sizes=sizes)  # what the fusion bank replays
    a.out.write_text(json.dumps(out, indent=2))
    print(f"{'class':18s} " + " | ".join(f"{k:^29s}" for k in ("raw max conf", "logreg stacker", "hgb stacker")))
    print(f"{'':18s} " + " | ".join(f"{'AP':>6s} {'AUROC':>6s} {'P@tgt':>6s} {'FPRcln':>6s}" for _ in range(3)))
    for cls in DETECTOR_CLASSES:
        if cls.value not in out["raw_max_conf"]:
            continue
        row = [out[k][cls.value] for k in ("raw_max_conf", "logreg", "hgb")]
        print(f"{cls.value:18s} " + " | ".join(f"{r['AP']:6.3f} {r['AUROC']:6.3f} {r['precision_at_target']:6.3f} {r['FPR_clean']:6.3f}"
                                               for r in row))
    for k in ("raw_max_conf", "logreg", "hgb"):
        v = out[k].values()
        print(f"mean {k:14s} AP={np.mean([x['AP'] for x in v]):.3f} AUROC={np.mean([x['AUROC'] for x in v]):.3f} "
              f"FPR_clean={np.mean([x['FPR_clean'] for x in v]):.3f}")


if __name__ == "__main__":
    main()
