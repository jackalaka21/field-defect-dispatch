"""Image-level out-of-fold evaluation of the k-fold detectors, and recall-targeted per-class thresholds.

The fusion stage consumes "which defect classes are in this photo, how confident", not pixel-accurate boxes,
so the operating thresholds are tuned on image-level presence: for each (image, class) the detector's max
confidence versus whether the class is labelled in that image. This needs no IoU matching and is the metric
that matches how the detector is used. It re-creates the exact CV split used in training (same seed/groups).

    python scripts/eval_detector_oof.py --cv-dir runs/detector_cv_block --groups-file groups_block.csv --imgsz 640
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch.det_eval import Record, choose_threshold, read_yolo_labels  # noqa: E402
from dispatch.taxonomy import DETECTOR_CLASSES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector_cv import RECALL_TARGET, index_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/defects"))
    ap.add_argument("--cv-dir", type=Path, required=True)
    ap.add_argument("--groups-file", default="groups.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--only-folds", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from ultralytics import YOLO

    imgs, strata, groups = index_dataset(a.data, a.groups_file)
    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    K = len(DETECTOR_CLASSES)
    score, present, area, fold_sizes = [], [], [], []
    for k, (_, va) in enumerate(skf.split(imgs, strata, groups)):
        if a.only_folds is not None and k not in a.only_folds:
            continue
        best = a.cv_dir / f"fold{k}" / "run" / "weights" / "best.pt"
        model = YOLO(str(best if best.exists() else best.with_name("last.pt")))
        for i in va:
            res = model.predict(str(imgs[i]), imgsz=a.imgsz, conf=0.001, iou=0.6, device=a.device, verbose=False)[0]
            s, ar = np.zeros(K), np.zeros(K)
            for c, conf, (bx1, by1, bx2, by2) in zip(res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy(),
                                                    res.boxes.xyxyn.cpu().numpy()):
                if conf > s[c]:
                    s[c], ar[c] = float(conf), float((bx2 - bx1) * (by2 - by1))  # area of the top-scoring box
            p = np.zeros(K, bool)
            for c, _ in read_yolo_labels(a.data / "labels" / f"{imgs[i].stem}.txt"):
                p[c] = True
            score.append(s)
            area.append(ar)
            present.append(p)
        fold_sizes.append(len(va))
        print(f"fold {k}: scored {len(va)} images", flush=True)
    S, P, A = np.array(score), np.array(present), np.array(area)

    out = {"n_images": int(len(S)), "per_class": {}}
    thresholds = {}
    for j, cls in enumerate(DETECTOR_CLASSES):
        n_pos = int(P[:, j].sum())
        if n_pos == 0 or n_pos == len(P):
            out["per_class"][cls.value] = {"n_positive_images": n_pos, "note": "no labels in this dataset"}
            thresholds[cls.value] = 0.5
            continue
        recs = [Record(j, float(S[i, j]), bool(P[i, j])) for i in range(len(S))]
        t = choose_threshold(recs, n_pos, RECALL_TARGET[cls.value])
        thresholds[cls.value] = t["threshold"]
        out["per_class"][cls.value] = {
            "n_positive_images": n_pos, "image_AP": float(average_precision_score(P[:, j], S[:, j])),
            "image_AUROC": float(roc_auc_score(P[:, j], S[:, j])), "prevalence": n_pos / len(P),
            "recall_target": RECALL_TARGET[cls.value], **t}
    # false positives on photos with no labelled defect at all (CODEBRIM photos without annotations; a few may hold
    # faint unlabelled stains, so this is a pessimistic estimate)
    clean = ~P.any(axis=1)
    out["clean_images"] = int(clean.sum())
    if clean.any():
        trained = [j for j, c in enumerate(DETECTOR_CLASSES) if out["per_class"][c.value].get("n_positive_images", 0) > 0]
        flagged = np.zeros(len(S), bool)
        for j in trained:
            hit = S[:, j] >= thresholds[DETECTOR_CLASSES[j].value]
            out["per_class"][DETECTOR_CLASSES[j].value]["false_positive_rate_on_clean"] = float(hit[clean].mean())
            flagged |= hit
        out["clean_flagged_for_any_class"] = float(flagged[clean].mean())
        out["defect_photos_flagged_for_any_class"] = float(flagged[~clean].mean())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix(".npz"), scores=S, present=P, areas=A, fold_sizes=np.array(fold_sizes))
    a.out.write_text(json.dumps(out, indent=2))
    a.out.with_name(a.out.stem + "_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    for name, r in out["per_class"].items():
        if "image_AP" in r:
            print(f"{name:18s} AP={r['image_AP']:.3f} AUROC={r['image_AUROC']:.3f} prev={r['prevalence']:.2f} "
                  f"thr={r['threshold']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"target={r['recall_target']:.2f} reached={r['reached_target']} "
                  f"FPR(clean)={r.get('false_positive_rate_on_clean', float('nan')):.3f}")
        else:
            print(f"{name:18s} {r['note']}")
    if "clean_flagged_for_any_class" in out:
        print(f"clean photos: {out['clean_images']}; flagged for any class: {out['clean_flagged_for_any_class']:.3f}; "
              f"defect photos flagged for any class: {out['defect_photos_flagged_for_any_class']:.3f}")


if __name__ == "__main__":
    main()
