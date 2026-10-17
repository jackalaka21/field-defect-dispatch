"""Site-grouped k-fold fine-tuning of YOLOv8, out-of-fold threshold tuning, optional final fit.

Dataset layout (YOLO format, class ids follow dispatch.taxonomy.DETECTOR_CLASSES):
    <data>/images/*.jpg   <data>/labels/*.txt     optional: <data>/groups.csv  (image_stem,site)

Images from the same site (or the same inspection) must never straddle train and validation, so the
default group is the stem prefix before the first '_' ("siteA_0001.jpg" -> "siteA").

    python scripts/train_detector_cv.py --data data/defects --weights yolov8s.pt --folds 5 --epochs 60
    python scripts/train_detector_cv.py --data <toy> --weights yolov8n.yaml --epochs 1 --folds 2 --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch.det_eval import choose_threshold, match_image, read_yolo_labels  # noqa: E402
from dispatch.taxonomy import DETECTOR_CLASSES  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# recall targets: hazards must be caught; cosmetic classes can trade recall for precision
RECALL_TARGET = {"exposed_rebar": 0.95, "rebar_corrosion": 0.90, "spalling": 0.90, "structural_crack": 0.95,
                 "hairline_crack": 0.75, "water_leak": 0.85, "efflorescence": 0.75}


def index_dataset(data: Path, groups_file: str = "groups.csv") -> tuple[list[Path], np.ndarray, np.ndarray]:
    groups_csv = data / groups_file
    given = {}
    if groups_csv.exists():
        with groups_csv.open() as f:
            given = {r[0]: r[1] for r in csv.reader(f) if len(r) >= 2}
    imgs = sorted(p for p in (data / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    strata, groups = [], []
    for p in imgs:
        labels = read_yolo_labels(data / "labels" / f"{p.stem}.txt")
        strata.append(min((c for c, _ in labels), default=-1))  # rarest-friendly: lowest class id present
        groups.append(given.get(p.stem, p.stem.split("_")[0]))
    return imgs, np.array(strata), np.array(groups)


def write_split(fold_dir: Path, train: list[Path], val: list[Path]) -> Path:
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "train.txt").write_text("\n".join(str(p.resolve()) for p in train))
    (fold_dir / "val.txt").write_text("\n".join(str(p.resolve()) for p in val))
    cfg = {"path": str(fold_dir.resolve()), "train": "train.txt", "val": "val.txt",
           "names": {i: c.value for i, c in enumerate(DETECTOR_CLASSES)}}
    out = fold_dir / "data.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def train(weights: str, data_yaml: Path, run_dir: Path, a: argparse.Namespace):
    from ultralytics import YOLO

    model = YOLO(weights)
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, seed=a.seed,
                project=str(run_dir.parent.resolve()), name=run_dir.name, exist_ok=True, patience=a.patience,
                plots=False, amp=not a.smoke, workers=a.workers, verbose=False,
                # field-condition augmentation: lighting, glare, odd angles; no vertical flip (gravity matters)
                hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, degrees=10, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0)
    best = run_dir / "weights" / "best.pt"
    return YOLO(str(best if best.exists() else run_dir / "weights" / "last.pt"))


def predict_records(model, imgs: list[Path], labels_dir: Path, a: argparse.Namespace):
    from PIL import Image

    recs: dict[int, list] = {i: [] for i in range(len(DETECTOR_CLASSES))}
    n_gt = {i: 0 for i in range(len(DETECTOR_CLASSES))}
    for p in imgs:
        W, H = Image.open(p).size
        res = model.predict(str(p), imgsz=a.imgsz, conf=0.001, iou=0.6, device=a.device, verbose=False)[0]
        preds = [(int(k), float(c), (x1 / W, y1 / H, x2 / W, y2 / H))
                 for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                                   res.boxes.cls.cpu().numpy())]
        r, g = match_image(preds, read_yolo_labels(labels_dir / f"{p.stem}.txt"))
        for rec in r:
            recs[rec.cls].append(rec)
        for c, n in g.items():
            n_gt[c] += n
    return recs, n_gt


def fit_final(a: argparse.Namespace, imgs, strata, groups) -> None:
    """Fit on all data. ~10% of groups are held out purely for checkpoint selection / early stopping, so the
    saved weights come from one run whose reported CV numbers are the out-of-fold ones, not these."""
    tr, va = next(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=a.seed).split(imgs, strata, groups))
    dy = write_split(a.out / "final", [imgs[i] for i in tr], [imgs[i] for i in va])
    train(a.weights, dy, a.out / "final" / "run", a)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(a.out / "final" / "run" / "weights" / "best.pt", "weights/detector.pt")
    print("saved weights/detector.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)  # fits an 8 GB RTX 3070 Ti with yolov8s at 640
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/detector_cv"))
    ap.add_argument("--groups-file", default="groups.csv", help="CSV of image_stem,group inside --data")
    ap.add_argument("--only-folds", type=int, nargs="*", default=None, help="run just these fold indices (config probes)")
    ap.add_argument("--smoke", action="store_true", help="tiny run: no AMP check, no plots, no final fit")
    ap.add_argument("--final-only", action="store_true", help="skip CV; just fit the deployable detector")
    ap.add_argument("--final", action="store_true", help="also fit on all data and save weights/detector.pt")
    a = ap.parse_args()

    imgs, strata, groups = index_dataset(a.data, a.groups_file)
    if len(set(groups)) < a.folds:
        sys.exit(f"need at least {a.folds} distinct groups, found {len(set(groups))}")
    if a.final_only:
        fit_final(a, imgs, strata, groups)
        return
    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    labels_dir = a.data / "labels"
    pooled = {i: [] for i in range(len(DETECTOR_CLASSES))}
    pooled_gt = {i: 0 for i in range(len(DETECTOR_CLASSES))}
    for k, (tr, va) in enumerate(skf.split(imgs, strata, groups)):
        if a.only_folds is not None and k not in a.only_folds:
            continue
        fold_dir = a.out / f"fold{k}"
        dy = write_split(fold_dir, [imgs[i] for i in tr], [imgs[i] for i in va])
        model = train(a.weights, dy, fold_dir / "run", a)
        recs, n_gt = predict_records(model, [imgs[i] for i in va], labels_dir, a)
        for c in recs:
            pooled[c] += recs[c]
            pooled_gt[c] += n_gt[c]
        print(f"fold {k}: {len(tr)} train / {len(va)} val images")

    thresholds, per_class = {}, {}
    for i, cls in enumerate(DETECTOR_CLASSES):
        r = choose_threshold(pooled[i], pooled_gt[i], RECALL_TARGET[cls.value])
        thresholds[cls.value] = r["threshold"]
        per_class[cls.value] = {**r, "n_gt": pooled_gt[i], "recall_target": RECALL_TARGET[cls.value]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (a.out / "report.json").write_text(json.dumps(
        {"folds": a.folds, "n_images": len(imgs), "per_class_oof": per_class,
         "note": "thresholds tuned on pooled out-of-fold predictions"}, indent=2))
    for name, r in per_class.items():
        print(f"{name:18s} thr={r['threshold']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"(target {r['recall_target']:.2f}, n_gt={r['n_gt']})")

    if a.final and not a.smoke:
        fit_final(a, imgs, strata, groups)

if __name__ == "__main__":
    main()
