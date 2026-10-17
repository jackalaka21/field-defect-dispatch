"""Toy defect images in YOLO format. Used ONLY to smoke-test the detector training/CV code path
without downloading anything; the resulting detector is meaningless and must never be reported."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..taxonomy import DETECTOR_CLASSES

_COLOURS = [(150, 60, 40), (120, 70, 20), (90, 90, 90), (30, 30, 30), (200, 200, 200), (40, 60, 160), (235, 235, 225)]


def make_toy_dataset(root: Path, n: int = 60, size: int = 256, seed: int = 0, n_sites: int = 6) -> Path:
    from PIL import Image

    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        site = int(rng.integers(0, n_sites))
        img = np.clip(rng.normal(128, 18, (size, size, 3)), 0, 255).astype(np.uint8)
        lines = []
        for _ in range(int(rng.integers(1, 3))):
            c = int(rng.integers(0, len(DETECTOR_CLASSES)))
            w, h = int(rng.integers(30, 90)), int(rng.integers(30, 90))
            x, y = int(rng.integers(0, size - w)), int(rng.integers(0, size - h))
            img[y:y + h, x:x + w] = np.array(_COLOURS[c], dtype=np.uint8)
            lines.append(f"{c} {(x + w / 2) / size:.5f} {(y + h / 2) / size:.5f} {w / size:.5f} {h / size:.5f}")
        stem = f"site{site}_img{i:04d}"
        Image.fromarray(img).save(root / "images" / f"{stem}.jpg", quality=95)
        (root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        rows.append(stem)
    return root
