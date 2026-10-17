"""CODEBRIM (Mundt et al., CVPR 2019) annotation -> YOLO labels.

Each annotated box carries multi-label flags (a box can be Spallation + ExposedBars + CorrosionStain at once).
It is expanded into one YOLO box per active flag, which is the usual way to train a single-label detector on
multi-label boxes. The <folder> field names the bridge/site and is used as the CV group.

CODEBRIM licence: non-commercial research and education only; do not redistribute the data; cite the paper.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .taxonomy import DETECTOR_CLASSES, DefectClass

# CODEBRIM has no hairline-crack or water-leak labels, and its generic "Crack" is mapped to the structural class
# (the conservative choice: cracks of unknown width should get an engineer's attention).
FLAG_TO_CLASS: dict[str, DefectClass] = {
    "ExposedBars": DefectClass.EXPOSED_REBAR,
    "CorrosionStain": DefectClass.REBAR_CORROSION,
    "Spallation": DefectClass.SPALLING,
    "Crack": DefectClass.STRUCTURAL_CRACK,
    "Efflorescence": DefectClass.EFFLORESCENCE,
}
CLASS_INDEX = {c: i for i, c in enumerate(DETECTOR_CLASSES)}

Annotation = tuple[str, int, int, list[tuple[int, tuple[float, float, float, float]]]]


def parse_annotation(xml_text: str) -> Annotation:
    """Return (site, width, height, [(class_index, (cx, cy, w, h) normalised)])."""
    root = ET.fromstring(xml_text)
    site = (root.findtext("folder") or "unknown").strip()
    W, H = int(root.findtext("size/width")), int(root.findtext("size/height"))
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1 = max(0.0, float(bb.findtext("xmin"))), max(0.0, float(bb.findtext("ymin")))
        x2, y2 = min(float(W), float(bb.findtext("xmax"))), min(float(H), float(bb.findtext("ymax")))
        if x2 <= x1 or y2 <= y1:
            continue
        flags = obj.find("Defect")
        if flags is None:
            continue
        xywh = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
        for flag, cls in FLAG_TO_CLASS.items():
            if (flags.findtext(flag) or "0").strip() == "1":
                boxes.append((CLASS_INDEX[cls], xywh))
    return site, W, H, boxes


def to_yolo_lines(boxes: list[tuple[int, tuple[float, float, float, float]]]) -> list[str]:
    return [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, (cx, cy, w, h) in boxes]
