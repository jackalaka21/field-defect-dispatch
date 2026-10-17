"""Extract CODEBRIM_original_images.zip into a YOLO dataset (downscaled) with a per-image site list.

    python scripts/prepare_codebrim.py --zip data/raw/codebrim/CODEBRIM_original_images.zip --out data/defects
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from dispatch.codebrim import parse_annotation, to_yolo_lines  # noqa: E402
from dispatch.taxonomy import DETECTOR_CLASSES  # noqa: E402


def open_zip_fixing_offsets(path: Path) -> zipfile.ZipFile:
    """The CODEBRIM archive stores header offsets modulo 2**32 (no zip64 records) although it is 8 GB.

    zipfile guesses one constant "prepended data" shift from the end-of-directory record and adds it to every
    entry, which is right only after the wrap. Undo that shift, then restore the lost high bits per entry:
    directory entries are in file order, so a drop in the stored offset marks a wrap."""
    zf = zipfile.ZipFile(path)
    infos = zf.infolist()
    shift = infos[0].header_offset  # the first entry truly starts at 0
    wraps, prev = 0, -1
    for info in infos:
        stored = info.header_offset - shift
        if stored < prev:
            wraps += 1
        prev = stored
        info.header_offset = stored + (wraps << 32)
    return zf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/defects"))
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--with-negatives", action="store_true",
                    help="also convert photos that have no XML as empty-label background images (group 'neg')")
    a = ap.parse_args()

    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(parents=True, exist_ok=True)
    zf = open_zip_fixing_offsets(a.zip)
    names = [n for n in zf.namelist() if not n.startswith("__MACOSX") and "/._" not in n]
    xmls = {Path(n).stem: n for n in names if n.endswith(".xml")}
    jpgs = {Path(n).stem: n for n in names if n.lower().endswith(".jpg")}
    rows, per_class, skipped = [], Counter(), Counter()
    for stem, xml_name in sorted(xmls.items()):
        if stem not in jpgs:
            skipped["xml_without_image"] += 1
            continue
        site, W, H, boxes = parse_annotation(zf.read(xml_name).decode("utf-8"))
        if not boxes:
            skipped["no_defect_boxes"] += 1
            continue
        img = Image.open(io.BytesIO(zf.read(jpgs[stem])))
        if img.size != (W, H):
            skipped["size_mismatch"] += 1
            continue
        img.draft("RGB", (a.max_side, a.max_side))  # fast JPEG-side downscale
        img = img.convert("RGB")
        img.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
        img.save(a.out / "images" / f"{stem}.jpg", quality=92)
        (a.out / "labels" / f"{stem}.txt").write_text("\n".join(to_yolo_lines(boxes)) + "\n")
        rows.append((stem, site))
        for c, _ in boxes:
            per_class[DETECTOR_CLASSES[c].value] += 1
    n_neg = 0
    if a.with_negatives:
        # No XML means no annotated defect boxes. The photos look like plain concrete, but a few may hold faint
        # unlabelled stains, so treat them as noisy negatives.
        for stem in sorted(set(jpgs) - set(xmls)):
            img = Image.open(io.BytesIO(zf.read(jpgs[stem])))
            img.draft("RGB", (a.max_side, a.max_side))
            img = img.convert("RGB")
            img.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
            img.save(a.out / "images" / f"{stem}.jpg", quality=92)
            (a.out / "labels" / f"{stem}.txt").write_text("")
            rows.append((stem, "neg"))
            n_neg += 1
    with (a.out / "groups.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    summary = {"images_converted": len(rows), "sites": len({s for _, s in rows}),
               "images_without_xml": len(set(jpgs) - set(xmls)), "negatives_added": n_neg, "boxes_per_class": dict(per_class),
               "skipped": dict(skipped), "max_side": a.max_side}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
