"""Write CV group files for the converted CODEBRIM set.

CODEBRIM's <folder> field identifies only 3 survey sessions, too few for k-fold CV. Photo IDs are sequential
and neighbouring IDs are near-duplicate shots of one structure, so the default grouping is blocks of consecutive
IDs. `groups_survey.csv` keeps the 3 surveys for a stricter leave-one-survey-out check.

    python scripts/make_groups.py --data data/defects --block 20
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/defects"))
    ap.add_argument("--block", type=int, default=20)
    a = ap.parse_args()
    src = a.data / "groups.csv"
    if not (a.data / "groups_survey.csv").exists():
        (a.data / "groups_survey.csv").write_text(src.read_text())  # keep the original survey grouping
    rows = list(csv.reader((a.data / "groups_survey.csv").open()))
    block = [(stem, f"blk{int(re.search(r'(\d+)', stem).group(1)) // a.block:03d}") for stem, _ in rows]
    with (a.data / "groups_block.csv").open("w", newline="") as f:
        csv.writer(f).writerows(block)
    print(f"{len(rows)} images, {len({g for _, g in block})} block groups of {a.block} consecutive ids, "
          f"{len({s for _, s in rows})} surveys")


if __name__ == "__main__":
    main()
