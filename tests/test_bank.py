import json

import numpy as np

from dispatch.calibration import Calibrator
from dispatch.synth.bank import DetectorBank
from dispatch.synth.cases import generate_cases
from dispatch.taxonomy import DETECTOR_CLASSES, DefectClass


def make_bank(tmp_path):
    K = len(DETECTOR_CLASSES)
    scores = np.zeros((10, K))
    present = np.zeros((10, K), bool)
    areas = np.full((10, K), 0.09)
    for i in range(5):  # rows 0-4: exposed_rebar photos, detector sees it at 0.5 and misses a second class
        present[i, 0] = True
        scores[i, 0] = 0.5
        scores[i, 2] = 0.01  # below threshold -> never reported
    scores[5:, 0] = 0.0  # rows 5-9: clean photos; row 9 is a false positive
    scores[9, 3] = 0.4
    np.savez(tmp_path / "b.npz", scores=scores, present=present, areas=areas, fold_sizes=np.array([2, 2, 2, 2, 2]))
    (tmp_path / "t.json").write_text(json.dumps({c.value: 0.1 for c in DETECTOR_CLASSES}))
    Calibrator({}).save(tmp_path / "c.json")  # identity calibration
    return DetectorBank(tmp_path / "b.npz", tmp_path / "t.json", tmp_path / "c.json")


def test_bank_replays_real_rows_and_respects_thresholds(tmp_path):
    bank = make_bank(tmp_path)
    rng = np.random.default_rng(0)
    d = bank.sample(DefectClass.EXPOSED_REBAR, rng)
    assert [(x.cls, round(x.conf, 2)) for x in d] == [(DefectClass.EXPOSED_REBAR, 0.5)]
    assert not bank.covers(DefectClass.WATER_LEAK) and bank.covers(DefectClass.NO_DEFECT)


def test_dev_and_holdout_views_share_no_rows(tmp_path):
    bank = make_bank(tmp_path)
    dev, hold = bank.view("dev"), bank.view("holdout")
    assert set(dev.by_class[DefectClass.EXPOSED_REBAR]) == {0, 1, 2, 3, 4} & set(range(6))
    assert not set(dev.clean) & set(hold.clean)
    assert max(dev.clean, default=-1) < 6 <= min(hold.clean, default=99)


def test_cases_use_the_bank_for_covered_classes(tmp_path):
    bank = make_bank(tmp_path).view("dev")
    cases = generate_cases(200, seed=1, bank=bank)
    rebar = [c for c in cases if c.true_class is DefectClass.EXPOSED_REBAR]
    assert rebar and all(d.conf == 0.5 for c in rebar for d in c.evidence.detections)
