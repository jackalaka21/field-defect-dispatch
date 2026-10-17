import numpy as np
import pytest

from dispatch.det_eval import Record, choose_threshold, match_image, read_yolo_labels


def test_read_yolo_labels(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("2 0.5 0.5 0.2 0.4\n\n0 0.25 0.25 0.1 0.1\n")
    labels = read_yolo_labels(p)
    assert labels[0][0] == 2 and labels[0][1] == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert read_yolo_labels(tmp_path / "missing.txt") == []


def test_match_image_greedy_and_class_aware():
    gts = [(0, (0.1, 0.1, 0.5, 0.5)), (1, (0.6, 0.6, 0.9, 0.9))]
    preds = [(0, 0.9, (0.1, 0.1, 0.5, 0.5)),     # hit
             (0, 0.8, (0.1, 0.1, 0.5, 0.5)),     # duplicate of the same GT -> false positive
             (1, 0.7, (0.1, 0.1, 0.5, 0.5))]     # right place, wrong class -> false positive
    recs, n_gt = match_image(preds, gts)
    assert n_gt == {0: 1, 1: 1}
    by = {(r.cls, r.conf): r.tp for r in recs}
    assert by == {(0, 0.9): True, (0, 0.8): False, (1, 0.7): False}


def test_threshold_reaches_recall_with_best_precision():
    # 10 GT. Confident hits, then a noisy tail that supplies the last recall.
    recs = [Record(0, c, True) for c in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)]
    recs += [Record(0, 0.5, False), Record(0, 0.4, True), Record(0, 0.3, False), Record(0, 0.2, True)]
    r = choose_threshold(recs, n_gt=10, recall_target=0.8)
    assert r["reached_target"] and r["threshold"] == pytest.approx(0.6) and r["precision"] == 1.0
    # 12 GT but only 10 true positives exist, so recall 1.0 is unreachable -> falls back to best F2
    r = choose_threshold(recs, n_gt=12, recall_target=1.0)
    assert not r["reached_target"] and 0.0 < r["recall"] < 1.0


def test_threshold_degenerate_inputs():
    assert choose_threshold([], 5, 0.9)["recall"] == 0.0
    assert not choose_threshold([Record(0, 0.9, True)], 0, 0.9)["reached_target"]


def test_calibrator_is_monotone_clips_and_round_trips(tmp_path):
    from dispatch.calibration import Calibrator
    from dispatch.taxonomy import DETECTOR_CLASSES, DefectClass

    rng = np.random.default_rng(0)
    K = len(DETECTOR_CLASSES)
    scores = rng.uniform(0, 0.4, size=(400, K))
    present = rng.uniform(size=(400, K)) < (scores * 2.0)  # P(present) = 2 * score
    present[:, 5] = False  # a class with no labels stays uncalibrated
    cal = Calibrator.fit(scores, present, DETECTOR_CLASSES)
    c = DefectClass.EXPOSED_REBAR
    vals = [cal(c, v) for v in np.linspace(0, 0.4, 20)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert 0.0 <= min(vals) and max(vals) <= 1.0
    assert cal(c, 5.0) == cal(c, 0.4 + 1.0)  # out of range clips
    assert cal(DETECTOR_CLASSES[5], 0.37) == pytest.approx(0.37)  # untouched
    cal.save(tmp_path / "c.json")
    assert Calibrator.load(tmp_path / "c.json")(c, 0.2) == pytest.approx(cal(c, 0.2))
