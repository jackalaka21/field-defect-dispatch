import numpy as np
import pytest

from dispatch.stacking import ImageScorer, class_stats, design


def test_class_stats_summarises_boxes_per_class():
    cls = np.array([0, 0, 0, 0, 2])
    conf = np.array([0.9, 0.5, 0.2, 0.05, 0.3])
    area = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    f = class_stats(cls, conf, area, 3)
    assert f[0].tolist() == pytest.approx([0.9, 0.9 + 0.5 + 0.2, 3.0, 0.1])  # top-3 sum, 3 boxes >= 0.1, top box area
    assert f[1].tolist() == [0, 0, 0, 0]
    assert f[2].tolist() == pytest.approx([0.3, 0.3, 1.0, 0.5])


def test_design_shape():
    assert design(np.zeros((7, 5, 4))).shape == (7, 21)


def test_scorer_learns_co_occurrence_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    n, K = 600, 3
    present = np.zeros((n, K), bool)
    present[:, 0] = rng.random(n) < 0.4
    present[:, 1] = present[:, 0] & (rng.random(n) < 0.8)  # class 1 co-occurs with class 0
    present[:, 2] = False  # unlabelled class
    F = np.zeros((n, K, 4))
    F[:, 0, 0] = np.where(present[:, 0], rng.uniform(0.3, 0.9, n), rng.uniform(0, 0.4, n))
    F[:, 1, 0] = rng.uniform(0, 0.3, n)  # class 1's own boxes are uninformative; class 0's are the clue
    sc = ImageScorer.fit(F, present)
    p = sc.predict_proba(F)
    assert p[:, 2].max() == 0.0  # never reported
    assert p[present[:, 1], 1].mean() > p[~present[:, 1], 1].mean() + 0.2
    sc.save(tmp_path / "s.joblib")
    assert np.allclose(ImageScorer.load(tmp_path / "s.joblib").predict_proba(F), p)
