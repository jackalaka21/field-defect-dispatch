import json

import pytest

from dispatch.fusion import CostMatrix, load_operating_point


def test_load_operating_point_scales_only_the_under_costs(tmp_path):
    p = tmp_path / "op.json"
    p.write_text(json.dumps({"under_scale": 3.0, "review_prob": 0.5}))
    cost, cfg = load_operating_point(p)
    base = CostMatrix()
    assert cost.under == pytest.approx(tuple(3.0 * u for u in base.under))
    assert cost.over == base.over
    assert cfg.review_prob == 0.5
