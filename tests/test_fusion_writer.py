import json

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

from dispatch.fusion import (CaseEvidence, CostMatrix, PriorityModel, build_features, build_matrix,
                             expected_cost_rank, resolve_defect_type, rule_priority, safety_floor, FEATURE_NAMES)
from dispatch.metrics import keyword_error_rate, priority_metrics, word_error_rate
from dispatch.schemas import Cues, Detection, Ticket
from dispatch.synth.cases import generate_cases
from dispatch.taxonomy import DefectClass, Priority
from dispatch.writer import build_ticket, finalise_with_llm
from dispatch.fusion import Decision

D = DefectClass


def det(cls, conf=0.9, box=(0.1, 0.1, 0.4, 0.4)):
    return Detection(cls=cls, conf=conf, box=box)


def test_detection_box_validation():
    with pytest.raises(ValidationError):
        Detection(cls=D.SPALLING, conf=0.5, box=(0.5, 0.1, 0.4, 0.4))
    with pytest.raises(ValidationError):
        Detection(cls=D.SPALLING, conf=1.5, box=(0.1, 0.1, 0.4, 0.4))


def test_feature_vector_shape_and_agreement_flags():
    ev = CaseEvidence(detections=[det(D.EXPOSED_REBAR)], cues=Cues(mentions=[D.REBAR_CORROSION]))
    f = dict(zip(FEATURE_NAMES, build_features(ev)))
    assert f["conflict"] == 1.0 and f["agree"] == 0.0
    ev2 = CaseEvidence(detections=[det(D.EXPOSED_REBAR)], cues=Cues(mentions=[D.EXPOSED_REBAR]))
    assert dict(zip(FEATURE_NAMES, build_features(ev2)))["agree"] == 1.0
    assert build_matrix([ev, ev2]).shape == (2, len(FEATURE_NAMES))


def test_floor_pins_confident_hazards_but_not_weak_detections():
    assert safety_floor(CaseEvidence(detections=[det(D.EXPOSED_REBAR, 0.91)])) is Priority.P2
    assert safety_floor(CaseEvidence(detections=[det(D.EXPOSED_REBAR, 0.30)])) is Priority.P4
    assert safety_floor(CaseEvidence(detections=[det(D.EFFLORESCENCE, 0.95)])) is Priority.P4


def test_floor_released_by_dismissive_voice():
    ev = CaseEvidence(cues=Cues(mentions=[D.EXPOSED_REBAR], severity=-0.7))
    assert safety_floor(ev) is Priority.P4
    ev = CaseEvidence(cues=Cues(mentions=[D.EXPOSED_REBAR], severity=0.0))
    assert safety_floor(ev) is Priority.P2


def test_headline_example_is_p1_with_the_critical_work_order():
    ev = CaseEvidence(
        detections=[det(D.EXPOSED_REBAR, 0.91)],
        cues=Cues(mentions=[D.REBAR_CORROSION], severity=0.9, asset_id="B4"),
        asset_criticality=0.95,
    )
    assert rule_priority(ev) is Priority.P1
    t = build_ticket(ev, Decision(Priority.P1, 0.9, False), "T-1")
    assert t.work_order_id == "WO-STR-01" and "shoring" in t.work_order.lower()
    assert t.category.value == "Structural Hazard"


def test_resolve_defect_type_prefers_hazard_and_handles_empty():
    assert resolve_defect_type(CaseEvidence()) is D.NO_DEFECT
    ev = CaseEvidence(detections=[det(D.EFFLORESCENCE, 0.8)], cues=Cues(mentions=[D.EXPOSED_REBAR]))
    assert resolve_defect_type(ev) is D.EXPOSED_REBAR


def test_cost_sensitive_decision_escalates_when_p1_is_plausible():
    cost = CostMatrix()
    # argmax would say P2 (rank 1) but a 35% chance of P1 makes escalating cheaper
    assert expected_cost_rank(np.array([0.35, 0.45, 0.15, 0.05]), cost) == 0
    # certainty keeps the low priority
    assert expected_cost_rank(np.array([0.0, 0.0, 0.05, 0.95]), cost) == 3


def test_model_never_goes_below_floor_and_flags_review():
    cases = generate_cases(600, seed=3)
    ev = [c.evidence for c in cases]
    y = np.array([c.true_priority.rank for c in cases])
    model = PriorityModel(LogisticRegression(max_iter=2000)).fit(build_matrix(ev), y)
    for e, d in zip(ev, model.decide_many(ev)):
        assert d.priority.rank <= safety_floor(e).rank
        assert d.needs_review == bool(d.reasons)


def test_ticket_schema_rejects_inconsistent_work_order():
    base = dict(ticket_id="T", asset_id=None, priority="P1", defect_type="spalling", category="Structural Hazard",
                work_order_id="WO-STR-03", work_order="Concrete patch repair and spall removal",
                confidence=0.5, needs_review=False, review_reasons=[], evidence="x")
    Ticket(**base)
    with pytest.raises(ValidationError):
        Ticket(**{**base, "work_order": "something invented"})
    with pytest.raises(ValidationError):
        Ticket(**{**base, "work_order_id": "WO-FAKE-99"})
    with pytest.raises(ValidationError):
        Ticket(**{**base, "needs_review": True})


def _ev():
    return CaseEvidence(detections=[det(D.EXPOSED_REBAR, 0.9)], cues=Cues(mentions=[D.EXPOSED_REBAR], asset_id="C1"),
                        asset_criticality=0.95)


def test_llm_retries_on_invalid_json_then_accepts():
    ev = _ev()
    draft = build_ticket(ev, Decision(Priority.P1, 0.9, False), "T-1")
    replies = iter(["not json at all", draft.model_dump_json()])
    out = finalise_with_llm(ev, draft, lambda p: next(replies))
    assert out == draft


def test_llm_cannot_lower_priority_below_floor_or_draft():
    ev = _ev()
    draft = build_ticket(ev, Decision(Priority.P1, 0.9, False), "T-1")
    downgraded = json.loads(draft.model_dump_json()) | {"priority": "P3"}
    out = finalise_with_llm(ev, draft, lambda p: json.dumps(downgraded), retries=1)
    assert out == draft  # every attempt rejected -> deterministic ticket


def test_llm_cannot_invent_work_orders():
    ev = _ev()
    draft = build_ticket(ev, Decision(Priority.P1, 0.9, False), "T-1")
    bad = json.loads(draft.model_dump_json()) | {"work_order_id": "WO-MADE-UP"}
    assert finalise_with_llm(ev, draft, lambda p: json.dumps(bad), retries=0) == draft


def test_metrics():
    y = np.array([0, 0, 1, 3])
    m = priority_metrics(y, np.array([0, 2, 1, 3]))
    assert m["critical_miss_rate"] == 0.5 and m["under_rate"] == 0.25 and m["accuracy"] == 0.75
    assert word_error_rate("a b c d", "a b x d") == 0.25
    assert keyword_error_rate("severe rebar corrosion", "severe re bar corrosion", {"rebar", "corrosion"}) == (1, 2)


def test_generator_is_deterministic_and_covers_all_priorities():
    a, b = generate_cases(400, seed=5), generate_cases(400, seed=5)
    assert [c.true_priority for c in a] == [c.true_priority for c in b]
    assert {c.true_priority for c in a} == set(Priority)
