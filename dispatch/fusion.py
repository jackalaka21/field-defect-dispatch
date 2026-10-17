"""Multimodal fusion: evidence -> features -> cost-sensitive priority with a hard safety floor.

Design rule: a learned model may *raise* urgency freely but can never go below the deterministic
safety floor derived from the defect profile, and the decision minimises expected cost under an
asymmetric matrix (missing a P1 is far worse than over-escalating a P4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from .assets import Asset
from .schemas import Cues, Detection
from .taxonomy import DETECTOR_CLASSES, PROFILES, DefectClass, DefectCategory, Priority

N_PRIORITIES = 4


class CaseEvidence(BaseModel):
    detections: list[Detection] = []
    cues: Cues = Cues()
    asr_confidence: float = 1.0
    asset_criticality: float | None = None  # None when the asset could not be resolved

    @property
    def top(self) -> Detection | None:
        return max(self.detections, key=lambda d: d.conf, default=None)


@dataclass(frozen=True)
class FusionConfig:
    det_floor_conf: float = 0.6  # a detection this confident can impose its safety floor
    det_type_conf: float = 0.5  # ...and can name the defect type
    mild_voice: float = -0.3  # voice severity at or below this releases the voice-based floor
    review_prob: float = 0.55  # max class probability below this -> human review
    min_asr_conf: float = 0.5


@dataclass(frozen=True)
class CostMatrix:
    """cost[true_rank][pred_rank]. Under-prioritising is weighted by how urgent the truth was."""

    under: tuple[float, ...] = (10.0, 5.0, 2.5, 1.0)  # per level of under-prioritisation, by true rank
    over: float = 1.0  # per level of over-prioritisation

    def matrix(self) -> np.ndarray:
        m = np.zeros((N_PRIORITIES, N_PRIORITIES))
        for t in range(N_PRIORITIES):
            for p in range(N_PRIORITIES):
                m[t, p] = self.under[t] * (p - t) if p > t else self.over * (t - p)
        return m


def load_operating_point(path) -> tuple["CostMatrix", "FusionConfig"]:
    """Cost matrix and review threshold chosen by scripts/cv_fusion.py on out-of-fold data."""
    import json
    from dataclasses import replace
    from pathlib import Path

    raw = json.loads(Path(path).read_text())
    base = CostMatrix()
    return (CostMatrix(under=tuple(u * raw["under_scale"] for u in base.under), over=base.over),
            replace(FusionConfig(), review_prob=raw["review_prob"]))


# --------------------------------------------------------------------------- features
FEATURE_NAMES: list[str] = (
    [f"conf_{c.value}" for c in DETECTOR_CLASSES]
    + [f"top_is_{c.value}" for c in DETECTOR_CLASSES]
    + ["top_conf", "top_area", "n_dets"]
    + [f"said_{c.value}" for c in DefectClass]
    + ["voice_severity", "voice_urgent", "asset_criticality", "asset_resolved", "asr_conf",
       "agree", "conflict", "det_floor_rank", "voice_floor_rank"]
)


def _floor_rank(cls: DefectClass) -> int:
    return PROFILES[cls].floor.rank


def build_features(ev: CaseEvidence) -> np.ndarray:
    top = ev.top
    conf_by_class = {c: 0.0 for c in DETECTOR_CLASSES}
    for d in ev.detections:
        if d.cls in conf_by_class:
            conf_by_class[d.cls] = max(conf_by_class[d.cls], d.conf)
    f: list[float] = [conf_by_class[c] for c in DETECTOR_CLASSES]
    f += [1.0 if top is not None and top.cls is c else 0.0 for c in DETECTOR_CLASSES]
    f += [top.conf if top else 0.0, top.area if top else 0.0, float(len(ev.detections))]
    mentioned = set(ev.cues.mentions)
    f += [1.0 if c in mentioned else 0.0 for c in DefectClass]
    real_mentions = [m for m in mentioned if m is not DefectClass.NO_DEFECT]
    agree = float(top is not None and top.cls in mentioned)
    conflict = float(top is not None and bool(real_mentions) and top.cls not in mentioned)
    f += [
        ev.cues.severity,
        float(ev.cues.urgent),
        ev.asset_criticality if ev.asset_criticality is not None else 0.0,
        float(ev.asset_criticality is not None),
        ev.asr_confidence,
        agree,
        conflict,
        float(_floor_rank(top.cls)) if top else 3.0,
        float(min((_floor_rank(m) for m in real_mentions), default=3)),
    ]
    out = np.asarray(f, dtype=np.float64)
    assert out.shape[0] == len(FEATURE_NAMES)
    return out


def build_matrix(evidence: list[CaseEvidence]) -> np.ndarray:
    return np.vstack([build_features(e) for e in evidence])


# --------------------------------------------------------------------------- rules
def candidate_classes(ev: CaseEvidence, cfg: FusionConfig = FusionConfig()) -> list[DefectClass]:
    """Classes the evidence supports, detector first."""
    out: list[DefectClass] = []
    top = ev.top
    if top is not None and top.conf >= cfg.det_type_conf and top.cls is not DefectClass.NO_DEFECT:
        out.append(top.cls)
    for m in ev.cues.mentions:
        if m is not DefectClass.NO_DEFECT and m not in out:
            out.append(m)
    return out


def safety_floor(ev: CaseEvidence, cfg: FusionConfig = FusionConfig()) -> Priority:
    """Least urgent priority permitted. A confident detector hit or a non-dismissive voice mention
    of a hazard class pins the ticket at or above that class's floor."""
    ranks: list[int] = []
    top = ev.top
    if top is not None and top.conf >= cfg.det_floor_conf:
        ranks.append(_floor_rank(top.cls))
    if ev.cues.severity > cfg.mild_voice:
        ranks += [_floor_rank(m) for m in ev.cues.mentions]
    return Priority.from_rank(min(ranks, default=3))


def rule_priority(ev: CaseEvidence, cfg: FusionConfig = FusionConfig()) -> Priority:
    """Deterministic baseline: the floor, escalated by voice severity/urgency and asset criticality."""
    floor = safety_floor(ev, cfg).rank
    rank = floor
    cands = candidate_classes(ev, cfg)
    structural = any(PROFILES[c].category is DefectCategory.STRUCTURAL_HAZARD for c in cands)
    if structural:
        if ev.cues.severity >= 0.6 or ev.cues.urgent:
            rank -= 1
        if ev.asset_criticality is not None and ev.asset_criticality >= 0.9:
            rank -= 1
    if ev.cues.severity <= -0.6 and not ev.cues.urgent:
        rank += 1
    # downgrades from dismissive wording may never undercut the floor imposed by a confident hit
    return Priority.from_rank(min(rank, floor))


def resolve_defect_type(ev: CaseEvidence, cfg: FusionConfig = FusionConfig()) -> DefectClass:
    cands = candidate_classes(ev, cfg)
    if not cands:
        return DefectClass.NO_DEFECT
    return min(cands, key=lambda c: (_floor_rank(c), cands.index(c)))


# --------------------------------------------------------------------------- decision
@dataclass
class Decision:
    priority: Priority
    confidence: float
    needs_review: bool
    reasons: list[str] = field(default_factory=list)
    probs: np.ndarray | None = None


def expected_cost_rank(proba: np.ndarray, cost: CostMatrix) -> int:
    """argmin over predicted rank of sum_t p(t) * cost[t, pred]."""
    return int(np.argmin(proba @ cost.matrix()))


def review_reasons(ev: CaseEvidence, max_prob: float, cfg: FusionConfig = FusionConfig()) -> list[str]:
    reasons: list[str] = []
    top = ev.top
    real = [m for m in ev.cues.mentions if m is not DefectClass.NO_DEFECT]
    if max_prob < cfg.review_prob:
        reasons.append(f"low priority confidence ({max_prob:.2f})")
    if top is not None and top.conf >= cfg.det_floor_conf and real and top.cls not in real:
        reasons.append("photo and voice name different defects")
    if top is not None and top.conf >= cfg.det_floor_conf and DefectClass.NO_DEFECT in ev.cues.mentions:
        reasons.append("photo shows a defect but voice reports none")
    if ev.cues.asset_id is None or ev.asset_criticality is None:
        reasons.append("asset not resolved against the registry")
    if ev.asr_confidence < cfg.min_asr_conf:
        reasons.append(f"low transcription confidence ({ev.asr_confidence:.2f})")
    return reasons


class PriorityModel:
    """Wraps any sklearn classifier exposing predict_proba. Fit on feature rows and rank labels."""

    def __init__(self, estimator, cost: CostMatrix = CostMatrix(), cfg: FusionConfig = FusionConfig()):
        self.estimator = estimator
        self.cost = cost
        self.cfg = cfg

    def fit(self, X: np.ndarray, y_rank: np.ndarray) -> "PriorityModel":
        self.estimator.fit(X, y_rank)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.estimator.predict_proba(X)
        full = np.zeros((raw.shape[0], N_PRIORITIES))
        for j, c in enumerate(self.estimator.classes_):
            full[:, int(c)] = raw[:, j]
        return full

    def decide_many(self, evidence: list[CaseEvidence]) -> list[Decision]:
        proba = self.predict_proba(build_matrix(evidence))
        out: list[Decision] = []
        for ev, p in zip(evidence, proba):
            rank = expected_cost_rank(p, self.cost)
            rank = min(rank, safety_floor(ev, self.cfg).rank)  # never less urgent than the floor
            reasons = review_reasons(ev, float(p.max()), self.cfg)
            out.append(Decision(Priority.from_rank(rank), float(p.max()), bool(reasons), reasons, p))
        return out
