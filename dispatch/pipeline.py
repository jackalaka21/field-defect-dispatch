"""End-to-end: photo + voice memo -> validated ticket."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .assets import Asset, default_registry
from .fusion import (CaseEvidence, Decision, FusionConfig, PriorityModel, review_reasons, rule_priority)
from .schemas import Detection, Ticket, Transcript
from .text import extract_cues
from .writer import LLM, build_ticket, finalise_with_llm


class Detector(Protocol):
    def detect(self, image: str | Path) -> list[Detection]: ...


class Recogniser(Protocol):
    def transcribe(self, audio: str | Path) -> Transcript: ...


class DispatchPipeline:
    def __init__(self, detector: Detector, recogniser: Recogniser, model: PriorityModel | None = None,
                 registry: dict[str, Asset] | None = None, llm: LLM | None = None, cfg: FusionConfig = FusionConfig()):
        self.detector, self.recogniser, self.model = detector, recogniser, model
        self.registry = registry or default_registry()
        self.llm, self.cfg = llm, cfg

    def evidence(self, image: str | Path, audio: str | Path) -> CaseEvidence:
        transcript = self.recogniser.transcribe(audio)
        cues = extract_cues(transcript.text, self.registry)
        asset = self.registry.get(cues.asset_id) if cues.asset_id else None
        return CaseEvidence(detections=self.detector.detect(image), cues=cues, asr_confidence=transcript.confidence,
                            asset_criticality=asset.structural_criticality if asset else None)

    def decide(self, ev: CaseEvidence) -> Decision:
        if self.model is not None:
            return self.model.decide_many([ev])[0]
        reasons = review_reasons(ev, 1.0, self.cfg)  # no probabilities without a learned model
        return Decision(rule_priority(ev, self.cfg), 0.5, bool(reasons), reasons)

    def run(self, image: str | Path, audio: str | Path, ticket_id: str) -> Ticket:
        ev = self.evidence(image, audio)
        ticket = build_ticket(ev, self.decide(ev), ticket_id, self.cfg)
        return finalise_with_llm(ev, ticket, self.llm, cfg=self.cfg) if self.llm else ticket
