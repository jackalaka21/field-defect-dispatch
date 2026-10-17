"""Ticket assembly. A deterministic writer is the default; an LLM writer is optional and always
validated against the Pydantic schema, retried with the validation error, floor-checked, and
replaced by the deterministic ticket if it never produces something valid."""
from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from .fusion import CaseEvidence, Decision, FusionConfig, resolve_defect_type, safety_floor
from .schemas import Ticket
from .taxonomy import PROFILES, WORK_ORDERS, Priority


def _evidence_line(ev: CaseEvidence, cls_name: str) -> str:
    top = ev.top
    parts = []
    if top is not None:
        parts.append(f"photo: {top.cls.value} {top.conf:.2f}")
    if ev.cues.mentions:
        parts.append("voice: " + ", ".join(m.value for m in ev.cues.mentions))
    parts.append(f"voice severity {ev.cues.severity:+.1f}")
    if ev.cues.urgent:
        parts.append("urgent wording")
    return "; ".join(parts) + f" -> {cls_name}"


def build_ticket(ev: CaseEvidence, decision: Decision, ticket_id: str, cfg: FusionConfig = FusionConfig()) -> Ticket:
    cls = resolve_defect_type(ev, cfg)
    profile = PROFILES[cls]
    wo_id = profile.critical_work_order if (decision.priority is Priority.P1 and profile.critical_work_order) else profile.work_order
    return Ticket(
        ticket_id=ticket_id,
        asset_id=ev.cues.asset_id,
        priority=decision.priority,
        defect_type=cls,
        category=profile.category,
        work_order_id=wo_id,
        work_order=WORK_ORDERS[wo_id],
        confidence=round(decision.confidence, 4),
        needs_review=decision.needs_review,
        review_reasons=decision.reasons,
        evidence=_evidence_line(ev, cls.value),
    )


LLM = Callable[[str], str]  # prompt -> raw model text (expected to be a JSON ticket)


def _prompt(ev: CaseEvidence, draft: Ticket, error: str | None) -> str:
    catalog = json.dumps(WORK_ORDERS)
    base = (
        "You finalise a maintenance dispatch ticket. Return ONLY a JSON object matching the draft's "
        "fields. You may correct wording in `evidence` but must choose work_order_id from the catalog "
        f"and must not make `priority` less urgent than the draft.\nCatalog: {catalog}\n"
        f"Evidence: {ev.model_dump_json()}\nDraft: {draft.model_dump_json()}\n"
    )
    return base + (f"Your previous answer was invalid: {error}\n" if error else "")


def finalise_with_llm(ev: CaseEvidence, draft: Ticket, llm: LLM, retries: int = 2,
                      cfg: FusionConfig = FusionConfig()) -> Ticket:
    floor = safety_floor(ev, cfg)
    error: str | None = None
    for _ in range(retries + 1):
        raw = llm(_prompt(ev, draft, error))
        try:
            ticket = Ticket.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            error = str(exc)[:300]
            continue
        if ticket.priority.rank > floor.rank:
            error = f"priority {ticket.priority.value} is below the safety floor {floor.value}"
            continue
        if ticket.priority.rank > draft.priority.rank:
            error = "priority may not be less urgent than the draft"
            continue
        return ticket
    return draft  # the deterministic ticket is always a safe answer
