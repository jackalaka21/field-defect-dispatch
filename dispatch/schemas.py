"""Pydantic contracts between pipeline stages and for the final ticket."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from .taxonomy import WORK_ORDERS, DefectCategory, DefectClass, Priority


class Detection(BaseModel):
    cls: DefectClass
    conf: float = Field(ge=0.0, le=1.0)
    # normalised xyxy in [0, 1]
    box: tuple[float, float, float, float]

    @field_validator("box")
    @classmethod
    def _box_ok(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = v
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"box must be normalised xyxy with x1<x2, y1<y2: {v}")
        return v

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)


class Transcript(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class Cues(BaseModel):
    """What the technician said, reduced to machine-usable signals."""

    mentions: list[DefectClass] = Field(default_factory=list)
    severity: float = Field(ge=-1.0, le=1.0, default=0.0)  # -1 trivial ... +1 severe
    urgent: bool = False
    asset_id: str | None = None


class Ticket(BaseModel):
    ticket_id: str
    asset_id: str | None
    priority: Priority
    defect_type: DefectClass
    category: DefectCategory
    work_order_id: str
    work_order: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    evidence: str

    @model_validator(mode="after")
    def _consistent(self) -> "Ticket":
        if self.work_order_id not in WORK_ORDERS:
            raise ValueError(f"unknown work order {self.work_order_id}")
        if WORK_ORDERS[self.work_order_id] != self.work_order:
            raise ValueError("work_order text does not match the catalog entry")
        if self.needs_review and not self.review_reasons:
            raise ValueError("needs_review requires at least one reason")
        return self
