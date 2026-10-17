"""Defect taxonomy, priority scale and the work-order catalog.

The detector classes deliberately include `rebar_corrosion` and `spalling`: a technician who says
"corrosion" about a photo of exposed rebar must be expressible in the vision label space too.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Priority(str, Enum):
    P1 = "P1"  # critical: act immediately
    P2 = "P2"  # high: within 24 h
    P3 = "P3"  # medium: within a week
    P4 = "P4"  # low: routine / cosmetic

    @property
    def rank(self) -> int:
        return PRIORITY_ORDER.index(self)

    @classmethod
    def from_rank(cls, rank: int) -> "Priority":
        return PRIORITY_ORDER[max(0, min(rank, len(PRIORITY_ORDER) - 1))]


PRIORITY_ORDER = [Priority.P1, Priority.P2, Priority.P3, Priority.P4]


class DefectClass(str, Enum):
    EXPOSED_REBAR = "exposed_rebar"
    REBAR_CORROSION = "rebar_corrosion"
    SPALLING = "spalling"
    STRUCTURAL_CRACK = "structural_crack"
    HAIRLINE_CRACK = "hairline_crack"
    WATER_LEAK = "water_leak"
    EFFLORESCENCE = "efflorescence"
    NO_DEFECT = "no_defect"


class DefectCategory(str, Enum):
    STRUCTURAL_HAZARD = "Structural Hazard"
    WATER_INGRESS = "Water Ingress"
    COSMETIC = "Cosmetic"
    NONE = "No Defect"


@dataclass(frozen=True)
class DefectProfile:
    category: DefectCategory
    floor: Priority  # the least urgent priority this defect may ever be given (safety floor)
    work_order: str  # key into WORK_ORDERS
    critical_work_order: str | None = None  # used when the ticket is escalated to P1


WORK_ORDERS: dict[str, str] = {
    "WO-STR-01": "Immediate shoring and rebar passivation",
    "WO-STR-02": "Rebar corrosion treatment and concrete reinstatement",
    "WO-STR-03": "Concrete patch repair and spall removal",
    "WO-STR-04": "Structural crack assessment and epoxy injection",
    "WO-STR-05": "Immediate area isolation and engineer inspection",
    "WO-WTR-01": "Leak source tracing and waterproofing repair",
    "WO-CSM-01": "Crack monitoring and sealant at next maintenance cycle",
    "WO-CSM-02": "Surface cleaning and efflorescence treatment",
    "WO-NON-01": "No action required",
}

PROFILES: dict[DefectClass, DefectProfile] = {
    DefectClass.EXPOSED_REBAR: DefectProfile(
        DefectCategory.STRUCTURAL_HAZARD, Priority.P2, "WO-STR-02", "WO-STR-01"
    ),
    DefectClass.REBAR_CORROSION: DefectProfile(
        DefectCategory.STRUCTURAL_HAZARD, Priority.P2, "WO-STR-02", "WO-STR-01"
    ),
    DefectClass.SPALLING: DefectProfile(
        DefectCategory.STRUCTURAL_HAZARD, Priority.P3, "WO-STR-03", "WO-STR-05"
    ),
    DefectClass.STRUCTURAL_CRACK: DefectProfile(
        DefectCategory.STRUCTURAL_HAZARD, Priority.P2, "WO-STR-04", "WO-STR-05"
    ),
    DefectClass.HAIRLINE_CRACK: DefectProfile(DefectCategory.COSMETIC, Priority.P4, "WO-CSM-01"),
    DefectClass.WATER_LEAK: DefectProfile(DefectCategory.WATER_INGRESS, Priority.P3, "WO-WTR-01"),
    DefectClass.EFFLORESCENCE: DefectProfile(DefectCategory.COSMETIC, Priority.P4, "WO-CSM-02"),
    DefectClass.NO_DEFECT: DefectProfile(DefectCategory.NONE, Priority.P4, "WO-NON-01"),
}

CLASSES: list[DefectClass] = list(DefectClass)
# Classes the detector is trained on (no_defect is the absence of detections).
DETECTOR_CLASSES: list[DefectClass] = [c for c in DefectClass if c is not DefectClass.NO_DEFECT]

# Words used to prime the speech recogniser and to repair its output.
GLOSSARY: list[str] = [
    "rebar", "corrosion", "spalling", "efflorescence", "hairline", "shoring", "passivation",
    "delamination", "housing", "pump", "column", "slab", "wall", "crack", "leak", "seepage",
    "exposed", "severe", "critical", "structural", "immediately", "urgent",
]
