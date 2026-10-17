"""Synthetic inspection cases for developing and cross-validating the fusion stage.

This is a *fixture*: the numbers it produces validate the machinery (leakage-safe CV, cost-sensitive
decisions, safety floor), not real-world accuracy. Ground truth comes from a latent expert policy that
uses information (true severity, site strictness) the pipeline never sees directly, so fusion has to
infer it from noisy detector output, noisy speech and the asset registry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assets import Asset, default_registry
from ..fusion import CaseEvidence
from ..schemas import Detection
from ..taxonomy import PROFILES, DefectCategory, DefectClass, Priority
from ..text import extract_cues

D = DefectClass

_PRIORS = {D.EXPOSED_REBAR: .14, D.REBAR_CORROSION: .14, D.SPALLING: .12, D.STRUCTURAL_CRACK: .10,
           D.HAIRLINE_CRACK: .14, D.WATER_LEAK: .12, D.EFFLORESCENCE: .10, D.NO_DEFECT: .14}
_BASE_RISK = {D.EXPOSED_REBAR: 2.2, D.REBAR_CORROSION: 1.9, D.SPALLING: 1.4, D.STRUCTURAL_CRACK: 2.0,
              D.HAIRLINE_CRACK: 0.2, D.WATER_LEAK: 1.0, D.EFFLORESCENCE: 0.1, D.NO_DEFECT: 0.0}
_RECALL = {D.EXPOSED_REBAR: .90, D.REBAR_CORROSION: .80, D.SPALLING: .85, D.STRUCTURAL_CRACK: .80,
           D.HAIRLINE_CRACK: .60, D.WATER_LEAK: .85, D.EFFLORESCENCE: .85}
_CONFUSE = {D.EXPOSED_REBAR: (D.REBAR_CORROSION, .18), D.REBAR_CORROSION: (D.EXPOSED_REBAR, .18),
            D.SPALLING: (D.EXPOSED_REBAR, .10), D.STRUCTURAL_CRACK: (D.HAIRLINE_CRACK, .20),
            D.HAIRLINE_CRACK: (D.STRUCTURAL_CRACK, .12), D.WATER_LEAK: (D.EFFLORESCENCE, .12),
            D.EFFLORESCENCE: (D.WATER_LEAK, .12)}
_PHRASES = {
    D.EXPOSED_REBAR: ["exposed rebar", "rebar showing through the concrete"],
    D.REBAR_CORROSION: ["rebar corrosion", "corroded rebar", "heavy rust on the steel"],
    D.SPALLING: ["spalling", "concrete spalling", "delamination"],
    D.STRUCTURAL_CRACK: ["structural crack", "wide shear crack"],
    D.HAIRLINE_CRACK: ["hairline crack", "surface crack"],
    D.WATER_LEAK: ["water leak", "seepage"],
    D.EFFLORESCENCE: ["efflorescence", "white staining"],
    D.NO_DEFECT: ["no defect", "all clear"],
}
_SEVERE_WORDS = ["severe", "major", "extensive", "critical", "serious"]
_MILD_WORDS = ["minor", "small", "slight", "tiny"]
_ASSET_NOUN = {"B": "pump housing", "C": "column", "S": "slab", "W": "wall"}
_SPOKEN_LETTER = {"B": ["B", "be", "bee"], "C": ["C", "see"], "S": ["S", "es"], "W": ["W", "W"]}
_SPOKEN_NUM = {1: ["one", "won"], 2: ["two", "to"], 3: ["three"], 4: ["four", "for"], 5: ["five"],
               6: ["six"], 7: ["seven"], 8: ["eight", "ate"], 9: ["nine"], 10: ["ten"], 11: ["eleven"],
               12: ["twelve"]}


@dataclass
class Case:
    case_id: int
    site: int
    true_class: DefectClass
    severity: int
    asset: Asset
    true_priority: Priority
    evidence: CaseEvidence
    clean_text: str  # what the technician actually said (TTS source)
    asr_text: str  # after simulated recogniser errors


def _spoken_asset(rng: np.random.Generator, asset: Asset) -> str:
    letter, num = asset.asset_id[0], int(asset.asset_id[1:])
    noun = _ASSET_NOUN[letter]
    style = rng.integers(0, 3)
    if style == 0:
        ident = asset.asset_id
    elif style == 1:
        ident = f"{letter} {num}"
    else:
        ident = f"{rng.choice(_SPOKEN_LETTER[letter])} {rng.choice(_SPOKEN_NUM[num])}"
    return f"{noun} {ident}"


def _corrupt(rng: np.random.Generator, text: str, p: float) -> str:
    out = []
    for w in text.split(" "):
        if len(w) >= 5 and w.isalpha() and rng.random() < p:
            op = rng.integers(0, 3)
            i = int(rng.integers(1, len(w) - 1))
            if op == 0:
                w = w[:i] + w[i + 1:]  # dropped letter
            elif op == 1:
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:] if i + 2 <= len(w) else w  # swapped letters
            else:
                w = w[:i] + " " + w[i:]  # split word
        out.append(w)
    return " ".join(out)


def _detections(rng: np.random.Generator, true_class: DefectClass, sev: int) -> list[Detection]:
    dets: list[Detection] = []

    def box(area: float) -> tuple[float, float, float, float]:
        w = float(np.clip(np.sqrt(area) * rng.uniform(0.7, 1.3), 0.05, 0.9))
        h = float(np.clip(area / w, 0.05, 0.9))
        x1 = float(rng.uniform(0, 1 - w))
        y1 = float(rng.uniform(0, 1 - h))
        return (x1, y1, x1 + w, y1 + h)

    if true_class is not DefectClass.NO_DEFECT and rng.random() < min(0.98, _RECALL[true_class] * (0.75 + 0.05 * sev)):
        partner, p_conf = _CONFUSE[true_class]
        if rng.random() < p_conf:
            cls, conf = partner, float(rng.beta(3, 3))
        else:
            cls, conf = true_class, float(rng.beta(6 + sev, 2))
        dets.append(Detection(cls=cls, conf=float(np.clip(conf, 0.01, 0.99)),
                              box=box(0.02 + 0.05 * sev * rng.uniform(0.5, 1.5))))
    if rng.random() < (0.2 if true_class is DefectClass.NO_DEFECT else 0.10):
        fp = list(_RECALL)[int(rng.integers(0, len(_RECALL)))]
        dets.append(Detection(cls=fp, conf=float(np.clip(rng.beta(2.5, 3.5), 0.01, 0.99)),
                              box=box(float(rng.uniform(0.01, 0.08)))))
    return dets


def generate_cases(n: int, seed: int = 0, n_sites: int = 40, asr_error: float = 0.12) -> list[Case]:
    rng = np.random.default_rng(seed)
    registry = default_registry()
    assets = list(registry.values())
    classes = list(_PRIORS)
    probs = np.array([_PRIORS[c] for c in classes])
    probs = probs / probs.sum()
    site_bias = rng.normal(0.0, 0.25, size=n_sites)
    cases: list[Case] = []
    for i in range(n):
        site = int(rng.integers(0, n_sites))
        cls = classes[int(rng.choice(len(classes), p=probs))]
        hazard = PROFILES[cls].category is DefectCategory.STRUCTURAL_HAZARD
        sev = int(np.clip(round(rng.normal(3.4 if hazard else 2.4, 1.1)), 1, 5))
        w = np.array([1.5 if (hazard and a.structural_criticality > 0.7) else 1.0 for a in assets])
        asset = assets[int(rng.choice(len(assets), p=w / w.sum()))]
        score = (_BASE_RISK[cls] + 0.45 * (sev - 3)
                 + (1.1 * (asset.structural_criticality - 0.5) if hazard else 0.0)
                 + site_bias[site] + rng.normal(0, 0.2))
        pri = Priority.P1 if score >= 3.0 else Priority.P2 if score >= 2.0 else Priority.P3 if score >= 1.0 else Priority.P4

        # what the technician says
        said = cls
        if rng.random() < 0.10:
            said = None  # vague: "some damage"
        elif cls is not DefectClass.NO_DEFECT and rng.random() < 0.05:
            said = _CONFUSE[cls][0]
        parts = []
        sev_word = ""
        r = rng.random()
        if r < 0.10:
            sev_word = str(rng.choice(_MILD_WORDS if sev >= 3 else _SEVERE_WORDS))  # noise
        elif sev >= 4 and r < 0.75:
            sev_word = str(rng.choice(_SEVERE_WORDS))
        elif sev <= 2 and r < 0.70:
            sev_word = str(rng.choice(_MILD_WORDS))
        if said is None:
            parts.append(f"Found {sev_word + ' ' if sev_word else ''}damage")
        else:
            parts.append(f"Found {sev_word + ' ' if sev_word else ''}{rng.choice(_PHRASES[said])}")
        if rng.random() < 0.93:
            parts.append(f"near {_spoken_asset(rng, asset)}")
        if rng.random() < (0.5 if sev == 5 else 0.3 if pri is Priority.P1 else 0.03):
            parts.append(str(rng.choice(["needs attention immediately", "this is urgent"])))
        clean = " ".join(parts)
        asr = _corrupt(rng, clean, asr_error)

        cues = extract_cues(asr, registry)
        resolved = registry.get(cues.asset_id) if cues.asset_id else None
        n_err = sum(1 for a, b in zip(clean.split(), asr.split()) if a != b)
        asr_conf = float(np.clip(rng.normal(0.88 - 0.12 * n_err, 0.06), 0.05, 0.99))
        ev = CaseEvidence(
            detections=_detections(rng, cls, sev), cues=cues, asr_confidence=asr_conf,
            asset_criticality=resolved.structural_criticality if resolved else None,
        )
        cases.append(Case(i, site, cls, sev, asset, pri, ev, clean, asr))
    return cases
