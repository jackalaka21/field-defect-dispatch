"""Transcript repair and cue extraction.

`repair` fixes recogniser output against the domain glossary; `extract_cues` turns the repaired text
into the structured signals fusion consumes. Both are deterministic so they can be unit-tested.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from .assets import Asset, resolve_asset
from .schemas import Cues
from .taxonomy import GLOSSARY, DefectClass

_WORD = re.compile(r"[A-Za-z']+")
_SINGLE_TERMS = [g for g in GLOSSARY if " " not in g]

# Fuzzy matching only for longer words, otherwise common short words get mangled.
_MIN_LEN = 6
_MIN_SCORE = 84

_MENTION_PATTERNS: dict[DefectClass, list[str]] = {
    DefectClass.EXPOSED_REBAR: [r"\bexposed\b", r"\brebar\b", r"\breinforcing\b"],
    DefectClass.REBAR_CORROSION: [r"\bcorro\w*", r"\brust\w*"],
    DefectClass.SPALLING: [r"\bspall\w*", r"\bdelaminat\w*", r"\bchunk\w*"],
    DefectClass.STRUCTURAL_CRACK: [r"\bstructural crack", r"\bwide crack", r"\bshear crack", r"\bmajor crack"],
    DefectClass.HAIRLINE_CRACK: [r"\bhairline\b", r"\bsurface crack", r"\bminor crack"],
    DefectClass.WATER_LEAK: [r"\bleak\w*", r"\bseep\w*", r"\bdripping\b", r"\bwater ingress\b"],
    DefectClass.EFFLORESCENCE: [r"\befflorescen\w*", r"\bwhite (?:deposit|staining|powder)\b"],
    DefectClass.NO_DEFECT: [r"\bno (?:defect|damage|issue)s?\b", r"\ball clear\b", r"\blooks fine\b"],
}

_SEVERE = {
    "severe": 0.9, "critical": 1.0, "major": 0.8, "extensive": 0.8, "heavy": 0.7, "deep": 0.6,
    "large": 0.5, "serious": 0.8, "significant": 0.6, "dangerous": 1.0, "unsafe": 1.0,
    "collapsing": 1.0, "wide": 0.5, "bad": 0.5, "advanced": 0.6,
}
_MILD = {
    "minor": -0.7, "small": -0.5, "slight": -0.6, "light": -0.4, "superficial": -0.7,
    "cosmetic": -0.8, "tiny": -0.6, "negligible": -0.9, "faint": -0.5,
}
_URGENT = re.compile(r"\b(immediately|urgent\w*|right away|asap|evacuate|danger\w*)\b")
_NEGATORS = {"not", "no", "isn't", "isnt", "never", "without"}


def repair(text: str) -> str:
    """Join split glossary words ("re bar") and fuzzy-correct near-miss domain terms."""
    out = text
    for term in _SINGLE_TERMS:
        if len(term) >= 5:
            pat = r"\b" + r"\s?".join(re.escape(c) for c in term) + r"\b"
            out = re.sub(pat, term, out, flags=re.IGNORECASE)

    def fix(m: re.Match[str]) -> str:
        w = m.group(0)
        lw = w.lower()
        if len(lw) < _MIN_LEN or lw in _SINGLE_TERMS:
            return w
        best = process.extractOne(lw, _SINGLE_TERMS, scorer=fuzz.ratio, score_cutoff=_MIN_SCORE)
        return best[0] if best else w

    return _WORD.sub(fix, out)


def _severity(text: str) -> float:
    toks = _WORD.findall(text.lower())
    scores: list[float] = []
    for i, t in enumerate(toks):
        s = _SEVERE.get(t, _MILD.get(t))
        if s is None:
            continue
        if any(p in _NEGATORS for p in toks[max(0, i - 2):i]):
            s = -s * 0.5  # "not severe" is mild, not trivial
        scores.append(s)
    if not scores:
        return 0.0
    # the strongest cue wins: "small but severe corrosion" is still a severe finding
    return max(scores, key=abs)


def extract_cues(text: str, registry: dict[str, Asset]) -> Cues:
    fixed = repair(text)
    low = fixed.lower()
    mentions = [c for c, pats in _MENTION_PATTERNS.items() if any(re.search(p, low) for p in pats)]
    # a specific mention wins over the generic "exposed"/"rebar" words when corrosion is named
    if DefectClass.REBAR_CORROSION in mentions and DefectClass.EXPOSED_REBAR in mentions:
        if not re.search(r"\bexposed\b", low):
            mentions.remove(DefectClass.EXPOSED_REBAR)
    asset = resolve_asset(fixed, registry)
    return Cues(
        mentions=mentions,
        severity=max(-1.0, min(1.0, _severity(low))),
        urgent=bool(_URGENT.search(low)),
        asset_id=asset.asset_id if asset else None,
    )
