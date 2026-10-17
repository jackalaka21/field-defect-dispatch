"""Asset registry and resolution of spoken asset IDs ("pump housing be four" -> B4)."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    asset_id: str
    name: str
    structural_criticality: float  # 0 (non-structural) .. 1 (load-bearing)


def default_registry() -> dict[str, Asset]:
    reg: dict[str, Asset] = {}
    for n in range(1, 13):
        reg[f"B{n}"] = Asset(f"B{n}", f"Pump housing B{n}", 0.45)
    for n in range(1, 9):
        reg[f"C{n}"] = Asset(f"C{n}", f"Column C{n}", 0.95)
    for n in range(1, 7):
        reg[f"S{n}"] = Asset(f"S{n}", f"Slab S{n}", 0.75)
    for n in range(1, 7):
        reg[f"W{n}"] = Asset(f"W{n}", f"Wall W{n}", 0.35)
    return reg


# Letter names and number words as a recogniser tends to write them.
_LETTERS = {
    "b": "B", "be": "B", "bee": "B",
    "c": "C", "see": "C", "sea": "C",
    "s": "S", "es": "S", "ess": "S",
    "w": "W", "doubleu": "W",
}
_NUMBERS = {
    "zero": 0, "one": 1, "won": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4, "for": 4,
    "fore": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "ate": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_TOKEN = re.compile(r"[a-z]+|\d+", re.IGNORECASE)


def _num(tok: str) -> int | None:
    if tok.isdigit():
        return int(tok)
    return _NUMBERS.get(tok)


def resolve_asset(text: str, registry: dict[str, Asset]) -> Asset | None:
    """Return the registered asset mentioned in `text`, or None.

    Homophones ("for" -> 4) are only accepted directly after a letter name, so ordinary words such
    as "for" or "to" elsewhere in a sentence are never turned into asset numbers.
    """
    toks = [t.lower() for t in _TOKEN.findall(text.replace("-", " "))]
    # glued forms such as "b4" / "c12"
    for i, t in enumerate(toks):
        m = re.fullmatch(r"([bcsw])(\d+)", t)
        if m and f"{m.group(1).upper()}{int(m.group(2))}" in registry:
            return registry[f"{m.group(1).upper()}{int(m.group(2))}"]
    for i in range(len(toks) - 1):
        letter = _LETTERS.get(toks[i])
        if letter is None:
            continue
        n = _num(toks[i + 1])
        if n is not None and f"{letter}{n}" in registry:
            return registry[f"{letter}{n}"]
    return None
