import pytest

from dispatch.assets import default_registry, resolve_asset
from dispatch.taxonomy import DefectClass
from dispatch.text import extract_cues, repair

REG = default_registry()


@pytest.mark.parametrize("text,expected", [
    ("pump housing B4", "B4"),
    ("pump housing B 4", "B4"),
    ("pump housing b-4", "B4"),
    ("pump housing be four", "B4"),
    ("pump housing b for", "B4"),
    ("column see two", "C2"),
    ("slab S3 looks bad", "S3"),
    ("wall double u one", None),  # unsupported spelling must not guess
    ("went to the shop for four hours", None),  # homophones outside a letter context are ignored
    ("pump housing B99", None),  # not in the registry
])
def test_resolve_asset(text, expected):
    got = resolve_asset(text, REG)
    assert (got.asset_id if got else None) == expected


def test_repair_joins_split_words_and_fixes_near_misses():
    assert "rebar" in repair("found exposed re bar")
    assert "corrosion" in repair("severe corosion near b4")
    assert "spalling" in repair("concrete spaling on the slab")


def test_repair_leaves_ordinary_words_alone():
    text = "the inspector walked around the building"
    assert repair(text) == text


def test_cues_for_the_headline_example():
    c = extract_cues("Found severe rebar corrosion near pump housing B4", REG)
    assert DefectClass.REBAR_CORROSION in c.mentions
    assert DefectClass.EXPOSED_REBAR not in c.mentions  # 'rebar' alone is not 'exposed'
    assert c.severity >= 0.8
    assert c.asset_id == "B4"


def test_cues_negation_softens_severity():
    assert extract_cues("this is not severe", REG).severity < 0
    assert extract_cues("this is severe", REG).severity > 0.5


def test_cues_urgent_and_none():
    c = extract_cues("exposed rebar on column C1 needs attention immediately", REG)
    assert c.urgent and DefectClass.EXPOSED_REBAR in c.mentions
    assert DefectClass.NO_DEFECT in extract_cues("all clear on wall W2", REG).mentions


def test_strongest_cue_wins():
    assert extract_cues("small but severe corrosion", REG).severity > 0.5
