"""Corpus tests run on the pinned eCFR files in data/raw (2026-10-02)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from reglens.corpus.chunk import chunk_passages
from reglens.corpus.parse import parse_part

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
MANIFEST = json.loads((RAW / "manifest.json").read_text())


@pytest.fixture(scope="module")
def passages():
    out, warnings = [], []
    for part, meta in MANIFEST["parts"].items():
        secs = set(meta["sections"]) if meta["sections"] else None
        out += parse_part((RAW / meta["file"]).read_bytes(), part, secs, warnings)
    return out, warnings


def by_citation(passages):
    return {p.citation: p for p in passages[0]}


def test_citations_are_unique(passages):
    dupes = [c for c, n in Counter(p.citation for p in passages[0]).items() if n > 1]
    assert dupes == []


@pytest.mark.parametrize(
    "citation,starts",
    [
        ("12 CFR 1005.11(c)(1)", "(c) Time limits and extent of investigation"),
        ("12 CFR 1005.11(c)(2)(i)(A)", "(A) The institution requires but does not receive written confirmation"),
        ("12 CFR 1005.6(b)(1)", None),
        ("12 CFR 1005.18(b)(2)(ix)(D)(1)", "(D) Timing of initial assessment"),  # six-level nesting
        ("12 CFR 1005.2(b)(3)(ii)(D)(2)", "(2) A store gift card"),  # adjacent labels "(D)(1)"
        ("12 CFR 1026.13(a)(1)", None),
    ],
)
def test_rule_paths(passages, citation, starts):
    p = by_citation(passages).get(citation)
    assert p is not None, citation
    if starts:
        assert p.text.startswith(starts)


def test_interpretations_and_missing_heading_repair(passages):
    idx = by_citation(passages)
    assert idx["Supplement I to Part 1005, comment 11(c)-2"].text.startswith("2. Written confirmation of oral notice")
    # eCFR omits the 17(b)(3) heading; the block is re-labelled from its own text and the repair is logged
    assert idx["Supplement I to Part 1005, comment 17(b)(3)-2"].text.startswith("2. Limited-feature bank accounts")
    assert any("assigned 17(b)(3)" in w for w in passages[1])
    assert "Supplement I to Part 1026, comment 13(a)(3)-1" in idx


def test_scope_filter_keeps_only_reg_z_12_13(passages):
    assert {p.section for p in passages[0] if p.part == "1026"} == {"1026.12", "1026.13"}


def test_chunks_carry_parent_context_and_respect_size(passages):
    chunks = {c.citation: c for c in chunk_passages(passages[0])}
    c = chunks["12 CFR 1005.11(a)(1)(i)"]
    assert c.text == "(i) An unauthorized electronic fund transfer;"
    assert "The term “error” means" in c.context
    assert max(len(x.text) for x in chunks.values()) <= 1600
