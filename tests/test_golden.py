"""The golden set is checked against the law itself: every gold citation must exist in the corpus and
every required fact must appear verbatim (by regex) in at least one gold passage. If the regulation
changes, these tests fail before a single model call is made."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = [json.loads(line) for line in (ROOT / "evals" / "golden.jsonl").read_text().splitlines() if line.strip()]
CHUNKS = [json.loads(line) for line in (ROOT / "data" / "processed" / "chunks.jsonl").read_text().splitlines()]


def test_golden_schema():
    ids = [g["id"] for g in GOLDEN]
    assert len(ids) == len(set(ids)) == 38
    for g in GOLDEN:
        assert g["expected_status"] in {"answered", "not_in_corpus", "refused"}
        assert bool(g["gold"]) == (g["expected_status"] == "answered")


@pytest.mark.parametrize("item", [g for g in GOLDEN if g["expected_status"] == "answered"], ids=lambda g: g["id"])
def test_gold_citations_exist_and_contain_required_facts(item):
    gold_chunks = [c for c in CHUNKS if any(c["citation"].startswith(p) for p in item["gold"])]
    assert gold_chunks, f"{item['id']}: no corpus chunk matches {item['gold']}"
    text = " ".join(f"{c['context']} {c['text']}" for c in gold_chunks)
    for fact in item["must_include"]:
        assert re.search(fact, text, re.IGNORECASE), f"{item['id']}: required fact {fact!r} not in gold passages"
