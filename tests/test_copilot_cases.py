"""The copilot scenario set is checked like the golden set: every expected calculator output must be what the
calculator produces from the case's reference arguments, so the expectations cannot drift from the law code."""

import pytest

from reglens.copilot import DeadlineArgs, _run_deadlines
from reglens.eval.run import load_copilot_cases, score_copilot_case

CASES = load_copilot_cases()


def test_case_set_shape():
    assert len(CASES) == 10 and len({c["id"] for c in CASES}) == 10
    assert {c["expected_case_type"] for c in CASES} >= {
        "unauthorized_eft",
        "eft_error",
        "credit_card_billing_error",
        "funds_availability",
        "out_of_scope",
    }


@pytest.mark.parametrize("case", [c for c in CASES if c["reference_args"]], ids=lambda c: c["id"])
def test_expected_items_follow_from_reference_args(case):
    items = _run_deadlines(DeadlineArgs(**case["reference_args"]))["items"]
    for e in case["expect_items"]:
        assert any(
            i["citation"] == e["citation"] and all(i.get(k) == v for k, v in e.items() if k != "citation")
            for i in items
        ), e


def test_scorer_credits_correct_extraction_and_flags_wrong_dates():
    case = next(c for c in CASES if c["id"] == "C01")
    good = {
        "case_type": "unauthorized_eft",
        "timelines": [_run_deadlines(DeadlineArgs(**case["reference_args"]))],
        "handoff": {"queue": "disputes"},
        "draft_withheld": False,
        "claims": [{}],
        "checks": [{"verified": True}],
    }
    s = score_copilot_case(case, good, {"claims": [{}]})
    assert s["case_type_correct"] and s["deadlines_correct"] and s["handoff_correct"] and s["reply_released"]
    wrong_args = {**case["reference_args"], "notified": "2026-10-28"}  # model misread the notice date
    bad = {**good, "timelines": [_run_deadlines(DeadlineArgs(**wrong_args))]}
    assert not score_copilot_case(case, bad, {"claims": [{}]})["deadlines_correct"]
