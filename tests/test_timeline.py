"""The deadline calculator is checked against the law: every rule quote must appear verbatim in its paragraph."""

import json
from datetime import date

import pytest

from reglens.config import ROOT
from reglens.timeline import RULES, add_business_days, credit_card_billing_error, fed_holidays, unauthorized_eft
from reglens.verify import normalize

CHUNKS = [json.loads(x) for x in (ROOT / "data/processed/chunks.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("citation", sorted(RULES))
def test_every_rule_quote_is_verbatim_in_the_cited_paragraph(citation):
    text = " ".join(f"{c['context']} {c['text']}" for c in CHUNKS if c["citation"] == citation)
    assert text, f"{citation} not in corpus"
    assert normalize(RULES[citation]) in normalize(text)


def test_federal_reserve_holidays_2026():
    h = fed_holidays(2026)
    assert date(2026, 11, 26) in h and date(2026, 10, 12) in h  # Thanksgiving, Columbus Day
    assert date(2026, 7, 3) not in h  # July 4 2026 is a Saturday: the Fed does not move it to Friday


def test_business_days_skip_weekends_and_holidays():
    assert add_business_days(date(2026, 11, 25), 1) == date(2026, 11, 27)  # skips Thanksgiving
    assert add_business_days(date(2026, 10, 23), 10) == date(2026, 11, 6)  # Fri + two full weeks
    assert add_business_days(date(2026, 11, 6), 3) == date(2026, 11, 12)  # skips Veterans Day (Wed Nov 11)


def test_stolen_card_reported_next_morning_is_capped_at_50():
    t = unauthorized_eft(
        notified=date(2026, 10, 21), learned_of_loss=date(2026, 10, 20), transfers=[(date(2026, 10, 20), 400.0)]
    )
    top = t.items[0]
    assert top.citation == "12 CFR 1005.6(b)(1)" and top.amount == 50.0 and top.due == "2026-10-22"


def test_late_notice_moves_to_500_tier_with_the_establish_condition():
    t = unauthorized_eft(
        notified=date(2026, 11, 2),
        learned_of_loss=date(2026, 10, 20),
        transfers=[(date(2026, 10, 20), 40.0), (date(2026, 10, 28), 900.0)],
    )
    top = t.items[0]
    assert top.citation == "12 CFR 1005.6(b)(2)" and top.amount == 500.0 and "establishes" in top.note


def test_error_resolution_deadlines():
    t = unauthorized_eft(
        notified=date(2026, 10, 21), learned_of_loss=date(2026, 10, 20), transfers=[(date(2026, 10, 20), 400.0)]
    )
    due = {i.citation: i.due for i in t.items if i.due}
    assert due["12 CFR 1005.11(c)(1)"] == "2026-11-04"  # 10 business days after Oct 21
    assert due["12 CFR 1005.11(c)(2)"] == "2026-12-05"  # 45 calendar days


def test_credit_card_billing_error_windows():
    t = credit_card_billing_error(
        notice_received=date(2026, 10, 1), statement_sent=date(2026, 9, 15), unauthorized_use=True
    )
    due = {i.citation: (i.due, i.amount) for i in t.items}
    assert due["12 CFR 1026.13(c)(1)"][0] == "2026-10-31" and due["12 CFR 1026.13(c)(2)"][0] == "2026-12-30"
    assert due["12 CFR 1026.12(b)(1)(ii)"][1] == 50.0 and not t.warnings


def test_new_account_and_point_of_sale_extensions():
    from reglens.timeline import eft_error_resolution

    t = eft_error_resolution(date(2026, 10, 21), new_account=True, oral_notice=False)
    due = {i.citation: i.due for i in t.items if i.due}
    assert due["12 CFR 1005.11(c)(3)(i)"] == add_business_days(date(2026, 10, 21), 20).isoformat()
    assert due["12 CFR 1005.11(c)(3)(ii)"] == "2027-01-19"  # 90 calendar days
