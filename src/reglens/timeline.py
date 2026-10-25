"""Deterministic deadline and liability calculator for member cases. The model never does date arithmetic.

Every rule applied carries the citation and a verbatim quote of the regulation text it implements;
tests/test_timeline.py fails if any quote is not found in the pinned corpus, so the calculator is checked against
the law the same way the golden set is.

Business days: 12 CFR 1005.2(d) defines a business day by when the member's institution is open. We assume
Monday-Friday excluding Federal Reserve holidays (a Sunday holiday is observed Monday; a Saturday holiday is not
moved). Every result states this assumption. Outcomes that depend on facts the institution must establish are
reported as maximums, never as decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from functools import lru_cache

ASSUMPTION = (
    "Business days assumed Monday-Friday excluding Federal Reserve holidays (12 CFR 1005.2(d) defines them by when "
    "the institution is open). Amounts are regulatory maximums, not a decision on this member's case."
)

# citation -> verbatim text the rule is implemented from (checked against the corpus by tests)
RULES = {
    "12 CFR 1005.6(b)(1)": "If the consumer notifies the financial institution within two business days after learning "
    "of the loss or theft of the access device, the consumer's liability shall not exceed the lesser of $50",
    "12 CFR 1005.6(b)(2)": "If the consumer fails to notify the financial institution within two business days after "
    "learning of the loss or theft of the access device, the consumer's liability shall not exceed the lesser of $500",
    "12 CFR 1005.6(b)(3)": "A consumer must report an unauthorized electronic fund transfer that appears on a periodic "
    "statement within 60 days of the financial institution's transmittal of the statement",
    "12 CFR 1005.6(b)(4)": "If the consumer's delay in notifying the financial institution was due to extenuating "
    "circumstances, the institution shall extend the times specified above to a reasonable period.",
    "12 CFR 1005.11(b)(1)(i)": "Is received by the institution no later than 60 days after the institution sends the "
    "periodic statement",
    "12 CFR 1005.11(b)(2)": "A financial institution may require the consumer to give written confirmation of an error "
    "within 10 business days of an oral notice.",
    "12 CFR 1005.11(c)(1)": "shall determine whether an error occurred within 10 business days of receiving a notice "
    "of error. The institution shall report the results to the consumer within three business days after completing "
    "its investigation. The institution shall correct the error within one business day after determining that an "
    "error occurred.",
    "12 CFR 1005.11(c)(2)": "the institution may take up to 45 days from receipt of a notice of error to investigate",
    "12 CFR 1005.11(c)(2)(i)": "Provisionally credits the consumer's account in the amount of the alleged error "
    "(including interest where applicable) within 10 business days of receiving the error notice.",
    "12 CFR 1005.11(c)(2)(ii)": "Informs the consumer, within two business days after the provisional crediting",
    "12 CFR 1005.11(c)(3)(i)": "The applicable time is 20 business days in place of 10 business days",
    "12 CFR 1005.11(c)(3)(ii)": "The applicable time is 90 days in place of 45 days",
    "12 CFR 1026.12(b)(1)(ii)": "The liability of a cardholder for unauthorized use of a credit card shall not exceed "
    "the lesser of $50",
    "12 CFR 1026.13(b)(1)": "no later than 60 days after the creditor transmitted the first periodic statement that "
    "reflects the alleged billing error",
    "12 CFR 1026.13(c)(1)": "The creditor shall mail or deliver written acknowledgment to the consumer within 30 days "
    "of receiving a billing error notice",
    "12 CFR 1026.13(c)(2)": "within 2 complete billing cycles (but in no event later than 90 days) after receiving a "
    "billing error notice.",
}


@dataclass
class Item:
    label: str
    citation: str
    due: str | None = None  # ISO date
    amount: float | None = None
    note: str = ""

    @property
    def quote(self) -> str:
        return RULES[self.citation]


@dataclass
class Timeline:
    case_type: str
    items: list[Item] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assumption: str = ASSUMPTION

    def to_dict(self) -> dict:
        return {
            "case_type": self.case_type,
            "assumption": self.assumption,
            "warnings": self.warnings,
            "items": [{**asdict(i), "quote": i.quote} for i in self.items],
        }


# ---- business-day calendar -------------------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


@lru_cache(maxsize=64)
def fed_holidays(year: int) -> frozenset[date]:
    fixed = [date(year, 1, 1), date(year, 6, 19), date(year, 7, 4), date(year, 11, 11), date(year, 12, 25)]
    observed = {d + timedelta(days=1) if d.weekday() == 6 else d for d in fixed}  # Sunday -> Monday; Saturday stays
    observed |= {
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _last_weekday(year, 5, 0),  # Memorial Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 10, 0, 2),  # Columbus Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
    }
    return frozenset(observed)


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in fed_holidays(d.year)


def add_business_days(start: date, n: int) -> date:
    """The nth business day after `start` (start itself is not counted)."""
    d = start
    while n:
        d += timedelta(days=1)
        n -= is_business_day(d)
    return d


# ---- calculators ---------------------------------------------------------------------------------------------


def unauthorized_eft(
    notified: date,
    transfers: list[tuple[date, float]],
    learned_of_loss: date | None = None,
    statement_sent: date | None = None,
) -> Timeline:
    """Regulation E consumer liability for unauthorized electronic fund transfers (12 CFR 1005.6(b)).
    `learned_of_loss` is when the member learned the card/access device was lost or stolen (None if no device
    was lost, e.g. a hacked account); `statement_sent` is when the first statement showing the transfer was sent."""
    t = Timeline("unauthorized_electronic_fund_transfer")
    if not transfers:
        t.warnings.append("No unauthorized transfers were given, so the liability caps below are computed on $0.")
    before_notice = sum(a for d, a in transfers if d < notified)
    if learned_of_loss:
        two_day = add_business_days(learned_of_loss, 2)
        if notified <= two_day:
            t.items.append(
                Item(
                    "Maximum member liability (notified within two business days)",
                    "12 CFR 1005.6(b)(1)",
                    due=two_day.isoformat(),
                    amount=min(50.0, before_notice),
                )
            )
        else:
            within = sum(a for d, a in transfers if d <= two_day)
            after = sum(a for d, a in transfers if two_day < d < notified)
            t.items.append(
                Item(
                    "Maximum member liability (two-business-day notice deadline missed)",
                    "12 CFR 1005.6(b)(2)",
                    due=two_day.isoformat(),
                    amount=min(500.0, min(50.0, within) + after),
                    note="The amount above $50 applies only if the institution establishes those transfers would not "
                    "have occurred had notice been given within two business days.",
                )
            )
            t.items.append(Item("Deadlines extend for extenuating circumstances", "12 CFR 1005.6(b)(4)"))
    if statement_sent:
        sixty = statement_sent + timedelta(days=60)
        t.items.append(
            Item(
                "Report deadline for transfers on the periodic statement", "12 CFR 1005.6(b)(3)", due=sixty.isoformat()
            )
        )
        if notified > sixty:
            late = sum(a for d, a in transfers if sixty < d < notified)
            t.items.append(
                Item(
                    "Additional exposure for transfers after the 60-day window",
                    "12 CFR 1005.6(b)(3)",
                    amount=late,
                    note="Applies only to transfers the institution establishes would not have occurred with timely "
                    "notice; no $500 cap applies to these.",
                )
            )
    if not learned_of_loss and not statement_sent:
        t.warnings.append(
            "Need the date the member learned of the loss/theft, or the statement date, to apply the liability tiers."
        )
    t.items += eft_error_resolution(notified, statement_sent).items
    return t


def eft_error_resolution(
    notice_received: date,
    statement_sent: date | None = None,
    oral_notice: bool = True,
    new_account: bool = False,
    pos_or_foreign: bool = False,
) -> Timeline:
    """Regulation E error-resolution deadlines (12 CFR 1005.11(b)-(c)). `new_account`: the transfer occurred within
    30 days after the first deposit to the account; `pos_or_foreign`: point-of-sale debit card or not initiated
    within a state."""
    t = Timeline("electronic_fund_transfer_error")
    if statement_sent:
        last = statement_sent + timedelta(days=60)
        t.items.append(
            Item(
                "Last day the institution must receive the error notice",
                "12 CFR 1005.11(b)(1)(i)",
                due=last.isoformat(),
            )
        )
        if notice_received > last:
            t.warnings.append("Notice received after the 60-day window; the 1005.11 procedures may not apply.")
    if oral_notice:
        t.items.append(
            Item(
                "Written confirmation due (only if the institution requires it)",
                "12 CFR 1005.11(b)(2)",
                due=add_business_days(notice_received, 10).isoformat(),
            )
        )
    short = 20 if new_account else 10
    short_cite = "12 CFR 1005.11(c)(3)(i)" if new_account else "12 CFR 1005.11(c)(1)"
    t.items.append(
        Item(
            f"Investigation decision due ({short} business days), or provisional credit by this date",
            short_cite,
            due=add_business_days(notice_received, short).isoformat(),
        )
    )
    t.items.append(
        Item(
            "If extending: provisional credit to the member by",
            "12 CFR 1005.11(c)(2)(i)",
            due=add_business_days(notice_received, short).isoformat(),
            note="A $50 withholding is permitted in some unauthorized-transfer cases.",
        )
    )
    t.items.append(
        Item(
            "If extending: tell the member the amount and date of provisional credit",
            "12 CFR 1005.11(c)(2)(ii)",
            note="Within two business days after the provisional crediting.",
        )
    )
    long_days = 90 if (new_account or pos_or_foreign) else 45
    long_cite = "12 CFR 1005.11(c)(3)(ii)" if long_days == 90 else "12 CFR 1005.11(c)(2)"
    t.items.append(
        Item(
            f"Extended investigation must finish by ({long_days} calendar days)",
            long_cite,
            due=(notice_received + timedelta(days=long_days)).isoformat(),
        )
    )
    t.items.append(
        Item("After deciding: report results within 3 business days; correct an error within 1", "12 CFR 1005.11(c)(1)")
    )
    return t


def credit_card_billing_error(notice_received: date, statement_sent: date, unauthorized_use: bool = False) -> Timeline:
    """Regulation Z billing-error deadlines (12 CFR 1026.13) and unauthorized-use cap (12 CFR 1026.12(b))."""
    t = Timeline("credit_card_billing_error")
    last = statement_sent + timedelta(days=60)
    t.items.append(
        Item(
            "Last day the creditor must receive the written billing-error notice",
            "12 CFR 1026.13(b)(1)",
            due=last.isoformat(),
        )
    )
    if notice_received > last:
        t.warnings.append("Notice received after 60 days; 1026.13 procedures may not apply.")
    t.items.append(
        Item(
            "Written acknowledgment due", "12 CFR 1026.13(c)(1)", due=(notice_received + timedelta(days=30)).isoformat()
        )
    )
    t.items.append(
        Item(
            "Resolution due: two complete billing cycles, never later than",
            "12 CFR 1026.13(c)(2)",
            due=(notice_received + timedelta(days=90)).isoformat(),
        )
    )
    if unauthorized_use:
        t.items.append(
            Item("Maximum cardholder liability for unauthorized use", "12 CFR 1026.12(b)(1)(ii)", amount=50.0)
        )
    return t
