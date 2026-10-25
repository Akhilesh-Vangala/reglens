"""Copilot agent with a scripted tool-calling model: the graph, tools and deterministic finalize step are real."""

from datetime import date

from langchain_core.messages import AIMessage

from reglens.config import ROOT
from reglens.copilot import build_copilot
from reglens.kg import build_graph
from reglens.retrieval import Hit, load_chunks

CHUNKS = load_chunks(ROOT / "data" / "processed" / "chunks.jsonl")
CORPUS = {c.id: c for c in CHUNKS}
BY_CIT = {c.citation: c for c in CHUNKS}
GRAPH = build_graph(CHUNKS)
LIAB = BY_CIT["12 CFR 1005.6(b)(1)"]
INVESTIGATE = BY_CIT["12 CFR 1005.11(c)(1)"]


class Fixed:
    name = "fixed"

    def search(self, q, k=8):
        return [Hit(LIAB, 1.0, {}), Hit(INVESTIGATE, 0.9, {})]


class ScriptedChat:
    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        calls = self.turns.pop(0)
        return AIMessage(
            "",
            tool_calls=[{"name": n, "args": a, "id": f"c{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)],
            usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
        )


CASE = "Member says: my debit card was stolen on 2026-10-20 and I called you the next morning. There's a $400 charge."
DEADLINES = (
    "compute_deadlines",
    {
        "case_type": "unauthorized_eft",
        "notified": "2026-10-21",
        "learned_of_loss": "2026-10-20",
        "transfers": [{"date": "2026-10-20", "amount": 400}],
    },
)


def submit(reply, claims, case_type="unauthorized_eft", handoff=None):
    return (
        "submit_answer",
        {
            "case_type": case_type,
            "reply_to_member": reply,
            "claims": claims,
            "needs_human_review": True,
            "review_reason": "stolen card",
            "handoff": handoff,
        },
    )


def run(turns):
    llm = ScriptedChat(turns)
    app = build_copilot(Fixed(), GRAPH, CORPUS, llm, today=date(2026, 10, 25))
    return app.invoke({"case": CASE}), llm


GOOD = {
    "statement": "Liability is capped at $50",
    "chunk_id": LIAB.id,
    "quote": "the consumer's liability shall not exceed the lesser of $50 or the amount of unauthorized transfers",
}


def test_full_case_verified_claims_timeline_and_policy_handoff():
    out, llm = run(
        [
            [("search_regulations", {"query": "stolen debit card liability", "part": "1005"})],
            [DEADLINES, ("related_paragraphs", {"chunk_id": LIAB.id})],
            [
                submit(
                    "You reported within two business days, so the most you could owe is $50. We will "
                    "decide on your claim by November 4, 2026.",
                    [GOOD],
                )
            ],
        ]
    )
    r = out["result"]
    assert r["claims"][0]["citation"] == "12 CFR 1005.6(b)(1)" and not r["draft_withheld"]
    assert r["timelines"][0]["items"][0]["amount"] == 50.0
    assert r["handoff"]["queue"] == "disputes" and "handoff_added_by_policy" in r["events"]
    assert [s.get("tool") for s in out["steps"] if s["node"] == "tool"] == [
        "search_regulations",
        "compute_deadlines",
        "related_paragraphs",
        "submit_answer",
    ]


def test_fabricated_quote_is_dropped_and_reply_withheld():
    fake = {**GOOD, "quote": "the credit union must refund every disputed charge within one day"}
    out, _ = run([[("search_regulations", {"query": "x"})], [submit("We refund in one day.", [fake])]])
    r = out["result"]
    assert r["claims"] == [] and r["draft_withheld"] and r["reply_to_member"] == ""
    assert r["checks"][0]["reason"] == "quote_not_in_source" and r["needs_human_review"]


def test_quote_from_a_passage_no_tool_returned_is_rejected():
    unseen = BY_CIT["12 CFR 1026.13(c)(1)"]
    claim = {"statement": "s", "chunk_id": unseen.id, "quote": unseen.text[4:90]}
    out, _ = run([[("search_regulations", {"query": "x"})], [submit("text", [claim, GOOD])]])
    assert [c["reason"] for c in out["result"]["checks"]] == ["not_retrieved", "ok"]


def test_date_not_from_the_calculator_withholds_the_reply():
    out, _ = run(
        [[("search_regulations", {"query": "x"})], [DEADLINES], [submit("We will decide by November 1, 2026.", [GOOD])]]
    )
    r = out["result"]
    assert r["draft_withheld"] and any(e.startswith("dates_not_from_calculator") for e in r["events"])


def test_pii_is_masked_before_the_model_sees_the_case():
    llm = ScriptedChat([[submit("ok", [], case_type="out_of_scope")]])
    app = build_copilot(Fixed(), GRAPH, CORPUS, llm, today=date(2026, 10, 25))
    out = app.invoke({"case": "card 4111 1111 1111 1111 stolen, SSN 123-45-6789"})
    assert "4111" not in llm.seen[0][1].content and out["redactions"] == {"ssn": 1, "card_number": 1}


def test_agent_loop_is_bounded():
    out, llm = run([[("search_regulations", {"query": "x"})]] * 20)
    assert sum(s["node"] == "agent" for s in out["steps"]) == 8 and "no_submission" in out["result"]["events"]


def test_tool_errors_return_to_the_model_instead_of_crashing():
    out, _ = run([[("related_paragraphs", {"chunk_id": "nope"})], [submit("ok", [], case_type="out_of_scope")]])
    assert out["steps"][1]["result"].startswith("error: unknown chunk_id")


def test_dollar_amount_not_backed_by_quote_calculator_or_member_withholds_the_reply():
    out, _ = run(
        [[("search_regulations", {"query": "x"})], [DEADLINES], [submit("You owe at most $75 of the $400.", [GOOD])]]
    )
    r = out["result"]
    assert r["draft_withheld"] and any(e == "amounts_not_supported: [75.0]" for e in r["events"])


def test_amounts_parser():
    from reglens.copilot import amounts_in

    assert amounts_in("$50, $1,250.75 and $ 500") == {50.0, 1250.75, 500.0}


def test_part_filter_is_pushed_down_to_retrievers_that_support_it():
    class Filtering(Fixed):
        supports_filters = True

        def __init__(self):
            self.calls = []

        def search(self, q, k=8, part=None):
            self.calls.append(part)
            return super().search(q, k)

    r = Filtering()
    llm = ScriptedChat(
        [[("search_regulations", {"query": "x", "part": "1005"})], [submit("ok", [], case_type="out_of_scope")]]
    )
    build_copilot(r, GRAPH, CORPUS, llm, today=date(2026, 10, 25)).invoke({"case": "q"})
    assert r.calls == ["1005"]


def test_tool_specs_use_real_names_and_strict_schemas():
    from reglens.copilot import tool_specs

    specs = {t["function"]["name"]: t["function"]["parameters"] for t in tool_specs()}
    assert set(specs) == {"search_regulations", "related_paragraphs", "compute_deadlines", "submit_answer"}
    d = specs["compute_deadlines"]
    assert d["additionalProperties"] is False and set(d["required"]) == set(d["properties"])
    claim_schema = specs["submit_answer"]["properties"]["claims"]["items"]
    assert "$ref" not in str(specs) and claim_schema["additionalProperties"] is False


def test_stringified_nested_arguments_are_decoded():
    out, _ = run(
        [
            [("compute_deadlines", {**DEADLINES[1], "transfers": '[{"date": "2026-10-20", "amount": 400}]'})],
            [submit("ok", [], case_type="out_of_scope")],
        ]
    )
    assert out["timelines"][0]["items"][0]["amount"] == 50.0


def test_member_dates_without_year_and_sums_of_member_amounts_are_supported():
    from reglens.copilot import dates_in, subset_sums

    assert "2026-10-23" in dates_in("I called on October 23.", default_year=2026)
    assert dates_in("by November 4, 2026", default_year=2025) == {"2026-11-04"}  # an explicit year wins
    assert 940.0 in subset_sums({40.0, 900.0}) and 40.0 not in subset_sums({40.0, 900.0})


def test_total_of_member_amounts_does_not_withhold_but_an_invented_total_does():
    case = "Card lost 2026-10-01, reported 2026-10-23. Charges: $40 on 2026-10-02 and $900 on 2026-10-15."
    for total, withheld in ((940, False), (975, True)):
        llm = ScriptedChat(
            [
                [("search_regulations", {"query": "x"})],
                [submit(f"Together that is ${total}; your liability is at most $50.", [GOOD])],
            ]
        )
        out = build_copilot(Fixed(), GRAPH, CORPUS, llm, today=date(2026, 10, 25)).invoke({"case": case})
        assert out["result"]["draft_withheld"] is withheld, total


def test_call_cost_prices_cache_reads_writes_and_is_conservative_without_a_read_price():
    from reglens.copilot import call_cost

    # 10,000 input of which 8,000 cache reads and 1,000 cache writes; 500 output; Haiku 5.5 list prices
    with_read = call_cost(10_000, 500, 8_000, 1_000, (0.10, 0.50, 0.01))
    without = call_cost(10_000, 500, 8_000, 1_000, (0.10, 0.50))
    assert round(with_read * 1e6, 3) == round(1_000 * 0.10 + 8_000 * 0.01 + 1_000 * 0.125 + 500 * 0.50, 3)
    assert without > with_read  # unset read price never under-counts


def test_not_covered_question_releases_a_redirect_without_claims():
    out, _ = run(
        [
            [
                submit(
                    "I can't answer rate questions from these rules; a loan officer can help.",
                    [],
                    case_type="not_covered",
                )
            ]
        ]
    )
    r = out["result"]
    assert r["case_type"] == "not_covered" and not r["draft_withheld"] and r["claims"] == []


def test_prompt_cache_breakpoint_is_requested_only_from_the_anthropic_client():
    class Recording(ScriptedChat):
        _llm_type = "anthropic-chat"

        def __init__(self, turns):
            super().__init__(turns)
            self.kwargs = []

        def invoke(self, messages, **kwargs):
            self.kwargs.append(kwargs)
            return super().invoke(messages)

    llm = Recording([[submit("ok", [], case_type="out_of_scope")]])
    build_copilot(Fixed(), GRAPH, CORPUS, llm, today=date(2026, 10, 25)).invoke({"case": "q"})
    assert llm.kwargs == [{"cache_control": {"type": "ephemeral"}}]
