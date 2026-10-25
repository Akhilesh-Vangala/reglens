import pytest

from reglens.generation.generator import ScriptedGenerator
from reglens.graph import build_pipeline
from reglens.guard import BudgetExceeded, SpendLedger, redact
from reglens.retrieval import Hit
from tests.test_verify_and_graph import CORPUS, A, ans, claim


@pytest.mark.parametrize(
    "text,kind",
    [
        ("my SSN is 123-45-6789", "ssn"),
        ("card 4111 1111 1111 1111 was stolen", "card_number"),
        ("email me at jane.doe@example.com", "email"),
        ("call (703) 555-0142 please", "phone"),
        ("account number 00123456789 shows a charge", "account_number"),
    ],
)
def test_redact_masks_each_identifier(text, kind):
    r = redact(text)
    assert r.counts == {kind: 1} and f"[{kind.upper()} REDACTED]" in r.text


def test_redact_keeps_surrounding_text_intact():
    assert redact("card 4111 1111 1111 1111 stolen").text == "card [CARD_NUMBER REDACTED] stolen"


def test_redact_leaves_regulatory_numbers_alone():
    q = "Under 12 CFR 1005.6(b)(1), is liability $50 if reported within 2 business days or 60 days?"
    assert redact(q).text == q and not redact(q).redacted


def test_ledger_refuses_over_token_cap_and_over_dollar_cap(tmp_path):
    led = SpendLedger(tmp_path / "spend.json", usd_cap=0.01, max_tokens_per_request=1000)
    with pytest.raises(BudgetExceeded):
        led.admit(5000, None)
    led.record(1000, 100, 0.0095)
    with pytest.raises(BudgetExceeded):
        led.admit(500, 0.001)
    assert SpendLedger(tmp_path / "spend.json", None).state["calls"] == 1  # persisted


class Fixed:
    name = "fixed"

    def __init__(self):
        self.queries = []

    def search(self, q, k=8):
        self.queries.append(q)
        return [Hit(A, 1.0, {})]


def test_pipeline_redacts_before_retrieval_and_model():
    good = claim("a", "determine whether an error occurred within 10 business days")
    ret, gen = Fixed(), ScriptedGenerator([ans([good])])
    out = build_pipeline(ret, gen, CORPUS).invoke({"question": "card 4111111111111111 charged twice, SSN 123-45-6789"})
    assert "4111" not in ret.queries[0] and "4111" not in gen.prompts[0] and "123-45" not in gen.prompts[0]
    assert out["redactions"] == {"ssn": 1, "card_number": 1}


def test_pipeline_budget_refusal_escalates_without_calling_model(tmp_path):
    gen = ScriptedGenerator([])
    led = SpendLedger(tmp_path / "s.json", usd_cap=None, max_tokens_per_request=10)
    out = build_pipeline(Fixed(), gen, CORPUS, ledger=led).invoke({"question": "how long to investigate?"})
    assert gen.prompts == [] and out["final"]["needs_human_review"] and out["final"]["status"] == "not_in_corpus"
    assert any(e.startswith("budget_exceeded") for e in out["events"])


def test_anthropic_settings_and_client_construct_without_network(monkeypatch):
    from reglens.config import ModelSettings
    from reglens.generation.generator import ChatGenerator

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-5-5")
    s = ModelSettings.from_env()
    assert (s.provider, s.chat_deployment, s.embedding_deployment) == ("anthropic", "claude-haiku-5-5", "")
    assert ChatGenerator(s).model == "anthropic:claude-haiku-5-5"


def test_budgeted_generator_stops_before_the_call_that_would_cross_the_cap(tmp_path):
    from reglens.generation.generator import Budgeted
    from reglens.generation.schema import Answer

    good = ans([claim("a", "determine whether an error occurred within 10 business days")])
    inner = ScriptedGenerator([good, good, good])
    led = SpendLedger(tmp_path / "s.json", usd_cap=0.002)
    # worst case per call: ~33 input tokens + 1,000 output tokens at $0.10/$0.50 per 1M -> ~$0.0005
    gen = Budgeted(inner, led, 0.10, 0.50, max_out=1000)
    gen.generate("x" * 100, Answer)
    led.state["usd"] = 0.0019  # simulate earlier spend just under the cap
    with pytest.raises(BudgetExceeded):
        gen.generate("x" * 100, Answer)
    assert len(inner.prompts) == 1  # the refused call never reached the model
