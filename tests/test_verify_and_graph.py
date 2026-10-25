import numpy as np

from reglens.corpus.chunk import Chunk
from reglens.generation.generator import ScriptedGenerator
from reglens.graph import build_pipeline
from reglens.retrieval import BM25, Dense, Hit, LocalHybrid, rrf
from reglens.verify import check_claims

A = Chunk(
    "a",
    "1005",
    "1005.11",
    "rule",
    "12 CFR 1005.11(c)(1)",
    "§ 1005.11",
    "",
    "A financial institution shall investigate promptly and determine whether an error occurred "
    "within 10 business days of receiving a notice of error.",
)
B = Chunk(
    "b",
    "1005",
    "1005.6",
    "rule",
    "12 CFR 1005.6(b)(1)",
    "§ 1005.6",
    "",
    "If the consumer notifies the financial institution within two business days, liability is limited to $50.",
)
CORPUS = {"a": A, "b": B}


def claim(cid, quote):
    return {"statement": "s", "chunk_id": cid, "quote": quote}


def test_verifier_accepts_exact_quote_with_normalized_whitespace_and_case():
    [c] = check_claims([claim("a", "determine whether an error   occurred WITHIN 10 business days")], {"a": A}, CORPUS)
    assert c.verified and c.reason == "ok"


def test_verifier_rejects_paraphrase_unseen_passage_fake_id_and_short_quote():
    checks = check_claims(
        [
            claim("a", "must finish the investigation in ten business days"),  # paraphrase
            claim("b", "liability is limited to $50 when reported in time"),  # real passage, not retrieved
            claim("zzz", "anything at all that is long enough to count"),  # invented id
            claim("a", "10 business days"),  # too short
        ],
        {"a": A},
        CORPUS,
    )
    assert [c.reason for c in checks] == ["quote_not_in_source", "not_retrieved", "unknown_chunk", "quote_too_short"]


def test_rrf_rewards_agreement_between_retrievers():
    h = lambda c: Hit(c, 1.0, {})
    fused = rrf([[h(A), h(B)], [h(A)]], k=2)
    assert [x.chunk.id for x in fused] == ["a", "b"] and fused[0].score > fused[1].score


class ToyEmbedder:
    name = "toy"

    def embed(self, texts):
        vocab = ["investigate", "error", "liability", "stolen", "days", "notice"]
        return np.array([[t.lower().count(w) + 0.01 for w in vocab] for t in texts], dtype=float)


def test_hybrid_retrieval_ranks_relevant_chunk_first():
    chunks = [A, B]
    hybrid = LocalHybrid(BM25(chunks), Dense(chunks, ToyEmbedder()))
    assert hybrid.search("how long to investigate an error", 2)[0].chunk.id == "a"


def _pipeline(answers, verify=True):
    class Fixed:
        name = "fixed"

        def search(self, q, k=8):
            return [Hit(A, 1.0, {})]

    gen = ScriptedGenerator(answers)
    return build_pipeline(Fixed(), gen, CORPUS, verify=verify), gen


def ans(claims, status="answered"):
    return {"status": status, "answer": "x", "claims": claims, "needs_human_review": False, "review_reason": ""}


def test_graph_retries_once_with_feedback_then_accepts_verified_answer():
    good = claim("a", "determine whether an error occurred within 10 business days")
    app, gen = _pipeline([ans([claim("a", "a made up paraphrase of the rule here")]), ans([good])])
    out = app.invoke({"question": "q"})
    assert len(gen.prompts) == 2 and "failed verification" in gen.prompts[1]
    assert out["final"]["status"] == "answered" and out["final"]["claims"] == [{**good, "citation": A.citation}]


def test_graph_strips_unverified_claims_and_downgrades_empty_answer():
    bad = ans([claim("b", "liability is limited to $50 when reported in time")])
    app, _ = _pipeline([bad, bad])
    out = app.invoke({"question": "q"})
    assert out["final"]["status"] == "not_in_corpus" and out["final"]["claims"] == []
    assert out["final"]["needs_human_review"] and "downgraded_to_not_in_corpus" in out["events"]


def test_ablation_without_verifier_passes_draft_through():
    bad = ans([claim("b", "liability is limited to $50 when reported in time")])
    app, gen = _pipeline([bad], verify=False)
    out = app.invoke({"question": "q"})
    assert len(gen.prompts) == 1 and out["final"]["claims"] == bad["claims"]


def test_quote_from_parent_lead_in_is_attributed_to_the_parent_paragraph():
    from reglens.verify import parent_citation

    child = Chunk(
        "c",
        "1005",
        "1005.6",
        "rule",
        "12 CFR 1005.6(b)(2)(ii)",
        "§ 1005.6",
        "the consumer's liability shall not exceed the lesser of $500 or the sum of:",
        "(ii) The amount of unauthorized transfers that occur after the close of two business days.",
    )
    corpus = {"c": child}
    [in_context] = check_claims(
        [claim("c", "liability shall not exceed the lesser of $500 or the sum of")], corpus, corpus
    )
    [in_text] = check_claims(
        [claim("c", "unauthorized transfers that occur after the close of two business")], corpus, corpus
    )
    assert (in_context.citation, in_context.matched_in) == ("12 CFR 1005.6(b)(2)", "context")
    assert (in_text.citation, in_text.matched_in) == ("12 CFR 1005.6(b)(2)(ii)", "text")
    assert (
        parent_citation("Supplement I to Part 1005, comment 6(b)(1)-1")
        == "Supplement I to Part 1005, comment 6(b)(1)-1"
    )
