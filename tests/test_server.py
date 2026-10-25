import pytest
from fastapi.testclient import TestClient

import reglens.generation.generator as genmod
from reglens.generation.generator import ScriptedGenerator
from reglens.server import app as srv
from reglens.server import limits as srv_limits

client = TestClient(srv.app)


def test_meta_reports_pinned_corpus():
    m = client.get("/api/meta").json()
    assert (
        m["as_of"] == "2026-10-02"
        and m["stats"]["chunks"] == 2495
        and set(m["parts"]) == {"1005", "707", "229", "1026"}
    )


def test_search_masks_pii_and_returns_real_passages():
    r = client.get(
        "/api/search",
        params={"q": "my card 4111 1111 1111 1111 was stolen, what is my liability", "retriever_name": "bm25"},
    ).json()
    assert "4111" not in r["query"] and r["redactions"] == {"card_number": 1}
    assert len(r["hits"]) == 8 and all(h["citation"] for h in r["hits"])


def test_chunk_lookup_and_404():
    cid = client.get("/api/search", params={"q": "error resolution", "retriever_name": "bm25"}).json()["hits"][0]["id"]
    assert client.get(f"/api/chunk/{cid}").json()["id"] == cid
    assert client.get("/api/chunk/nope").status_code == 404


def test_ask_without_model_is_503_not_500(monkeypatch):
    monkeypatch.setattr(srv, "model_status", lambda: {"configured": False, "reason": "Missing OPENAI_API_KEY"})
    r = client.post("/api/ask", json={"question": "how long to investigate an error?"})
    assert r.status_code == 503 and "Answering is disabled" in r.json()["detail"]


def test_ask_end_to_end_with_scripted_model(monkeypatch):
    hit = client.get(
        "/api/search",
        params={"q": "investigate promptly within 10 business days notice of error", "retriever_name": "bm25"},
    ).json()
    chunk = srv.corpus()[hit["hits"][0]["id"]]
    quote = chunk.text[10:80]
    answer = {
        "status": "answered",
        "answer": "Ten business days.",
        "needs_human_review": False,
        "review_reason": "",
        "claims": [{"statement": "s", "chunk_id": chunk.id, "quote": quote}],
    }
    monkeypatch.setattr(
        srv, "model_status", lambda: {"configured": True, "provider": "scripted", "chat": "x", "embedding": "y"}
    )
    monkeypatch.setattr(
        srv.ModelSettings, "from_env", classmethod(lambda cls: cls("openai", "", "k", "", "x", "y", 0.1, 0.5, None))
    )
    monkeypatch.setattr(srv, "limiter", lambda: srv.Limiter(srv_limits.MemoryStore(), 10, 1.0))
    monkeypatch.setattr(genmod, "ChatGenerator", lambda s: ScriptedGenerator([answer]))
    r = client.post(
        "/api/ask",
        json={"question": "investigate promptly within 10 business days notice of error", "retriever": "bm25"},
    ).json()
    c = r["final"]["claims"][0]
    assert r["final"]["status"] == "answered" and c["citation"] == chunk.citation
    assert c["source"][c["span"][0] : c["span"][1]] == quote
    assert [s["node"] for s in r["stages"]] == ["guard", "retrieve", "generate", "verify", "finalize"]


@pytest.mark.parametrize("text,quote", [("Hello  “World”   again", '"world" again'), ("a\nb c", "b c")])
def test_span_maps_normalized_quote_back_to_source(text, quote):
    s = srv._span(text, quote)
    assert s and srv.normalize(text[s[0] : s[1]]) == srv.normalize(quote)


SETTINGS = classmethod(lambda cls: cls("anthropic", "", "k", "", "claude-haiku-5-5", "", 0.1, 0.5, None))


def _configured(monkeypatch, store, per_hour=10, daily=1.0):
    monkeypatch.setattr(
        srv, "model_status", lambda: {"configured": True, "provider": "anthropic", "chat": "h", "embedding": ""}
    )
    monkeypatch.setattr(srv.ModelSettings, "from_env", SETTINGS)
    monkeypatch.setattr(srv, "limiter", lambda: srv.Limiter(store, per_hour, daily))


def test_public_deployment_without_durable_limiter_fails_closed(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("KV_REST_API_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    _configured(monkeypatch, srv_limits.store_from_env())
    r = client.post("/api/copilot", json={"case": "my card was stolen yesterday"})
    assert r.status_code == 503 and "no durable rate limiter" in r.json()["detail"]


def test_per_client_rate_limit_returns_429(monkeypatch):
    store = srv_limits.MemoryStore()
    _configured(monkeypatch, store, per_hour=1)
    srv.limiter().admit("local", 0.0)  # first request of this hour used up
    r = client.post("/api/copilot", json={"case": "my card was stolen yesterday"})
    assert r.status_code == 429 and "rate limit" in r.json()["detail"]


def test_daily_budget_reserves_worst_case_and_refuses_past_cap(monkeypatch):
    lim = srv.Limiter(srv_limits.MemoryStore(), 100, daily_usd=0.05)
    day = lim.admit("a", 0.04)
    with pytest.raises(srv_limits.LimitExceeded, match="daily demo budget"):
        lim.admit("b", 0.04)
    lim.settle(day, 0.04, 0.001)  # actual cost was tiny: the reservation is released
    lim.admit("b", 0.04)


def test_verify_endpoint_accepts_verbatim_and_rejects_edits():
    cid = [c.id for c in srv.corpus().values() if c.citation == "12 CFR 1005.11(c)(1)"][0]
    good = "shall determine whether an error occurred within 10 business days of receiving a notice of error"
    ok = client.post("/api/verify", json={"chunk_id": cid, "quote": good}).json()
    bad = client.post("/api/verify", json={"chunk_id": cid, "quote": good.replace("10", "30")}).json()
    assert ok["verified"] and ok["source"][ok["span"][0] : ok["span"][1]].lower().startswith("shall determine")
    assert not bad["verified"] and bad["reason"] == "quote_not_in_source"


def test_timeline_and_graph_endpoints_need_no_model():
    t = client.post(
        "/api/timeline",
        json={
            "case_type": "unauthorized_eft",
            "notified": "2026-10-21",
            "learned_of_loss": "2026-10-20",
            "transfers": [{"date": "2026-10-20", "amount": 400}],
        },
    ).json()
    assert t["items"][0]["amount"] == 50.0 and t["items"][0]["citation"] == "12 CFR 1005.6(b)(1)"
    cid = [c.id for c in srv.corpus().values() if c.citation == "12 CFR 1005.6(b)(1)"][0]
    g = client.get(f"/api/graph/{cid}").json()
    assert any(n["relation"] == "INTERPRETS" for n in g["neighbors"])
    assert client.post("/api/timeline", json={"case_type": "nope", "notified": "x"}).status_code == 422


def test_copilot_endpoint_end_to_end_with_scripted_model(monkeypatch):
    import reglens.copilot as cp
    from tests.test_copilot import DEADLINES, GOOD, ScriptedChat, submit

    _configured(monkeypatch, srv_limits.MemoryStore())
    turns = [
        [("search_regulations", {"query": "stolen debit card liability", "part": "1005"})],
        [DEADLINES],
        [submit("The most you could owe is $50.", [GOOD])],
    ]
    monkeypatch.setattr(cp, "chat_model", lambda s: ScriptedChat(turns))
    monkeypatch.setattr(srv, "retriever", lambda name: __import__("tests.test_copilot").test_copilot.Fixed())
    r = client.post("/api/copilot", json={"case": "card stolen 2026-10-20, called 2026-10-21, $400 charge"}).json()
    assert r["result"]["claims"][0]["citation"] == "12 CFR 1005.6(b)(1)" and r["result"]["claims"][0]["span"]
    assert r["usage"]["calls"] == 3 and r["usage"]["usd"] > 0
