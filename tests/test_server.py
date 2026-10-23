import pytest
from fastapi.testclient import TestClient

import reglens.generation.generator as genmod
from reglens.generation.generator import ScriptedGenerator
from reglens.server import app as srv

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
    assert r.status_code == 503 and "No model configured" in r.json()["detail"]


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
        srv.ModelSettings, "from_env", classmethod(lambda cls: cls("openai", "", "k", "", "x", "y", None, None, None))
    )
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
