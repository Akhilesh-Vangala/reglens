"""HTTP service and demo UI.

GET  /                      demo page
GET  /api/meta              corpus provenance, model + limiter status, example questions and cases
GET  /api/search?q=&k=      retrieval only
GET  /api/chunk/{id}        one passage, verbatim
GET  /api/graph/{id}        citation-graph neighbors of a passage (Official Interpretations, cross-references)
POST /api/verify            run the deterministic quote verifier on a claim against a passage
POST /api/timeline          deterministic Reg E / Reg Z deadline and liability calculator
POST /api/ask               governed RAG pipeline (model)
POST /api/copilot           member-service copilot agent (model)
GET  /api/evals             latest evaluation summaries

Everything except /api/ask and /api/copilot is deterministic and needs no credentials. The two model endpoints
require a configured model, prices, and (on a public deployment) the durable rate limiter in limits.py; they
answer 503/429 with the reason instead of failing.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from reglens.config import ROOT, STATE_DIR, ModelSettings
from reglens.corpus.chunk import Chunk
from reglens.guard import SpendLedger, redact
from reglens.retrieval import BM25, load_chunks
from reglens.server.limits import Limiter, LimitExceeded, client_id
from reglens.verify import check_claims, normalize

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
STATIC = Path(__file__).parent / "static"
REPORTS = ROOT / "evals" / "reports"

app = FastAPI(title="RegLens", version="0.2.0")


@lru_cache(maxsize=1)
def corpus() -> dict[str, Chunk]:
    return {c.id: c for c in load_chunks(CHUNKS)}


@lru_cache(maxsize=1)
def bm25() -> BM25:
    return BM25(list(corpus().values()))


@lru_cache(maxsize=4)
def retriever(name: str):
    if name == "bm25":
        return bm25()
    from reglens.eval.run import make_retriever

    return make_retriever(name, list(corpus().values()))


def model_status() -> dict:
    try:
        s = ModelSettings.from_env()
    except RuntimeError as exc:
        return {"configured": False, "reason": str(exc)}
    return {"configured": True, "provider": s.provider, "chat": s.chat_deployment}


@lru_cache(maxsize=1)
def graph():
    from reglens.kg import build_graph
    from reglens.kg.neo4j_store import GraphService

    return GraphService(build_graph(list(corpus().values())))


DEFAULT_RETRIEVER = os.getenv("REGLENS_RETRIEVER", "dense_bge")  # qdrant_bge in Kubernetes


@lru_cache(maxsize=1)
def limiter() -> Limiter:
    return Limiter.from_env()


def answering_status() -> dict:
    m = model_status()
    if not m["configured"]:
        return {"enabled": False, "reason": m["reason"]}
    s = ModelSettings.from_env()
    if s.price_input_per_m is None or s.price_output_per_m is None:
        return {"enabled": False, "reason": "model prices are not configured"}
    if not limiter().enabled:
        return {"enabled": False, "reason": "no durable rate limiter configured for this public deployment"}
    return {"enabled": True, "rate_per_hour": limiter().per_hour, "daily_usd_cap": limiter().daily_usd}


def _admit(request: Request, calls: int, in_tokens: int, out_tokens: int):
    """Reserve the worst case of a model request against the per-client and daily limits."""
    st = answering_status()
    if not st["enabled"]:
        raise HTTPException(503, f"Answering is disabled: {st['reason']}")
    s = ModelSettings.from_env()
    worst = calls * (in_tokens * s.price_input_per_m + out_tokens * s.price_output_per_m) / 1e6
    try:
        return limiter().admit(client_id(request.headers), worst), worst, s
    except LimitExceeded as exc:
        raise HTTPException(429, str(exc)) from exc


@lru_cache(maxsize=1)
def ledger() -> SpendLedger:
    cap = os.getenv("REGLENS_USD_CAP")
    return SpendLedger(STATE_DIR / "spend.json", float(cap) if cap else None)


def _hit(h, rank: int) -> dict:
    c = h.chunk
    return {
        "rank": rank,
        "id": c.id,
        "citation": c.citation,
        "heading": c.heading,
        "kind": c.kind,
        "score": round(float(h.score), 4),
        "text": c.text[:400],
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True, "chunks": len(corpus()), "graph_backend": graph().backend, "retriever": DEFAULT_RETRIEVER}


@app.get("/api/meta")
def meta():
    manifest = json.loads((ROOT / "data" / "raw" / "manifest.json").read_text())
    stats = json.loads((ROOT / "data" / "processed" / "corpus_stats.json").read_text())
    golden = [json.loads(x) for x in (ROOT / "evals" / "golden.jsonl").read_text().splitlines() if x.strip()]
    pick = ["E01", "E03", "L01", "C02", "T01", "N01", "R01", "R03"]
    examples = [
        {"id": g["id"], "question": g["question"], "expected": g["expected_status"]} for g in golden if g["id"] in pick
    ]
    return {
        "as_of": manifest["as_of"],
        "parts": {p: {"name": m["name"], "sha256": m["sha256"][:12]} for p, m in manifest["parts"].items()},
        "stats": {k: stats[k] for k in ("passages", "chunks", "by_part", "by_kind")},
        "model": model_status(),
        "answering": answering_status(),
        "examples": examples,
        "cases": COPILOT_CASES,
        "graph": graph().stats(),
    }


COPILOT_CASES = [
    {
        "id": "stolen-card",
        "case": "Member: My debit card was stolen on Tuesday Oct 20 and I called you first thing the "
        "next morning (Oct 21). There's a $400 ATM withdrawal from Oct 20 I didn't make. What happens now?",
    },
    {
        "id": "late-report",
        "case": "Member: I lost my card on Oct 1 but only reported it today, Oct 23. Someone spent "
        "$40 on Oct 2 and $900 on Oct 15. Am I stuck with all of it?",
    },
    {
        "id": "card-dispute",
        "case": "Member: My credit card statement sent Sept 15 has a $250 charge from a store I "
        "never shopped at. I mailed you a dispute letter that you received Oct 1. When will this be fixed?",
    },
    {
        "id": "check-hold",
        "case": "Member: I deposited a $3,000 check yesterday and you put a hold on it. How long can "
        "you hold my money?",
    },
    {"id": "out-of-scope", "case": "Member: Should I sue the store that charged me twice? Write me a demand letter."},
]


@app.get("/api/graph/{chunk_id}")
def graph_neighbors(chunk_id: str):
    c = corpus().get(chunk_id)
    if c is None:
        raise HTTPException(404, "unknown chunk")
    out = []
    for rel, direction, cit in graph().neighbors(c.citation):
        n = graph().chunks_by_citation[cit][0]
        out.append(
            {
                "relation": rel,
                "direction": direction,
                "citation": cit,
                "id": n.id,
                "heading": n.heading,
                "text": n.text[:300],
            }
        )
    return {"citation": c.citation, "neighbors": out}


class VerifyRequest(BaseModel):
    chunk_id: str
    quote: str = Field(max_length=4000)
    retrieved: bool = True


@app.post("/api/verify")
def verify(req: VerifyRequest):
    c = corpus().get(req.chunk_id)
    if c is None:
        raise HTTPException(404, "unknown chunk")
    [chk] = check_claims([{"chunk_id": c.id, "quote": req.quote}], {c.id: c} if req.retrieved else {}, corpus())
    src = c.source_text()
    return {**chk.__dict__, "span": _span(src, req.quote) if chk.verified else None, "source": src}


@app.post("/api/timeline")
def timeline(req: dict):
    from reglens.copilot import DeadlineArgs, _run_deadlines

    try:
        out = _run_deadlines(DeadlineArgs(**req))
    except Exception as exc:  # noqa: BLE001  (validation errors are the caller's)
        raise HTTPException(422, str(exc)[:300]) from exc
    if "error" in out:
        raise HTTPException(422, out["error"])
    return out


class CopilotRequest(BaseModel):
    case: str = Field(min_length=10, max_length=3000)


@app.post("/api/copilot")
def copilot(req: CopilotRequest, request: Request):
    from reglens.copilot import MAX_STEPS, build_copilot, chat_model

    day, reserved, s = _admit(request, MAX_STEPS, 12_000, 4_000)
    app_ = build_copilot(
        retriever(DEFAULT_RETRIEVER),
        graph(),
        corpus(),
        chat_model(s),
        ledger=ledger(),
        prices=(s.price_input_per_m, s.price_output_per_m),
    )
    t0 = time.perf_counter()
    state = app_.invoke({"case": req.case})
    actual = state.get("usage", {}).get("usd", reserved)
    limiter().settle(day, reserved, actual)
    r = state["result"]
    for c in r["claims"]:
        c["span"] = _span(c["source"], c["quote"])
    return {
        "result": r,
        "steps": state["steps"],
        "usage": state["usage"],
        "redactions": state["redactions"],
        "seconds": round(time.perf_counter() - t0, 2),
        "model": model_status(),
    }


@app.get("/api/search")
def search(q: str, k: int = 8, retriever_name: str = DEFAULT_RETRIEVER):
    r = redact(q)
    t0 = time.perf_counter()
    try:
        hits = retriever(retriever_name).search(r.text, min(k, 20))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "query": r.text,
        "redactions": r.counts,
        "retriever": retriever_name,
        "latency_ms": round(1000 * (time.perf_counter() - t0), 1),
        "hits": [_hit(h, i + 1) for i, h in enumerate(hits)],
    }


@app.get("/api/chunk/{chunk_id}")
def chunk(chunk_id: str):
    c = corpus().get(chunk_id)
    if c is None:
        raise HTTPException(404, "unknown chunk")
    return c.to_dict()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    retriever: str = DEFAULT_RETRIEVER  # dense bge: best measured retriever (evals/reports, recall@5 0.76)
    k: int = Field(default=8, ge=1, le=20)


def _span(text: str, quote: str) -> list[int] | None:
    """Character span of a verified quote inside the passage, for highlighting. The verifier matches on
    normalized text; this maps back by searching the normalized passage with an index map."""
    norm_chars, index = [], []
    prev_space = False
    for i, ch in enumerate(text):
        n = normalize(ch) or " "
        if n == " ":
            if prev_space or not norm_chars:
                continue
            prev_space = True
        else:
            prev_space = False
        norm_chars.append(n)
        index.append(i)
    pos = "".join(norm_chars).find(normalize(quote))
    if pos < 0:
        return None
    end = pos + len(normalize(quote)) - 1
    return [index[pos], index[end] + 1]


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    from reglens.generation.generator import ChatGenerator
    from reglens.graph import build_pipeline

    status = model_status()
    day, reserved, s = _admit(request, 2, 8_000, 8_000)
    prices = (s.price_input_per_m, s.price_output_per_m)
    app_ = build_pipeline(retriever(req.retriever), ChatGenerator(s), corpus(), k=req.k, ledger=ledger(), prices=prices)
    stages, state, t_prev = [], {}, time.perf_counter()
    for update in app_.stream({"question": req.question}, stream_mode="updates"):
        now = time.perf_counter()
        for node, delta in update.items():
            stages.append({"node": node, "ms": round(1000 * (now - t_prev), 1)})
            state.update(delta or {})
        t_prev = now
    final = state["final"]
    for c in final["claims"]:
        ch = corpus()[c["chunk_id"]]
        c["citation"] = ch.citation
        c["source"] = ch.source_text()
        c["span"] = _span(c["source"], c["quote"])
    u = state.get("usage", {})
    cost = (u.get("input_tokens", 0) * prices[0] + u.get("output_tokens", 0) * prices[1]) / 1e6
    limiter().settle(day, reserved, cost)
    return {
        "final": final,
        "redactions": state.get("redactions", {}),
        "retrieved": [
            {"id": i, "citation": corpus()[i].citation, "heading": corpus()[i].heading}
            for i in state.get("retrieved", [])
        ],
        "checks": state.get("checks", []),
        "events": state.get("events", []),
        "usage": {**u, "usd": round(cost, 6) if cost is not None else None},
        "stages": stages,
        "model": status,
    }


@app.get("/api/evals")
def evals():
    """Latest report per kind, plus every retriever's most recent result measured on the current golden set
    (ablations run at different times are only comparable when the golden set hash matches)."""
    out: dict = {}
    runs = sorted(REPORTS.glob("*"), reverse=True)
    for d in runs:
        kind = d.name.split("-", 2)[-1]
        if kind not in out and (d / "summary.json").exists():
            out[kind] = {"run": d.name, **json.loads((d / "summary.json").read_text())}
    if "retrieval" in out:
        golden = out["retrieval"]["provenance"]["golden_sha256"]
        merged: dict = {}
        for d in runs:
            if d.name.endswith("-retrieval") and (d / "summary.json").exists():
                s = json.loads((d / "summary.json").read_text())
                if s["provenance"]["golden_sha256"] == golden:
                    for name, m in s["retrievers"].items():
                        merged.setdefault(name, {"run": d.name, **m})
        out["retrieval_all"] = merged
    return out


def main() -> None:
    import uvicorn

    uvicorn.run("reglens.server.app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
