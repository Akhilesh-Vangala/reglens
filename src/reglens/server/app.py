"""HTTP service and demo UI.

GET  /                      demo page
GET  /api/meta              corpus provenance, model status, example questions
GET  /api/search?q=&k=      retrieval only (BM25 works with no credentials)
GET  /api/chunk/{id}        one passage, verbatim
POST /api/ask               full governed pipeline; per-stage latency, tokens, cost, events
GET  /api/evals             latest evaluation summaries

The model client is created lazily: without credentials the service still serves retrieval and the
verifier, and /api/ask answers 503 with the reason instead of failing at startup.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from reglens.config import ROOT, ModelSettings
from reglens.corpus.chunk import Chunk
from reglens.guard import SpendLedger, redact
from reglens.retrieval import BM25, load_chunks
from reglens.verify import normalize

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
    return {"configured": True, "provider": s.provider, "chat": s.chat_deployment, "embedding": s.embedding_deployment}


@lru_cache(maxsize=1)
def ledger() -> SpendLedger:
    cap = os.getenv("REGLENS_USD_CAP")
    return SpendLedger(ROOT / "traces" / "spend.json", float(cap) if cap else None)


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
        "examples": examples,
    }


@app.get("/api/search")
def search(q: str, k: int = 8, retriever_name: str = "dense_bge"):
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
    retriever: str = "dense_bge"  # best measured retriever (evals/reports, recall@5 0.76)
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
def ask(req: AskRequest):
    status = model_status()
    if not status["configured"]:
        raise HTTPException(503, f"No model configured: {status['reason']}")
    from reglens.generation.generator import ChatGenerator
    from reglens.graph import build_pipeline

    s = ModelSettings.from_env()
    prices = (s.price_input_per_m, s.price_output_per_m) if s.price_input_per_m and s.price_output_per_m else None
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
    cost = (u.get("input_tokens", 0) * prices[0] + u.get("output_tokens", 0) * prices[1]) / 1e6 if prices else None
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
    out = {}
    for d in sorted(REPORTS.glob("*"), reverse=True):
        kind = d.name.split("-", 2)[-1]
        if kind not in out and (d / "summary.json").exists():
            out[kind] = {"run": d.name, **json.loads((d / "summary.json").read_text())}
    return out


def main() -> None:
    import uvicorn

    uvicorn.run("reglens.server.app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
