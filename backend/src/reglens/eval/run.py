"""Evaluation runner.

  reglens eval retrieval --retrievers bm25 dense_bge hybrid_bge dense_bge+rerank graph_bge_s1
  reglens eval answers  --configs A_closed_book B_rag_no_verify C_rag_verified --repeats 1
  reglens eval stores   (needs QDRANT_URL and NEO4J_URI)

Writes evals/reports/<stamp>-<kind>/{results.jsonl, summary.json, REPORT.md}. A report records the
corpus manifest hash, prompt version, model deployments and git commit so every number is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from reglens.config import ROOT, ModelSettings
from reglens.corpus.chunk import Chunk
from reglens.eval.metrics import bootstrap, paired_delta, retrieval_metrics, score_closed_book, score_rag
from reglens.generation.prompts import CLOSED_BOOK_TEMPLATE, POLICY_CLOSED, PROMPT_VERSION
from reglens.generation.schema import ClosedBookAnswer
from reglens.retrieval import load_chunks

GOLDEN = ROOT / "evals" / "golden.jsonl"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def provenance(extra: dict) -> dict:
    manifest = (ROOT / "data" / "raw" / "manifest.json").read_bytes()
    try:
        commit = (
            subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "uncommitted"
        )
    except OSError:
        commit = "unknown"
    return {
        "corpus_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "golden_sha256": hashlib.sha256(GOLDEN.read_bytes()).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "git_commit": commit,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }


def make_retriever(name: str, chunks: list[Chunk]):
    from reglens.retrieval import BM25, Dense, LocalHybrid

    if name == "bm25":
        return BM25(chunks)
    if name.endswith("+rerank"):
        from reglens.retrieval.local import Reranked

        return Reranked(make_retriever(name.removesuffix("+rerank"), chunks))
    if name in ("qdrant_bge", "qdrant_bge_exact"):
        from reglens.retrieval.qdrant import from_env

        return from_env(make_retriever("dense_bge", chunks), chunks, exact=name.endswith("exact"))
    if name.startswith("graph_bge"):  # graph_bge_s<seeds>
        from reglens.kg import build_graph
        from reglens.retrieval.graph import GraphExpanded

        seeds = int(name.rsplit("_s", 1)[1]) if "_s" in name else 3
        return GraphExpanded(make_retriever("dense_bge", chunks), build_graph(chunks), seeds=seeds)
    if name in ("dense_bge", "hybrid_bge"):
        from reglens.retrieval.local import LocalEmbedder

        emb = LocalEmbedder()
        dense = Dense(chunks, emb, ROOT / "data" / "processed" / f"embeddings_{emb.deployment}.npz")
        dense.name = name if name == "dense_bge" else "dense_bge"
        if name == "dense_bge":
            return dense
        hybrid = LocalHybrid(BM25(chunks), dense)
        hybrid.name = name
        return hybrid
    raise ValueError(f"unknown retriever {name!r}")


def _write(kind: str, rows: list[dict], summary: dict, report: str) -> Path:
    out = ROOT / "evals" / "reports" / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "REPORT.md").write_text(report)
    return out


def eval_retrieval(retriever_names: list[str], k: int = 8) -> Path:
    chunks, golden = load_chunks(CHUNKS), [g for g in load_golden() if g["expected_status"] == "answered"]
    rows, summary = [], {"provenance": provenance({"kind": "retrieval", "k": k}), "retrievers": {}}
    for name in retriever_names:
        r = make_retriever(name, chunks)
        per = []
        for g in golden:
            t0 = time.perf_counter()
            hits = r.search(g["question"], k)
            m = retrieval_metrics([h.chunk.citation for h in hits], g["gold"])
            row = {
                "retriever": name,
                "id": g["id"],
                "category": g["category"],
                **m,
                "latency_ms": round(1000 * (time.perf_counter() - t0), 1),
                "retrieved": [h.chunk.citation for h in hits],
            }
            rows.append(row)
            per.append(row)
        summary["retrievers"][name] = {
            metric: dict(
                zip(("mean", "lo", "hi"), (round(x, 3) for x in bootstrap([p[metric] for p in per])), strict=True)
            )
            for metric in ("recall@1", "recall@3", "recall@5", "recall@8", "mrr")
        }
        summary["retrievers"][name]["latency_ms_median"] = statistics.median(p["latency_ms"] for p in per)
    names = list(summary["retrievers"])
    by = {n: {r["id"]: r["recall@5"] for r in rows if r["retriever"] == n} for n in names}
    summary["paired_recall@5"] = {f"{a}->{b}": paired_delta(by[a], by[b]) for a in names for b in names if a < b}
    lines = [
        "# Retrieval evaluation",
        "",
        f"{len(golden)} answerable golden questions, k={k}.",
        "",
        "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@8 | MRR | p50 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, s in summary["retrievers"].items():
        fmt = lambda m, s=s: f"{s[m]['mean']:.2f} ({s[m]['lo']:.2f}-{s[m]['hi']:.2f})"
        lines.append(
            f"| {n} | {fmt('recall@1')} | {fmt('recall@3')} | {fmt('recall@5')} | {fmt('recall@8')} | "
            f"{fmt('mrr')} | {s['latency_ms_median']:.1f} |"
        )
    out = _write("retrieval", rows, summary, "\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def budget_from_env(s: ModelSettings):
    """Real-model runs must be capped: REGLENS_USD_CAP and both prices are required."""
    import os

    from reglens.config import STATE_DIR
    from reglens.guard import SpendLedger

    cap = os.getenv("REGLENS_USD_CAP")
    if not cap or s.price_input_per_m is None or s.price_output_per_m is None:
        raise RuntimeError("Set REGLENS_USD_CAP, PRICE_INPUT_PER_M and PRICE_OUTPUT_PER_M before any model run")
    return SpendLedger(STATE_DIR / "spend.json", float(cap), max_tokens_per_request=60_000)


def estimate_answers(configs: list[str], retriever_name: str = "dense_bge", repeats: int = 1, k: int = 8) -> dict:
    """Upper-bound cost of an answer run without calling a model: real prompt lengths, every RAG question
    assumed to need the verifier retry, every call assumed to use its full output allowance."""
    from reglens.copilot import MAX_STEPS
    from reglens.generation.prompts import POLICY, RAG_TEMPLATE, format_passages

    s = ModelSettings.from_env()
    chunks = load_chunks(CHUNKS)
    retriever = make_retriever(retriever_name, chunks)
    max_out = 8000 if s.provider == "anthropic" else 4000
    tok_in = tok_out = calls = 0
    for g in load_golden():
        rag = RAG_TEMPLATE.format(
            policy=POLICY,
            passages=format_passages([h.chunk for h in retriever.search(g["question"], k)]),
            question=g["question"],
        )
        closed = CLOSED_BOOK_TEMPLATE.format(policy_closed=POLICY_CLOSED, question=g["question"])
        for cfg in configs:
            if cfg == "D_copilot":  # worst case: every agent turn used, each with a full context and output
                tok_in += repeats * MAX_STEPS * 12_000
                tok_out += repeats * MAX_STEPS * max_out
                calls += repeats * MAX_STEPS
                continue
            n = 1 if cfg != "C_rag_verified" else 2
            p = closed if cfg == "A_closed_book" else rag
            tok_in += n * repeats * (len(p) // 3 + (600 if n == 2 else 0))
            tok_out += n * repeats * max_out
            calls += n * repeats
    usd = (tok_in * s.price_input_per_m + tok_out * s.price_output_per_m) / 1e6 if s.price_input_per_m else None
    return {
        "model": s.chat_deployment,
        "calls_max": calls,
        "input_tokens_max": tok_in,
        "output_tokens_max": tok_out,
        "usd_upper_bound": round(usd, 2) if usd is not None else None,
    }


def eval_answers(configs: list[str], retriever_name: str = "dense_bge", repeats: int = 1, k: int = 8) -> Path:
    from reglens.generation.generator import Budgeted, ChatGenerator
    from reglens.graph import build_pipeline
    from reglens.guard import BudgetExceeded

    s = ModelSettings.from_env()
    chunks = load_chunks(CHUNKS)
    corpus = {c.id: c for c in chunks}
    corpus_citations = {c.citation for c in chunks}
    golden = load_golden()
    ledger = budget_from_env(s)
    raw = ChatGenerator(s)
    gen = Budgeted(raw, ledger, s.price_input_per_m, s.price_output_per_m, raw.max_output_tokens)
    retriever = make_retriever(retriever_name, chunks) if any(c != "A_closed_book" for c in configs) else None
    rows, stopped = [], None
    try:
        _answer_rows(configs, repeats, golden, gen, retriever, corpus, corpus_citations, k, s, rows, build_pipeline)
    except BudgetExceeded as exc:
        stopped = f"budget: {exc}"
        print(f"STOPPED: {stopped}", flush=True)
    summary = summarize(rows, [c for c in configs if any(r["config"] == c for r in rows)])
    summary["stopped"] = stopped
    summary["spend_ledger"] = dict(ledger.state)
    summary["provenance"] = provenance(
        {
            "kind": "answers",
            "retriever": retriever_name,
            "k": k,
            "repeats": repeats,
            "provider": s.provider,
            "chat_deployment": s.chat_deployment,
            "embeddings": "bge-small-en-v1.5 (local)",
            "api_version": s.api_version,
            "prices_per_m": [s.price_input_per_m, s.price_output_per_m],
            "usd_cap": ledger.usd_cap,
        }
    )
    out = _write("answers", rows, summary, render(summary))
    print(render(summary))
    return out


def _copilot_as_pipeline_answer(state: dict) -> tuple[dict, dict]:
    """Map a copilot run onto the pipeline's answer shape so configuration D is scored by the same function."""
    r, sub = state["result"], state.get("submitted") or {}
    status = (
        "refused"
        if r["case_type"] == "out_of_scope"
        else "not_in_corpus"
        if r["case_type"] == "not_covered" or r["draft_withheld"]
        else "answered"
    )
    final = {
        "status": status,
        "answer": r["reply_to_member"],
        "claims": r["claims"],
        "needs_human_review": r["needs_human_review"],
    }
    return final, {"claims": sub.get("claims", [])}


def _answer_rows(configs, repeats, golden, gen, retriever, corpus, corpus_citations, k, s, rows, build_pipeline):
    from reglens.guard import BudgetExceeded

    for cfg in configs:
        if cfg == "D_copilot":
            from reglens.copilot import build_copilot, chat_model
            from reglens.kg import build_graph

            app = build_copilot(
                retriever,
                build_graph(list(corpus.values())),
                corpus,
                chat_model(s),
                ledger=gen.ledger,
                prices=(s.price_input_per_m, s.price_output_per_m, s.price_cache_read_per_m),
            )
        elif cfg == "A_closed_book":
            app = None
        else:
            app = build_pipeline(retriever, gen, corpus, k=k, verify=(cfg == "C_rag_verified"))
        for rep in range(repeats):
            for g in golden:
                t0 = time.perf_counter()
                if cfg == "A_closed_book":
                    prompt = CLOSED_BOOK_TEMPLATE.format(policy_closed=POLICY_CLOSED, question=g["question"])
                    res = gen.generate(prompt, ClosedBookAnswer)
                    final = res.parsed or {}
                    score = score_closed_book(g, final, corpus_citations)
                    usage = {"input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "calls": 1}
                    events, err = [], res.error
                elif cfg == "D_copilot":
                    state = app.invoke({"case": g["question"]})
                    if any(e.startswith("budget_exceeded") for e in state["result"]["events"]):
                        raise BudgetExceeded("copilot stopped by the spend cap")
                    final, draft = _copilot_as_pipeline_answer(state)
                    retrieved = {i: corpus[i] for i in state["seen"] if i in corpus}
                    score = score_rag(g, final, draft, retrieved, corpus)
                    usage, events, err = state["usage"], state["result"]["events"], None
                else:
                    state = app.invoke({"question": g["question"]})
                    final = state["final"]
                    retrieved = {i: corpus[i] for i in state["retrieved"]}
                    score = score_rag(g, final, state.get("draft"), retrieved, corpus)
                    usage, events = state["usage"], state["events"]
                    err = next((e for e in events if e.startswith("generation_error")), None)
                cost = usage.get("usd")  # the copilot prices each call itself, including cache reads and writes
                if cost is None and s.price_input_per_m is not None and s.price_output_per_m is not None:
                    cost = (
                        usage["input_tokens"] * s.price_input_per_m + usage["output_tokens"] * s.price_output_per_m
                    ) / 1e6
                rows.append(
                    {
                        "config": cfg,
                        "repeat": rep,
                        "id": g["id"],
                        "category": g["category"],
                        "expected_status": g["expected_status"],
                        "status": final.get("status"),
                        **score,
                        **{f"usage_{k2}": v for k2, v in usage.items() if k2 != "seconds"},
                        "latency_s": round(time.perf_counter() - t0, 2),
                        "cost_usd": cost,
                        "events": events,
                        "error": err,
                        "answer": final.get("answer", "")[:600],
                    }
                )
                correct = score.get("answer_correct", score["status_correct"])
                print(f"[{cfg}] {g['id']} status={final.get('status')} correct={correct}", flush=True)


def _mean_ci(vals: list[float]) -> dict:
    m, lo, hi = bootstrap(vals)
    return {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": len(vals)}


def summarize(rows: list[dict], configs: list[str]) -> dict:
    out: dict = {"configs": {}}
    per_item: dict[str, dict[str, float]] = {}
    for cfg in configs:
        rs = [r for r in rows if r["config"] == cfg]
        ans = [r for r in rs if r["expected_status"] == "answered"]
        # average repeats per question first, so CIs are over questions
        by_q: dict[str, list[float]] = {}
        for r in ans:
            by_q.setdefault(r["id"], []).append(float(r.get("answer_correct", False)))
        per_item[cfg] = {q: statistics.fmean(v) for q, v in by_q.items()}
        claims = sum(r.get("n_claims_shown", 0) for r in rs)
        drafted = sum(r.get("n_claims_drafted", r.get("n_claims_shown", 0)) for r in rs)
        s = {
            "answer_accuracy": _mean_ci(list(per_item[cfg].values())),
            "status_accuracy": _mean_ci([float(r["status_correct"]) for r in rs]),
            "refusal_accuracy": _mean_ci([float(r["status_correct"]) for r in rs if r["expected_status"] == "refused"]),
            "not_in_corpus_accuracy": _mean_ci(
                [float(r["status_correct"]) for r in rs if r["expected_status"] == "not_in_corpus"]
            ),
            "escalation_accuracy": _mean_ci([float(r["review_correct"]) for r in rs if "review_correct" in r]),
            "claims_shown": claims,
            "errors": sum(1 for r in rs if r.get("error")),
            "latency_s_median": statistics.median(r["latency_s"] for r in rs),
            "input_tokens_mean": round(statistics.fmean(r.get("usage_input_tokens", 0) for r in rs), 1),
            "output_tokens_mean": round(statistics.fmean(r.get("usage_output_tokens", 0) for r in rs), 1),
            "cost_usd_per_question": (
                round(statistics.fmean(r["cost_usd"] for r in rs), 6)
                if all(r["cost_usd"] is not None for r in rs)
                else None
            ),
        }

        # Like-for-like metrics. answer_accuracy is stricter for retrieval configs (it also needs a verified quote from
        # a gold paragraph), so these three put every configuration on the same footing.
        def per_q(rows_, fn):
            g: dict[str, list[float]] = {}
            for r in rows_:
                g.setdefault(r["id"], []).append(float(fn(r)))
            return [statistics.fmean(v) for v in g.values()]

        s["right_fact"] = _mean_ci(per_q(ans, lambda r: r["status"] == "answered" and r.get("facts_present", False)))
        s["verifiably_grounded"] = (
            None
            if cfg == "A_closed_book"
            else _mean_ci(per_q(ans, lambda r: r["status"] == "answered" and r.get("grounded_in_gold", False)))
        )
        s["declined_correctly"] = _mean_ci(
            per_q(
                [r for r in rs if r["expected_status"] != "answered"],
                lambda r: r["status"] in ("refused", "not_in_corpus"),
            )
        )
        if cfg == "A_closed_book":
            fab = sum(r.get("n_fabricated_citations", 0) for r in rs)
            s["fabricated_citation_rate"] = round(fab / claims, 3) if claims else None
        else:
            uns_shown = sum(r.get("n_unsupported_shown", 0) for r in rs)
            uns_draft = sum(r.get("n_unsupported_drafted", 0) for r in rs)
            s["unsupported_claim_rate_shown"] = round(uns_shown / claims, 3) if claims else None
            s["unsupported_claim_rate_drafted"] = round(uns_draft / drafted, 3) if drafted else None
            s["grounded_in_gold"] = _mean_ci(
                [float(r.get("grounded_in_gold", False)) for r in rs if r["expected_status"] == "answered"]
            )
        out["configs"][cfg] = s
    pairs = [
        ("A_closed_book", "B_rag_no_verify"),
        ("B_rag_no_verify", "C_rag_verified"),
        ("A_closed_book", "C_rag_verified"),
    ]
    out["paired_answer_accuracy"] = {
        f"{a}->{b}": paired_delta(per_item[a], per_item[b]) for a, b in pairs if a in per_item and b in per_item
    }
    return out


def rescore(report_dir: Path) -> Path:
    """Recompute summary.json and REPORT.md from a report's stored per-question rows, with no model calls. Used when
    a metric is added: the raw results and their provenance stay exactly as recorded."""
    rows = [json.loads(x) for x in (report_dir / "results.jsonl").read_text().splitlines() if x.strip()]
    old = json.loads((report_dir / "summary.json").read_text())
    summary = summarize(rows, list(old["configs"]))
    for key in ("stopped", "spend_ledger", "provenance"):
        if key in old:
            summary[key] = old[key]
    summary["rescored"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "by_commit": _git_commit()}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (report_dir / "REPORT.md").write_text(render(summary))
    return report_dir


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except OSError:
        return "unknown"


def render(summary: dict) -> str:
    p = summary.get("provenance", {})
    lines = [
        "# Answer evaluation",
        "",
        f"Model `{p.get('chat_deployment')}`, embeddings `{p.get('embeddings')}`, retriever "
        f"`{p.get('retriever')}` (k={p.get('k')}), prompt `{p.get('prompt_version')}`, repeats {p.get('repeats')}.",
        f"Corpus manifest sha256 `{str(p.get('corpus_manifest_sha256'))[:12]}`, golden sha256 "
        f"`{str(p.get('golden_sha256'))[:12]}`, commit `{p.get('git_commit')}`.",
        "",
        "| Config | Answer accuracy (95% CI) | Status acc. | Refusal | Not-in-corpus | Escalation | "
        "Unsupported claims shown | Fabricated citations | p50 s | Cost/question |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    f = lambda d: f"{d['mean']:.2f} ({d['ci95'][0]:.2f}-{d['ci95'][1]:.2f})" if d["n"] else "n/a"
    for cfg, s in summary["configs"].items():
        uns = s.get("unsupported_claim_rate_shown")
        fab = s.get("fabricated_citation_rate")
        cost = s["cost_usd_per_question"]
        lines.append(
            f"| {cfg} | {f(s['answer_accuracy'])} | {f(s['status_accuracy'])} | {f(s['refusal_accuracy'])} | "
            f"{f(s['not_in_corpus_accuracy'])} | {f(s['escalation_accuracy'])} | "
            f"{'n/a' if uns is None else f'{uns:.1%}'} | {'n/a' if fab is None else f'{fab:.1%}'} | "
            f"{s['latency_s_median']:.1f} | {'not priced' if cost is None else f'${cost:.5f}'} |"
        )
    lines += [
        "",
        "## Like for like",
        "",
        "Answer accuracy above is stricter for retrieval configurations (it also requires a verified quote from a gold "
        "paragraph). These columns use one definition for every configuration.",
        "",
        "| Config | Right fact | Verifiably grounded | Declined when it should |",
        "|---|---|---|---|",
    ]
    for cfg, s in summary["configs"].items():
        vg = s.get("verifiably_grounded")
        lines.append(
            f"| {cfg} | {f(s['right_fact'])} | {'n/a (no sources)' if vg is None else f(vg)} | "
            f"{f(s['declined_correctly'])} |"
        )
    lines += [
        "",
        "## Paired differences in answer accuracy (same questions)",
        "",
        "| Contrast | Delta (95% CI) | Significant | Improved | Regressed |",
        "|---|---|---|---|---|",
    ]
    for k, d in summary.get("paired_answer_accuracy", {}).items():
        lines.append(
            f"| {k} | {d['delta']:+.2f} ({d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f}) | "
            f"{'yes' if d['significant'] else 'no'} | {len(d['improved'])} | {', '.join(d['regressed']) or '-'} |"
        )
    lines += [
        "",
        "Unsupported claims are claims whose quote is not verbatim in a passage retrieved for that question. "
        "Fabricated citations (closed book) are cited paragraphs that do not exist in the pinned corpus. "
        "Grading is deterministic: regexes, citation prefixes and the verifier; no LLM judge.",
    ]
    return "\n".join(lines) + "\n"


def eval_stores(k: int = 8) -> Path:
    """Hold the Qdrant and Neo4j backends to the in-memory reference implementations on the pinned corpus.
    Needs QDRANT_URL and NEO4J_URI/NEO4J_PASSWORD (docker compose or the Kubernetes manifests)."""
    from reglens.kg import build_graph
    from reglens.kg.neo4j_store import Neo4jGraph
    from reglens.retrieval.qdrant import QdrantDense

    chunks, golden = load_chunks(CHUNKS), load_golden()
    dense = make_retriever("dense_bge", chunks)
    import os

    url = os.environ["QDRANT_URL"]
    t0 = time.perf_counter()
    hnsw = QdrantDense(dense, "reglens_eval_stores", url=url)
    build_s = time.perf_counter() - t0
    exact = QdrantDense(dense, "reglens_eval_stores", url=url, exact=True)
    identical, overlap, lat = 0, [], {"numpy": [], "qdrant_hnsw": []}
    for g in golden:
        t = time.perf_counter()
        ref = [h.chunk.id for h in dense.search(g["question"], k)]
        lat["numpy"].append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        got = [h.chunk.id for h in hnsw.search(g["question"], k)]
        lat["qdrant_hnsw"].append(1000 * (time.perf_counter() - t))
        identical += ref == [h.chunk.id for h in exact.search(g["question"], k)]
        overlap.append(len(set(ref) & set(got)) / k)
    graph = build_graph(chunks)
    neo = Neo4jGraph.from_env()
    t0 = time.perf_counter()
    loaded = neo.load(graph)
    load_s = time.perf_counter() - t0
    mismatches = [c for c in graph.chunks_by_citation if neo.neighbors(c) != graph.neighbors(c)]
    neo.close()
    summary = {
        "provenance": provenance({"kind": "stores", "k": k}),
        "qdrant": {
            "points": len(chunks),
            "dims": int(dense.matrix.shape[1]),
            "index_build_s": round(build_s, 2),
            "exact_top_k_identical": f"{identical}/{len(golden)}",
            "hnsw_recall_at_k_vs_exact": round(statistics.fmean(overlap), 4),
            "p50_ms": {n: round(statistics.median(v), 2) for n, v in lat.items()},
        },
        "neo4j": {
            **loaded,
            "load_s": round(load_s, 2),
            "graph_stats": graph.stats(),
            "neighbor_mismatches": len(mismatches),
            "nodes_checked": len(graph.chunks_by_citation),
        },
    }
    q = summary["qdrant"]
    report = "\n".join(
        [
            "# Store parity",
            "",
            f"Qdrant: {q['points']} points x {q['dims']} dims; exact top-{k} identical to numpy on "
            f"{q['exact_top_k_identical']} golden queries; HNSW recall@{k} vs exact {q['hnsw_recall_at_k_vs_exact']}.",
            f"Neo4j: {loaded['nodes']} nodes, {loaded['edges']} edges; neighbor mismatches vs in-memory: "
            f"{len(mismatches)} of {len(graph.chunks_by_citation)}.",
            "",
        ]
    )
    out = _write("stores", [], summary, report)
    print(report)
    return out


COPILOT_CASES = ROOT / "evals" / "copilot_cases.jsonl"


def load_copilot_cases() -> list[dict]:
    return [json.loads(x) for x in COPILOT_CASES.read_text().splitlines() if x.strip()]


def score_copilot_case(case: dict, result: dict, submitted: dict | None) -> dict:
    """Did the agent classify the case, extract the right dates into the calculator, hand off when it must, and
    release a reply? Expected calculator outputs are derived from reference arguments
    (backend/tests/test_copilot_cases)."""
    computed = [i for t in result["timelines"] for i in t["items"]]
    hits = [
        any(
            i["citation"] == e["citation"] and all(i.get(f) == v for f, v in e.items() if f != "citation")
            for i in computed
        )
        for e in case["expect_items"]
    ]
    drafted = len((submitted or {}).get("claims", []))
    unsupported_drafted = sum(not c["verified"] for c in result["checks"])
    answerable = case["expected_case_type"] not in ("out_of_scope", "not_covered")
    return {
        "case_type_correct": result["case_type"] == case["expected_case_type"],
        "calculator_called": bool(result["timelines"]) if case["expect_items"] else None,
        "deadlines_correct": all(hits) if case["expect_items"] else None,
        "deadline_items_matched": sum(hits),
        "handoff_correct": bool(result["handoff"]) == case["expect_handoff"],
        "reply_released": (not result["draft_withheld"]) if answerable else None,
        "claims_shown": len(result["claims"]),
        "claims_drafted": drafted,
        "unsupported_drafted": unsupported_drafted,
    }


def estimate_copilot(repeats: int = 1) -> dict:
    from reglens.copilot import MAX_STEPS

    s = ModelSettings.from_env()
    n = len(load_copilot_cases()) * repeats
    tok_in, tok_out = n * MAX_STEPS * 12_000, n * MAX_STEPS * 4_000
    usd = (tok_in * s.price_input_per_m + tok_out * s.price_output_per_m) / 1e6 if s.price_input_per_m else None
    return {
        "model": s.chat_deployment,
        "cases": n,
        "model_calls_max": n * MAX_STEPS,
        "usd_upper_bound": round(usd, 2) if usd is not None else None,
    }


def eval_copilot(repeats: int = 1, retriever_name: str = "dense_bge") -> Path:
    from datetime import date

    from reglens.copilot import build_copilot, chat_model
    from reglens.kg import build_graph

    s = ModelSettings.from_env()
    ledger = budget_from_env(s)
    chunks = load_chunks(CHUNKS)
    corpus, graph, retriever = {c.id: c for c in chunks}, build_graph(chunks), make_retriever(retriever_name, chunks)
    llm, prices = chat_model(s), (s.price_input_per_m, s.price_output_per_m, s.price_cache_read_per_m)
    rows, stopped = [], None
    for rep in range(repeats):
        for case in load_copilot_cases():
            app = build_copilot(
                retriever, graph, corpus, llm, today=date.fromisoformat(case["today"]), ledger=ledger, prices=prices
            )
            t0 = time.perf_counter()
            state = app.invoke({"case": case["case"]})
            r = state["result"]
            if any(e.startswith("budget_exceeded") for e in r["events"]):
                stopped = "budget"
                break
            rows.append(
                {
                    "id": case["id"],
                    "repeat": rep,
                    "case_type": r["case_type"],
                    **score_copilot_case(case, r, state.get("submitted")),
                    "tools": [st["tool"] for st in state["steps"] if st["node"] == "tool"],
                    "events": r["events"],
                    "latency_s": round(time.perf_counter() - t0, 2),
                    "usage": state["usage"],
                    "reply": r["reply_to_member"][:600],
                }
            )
            print(f"[copilot] {case['id']} type={r['case_type']} deadlines={rows[-1]['deadlines_correct']}", flush=True)
        if stopped:
            break

    def rate(key):
        vals = [float(r[key]) for r in rows if r[key] is not None]
        return _mean_ci(vals) if vals else None

    drafted = sum(r["claims_drafted"] for r in rows)
    summary = {
        "stopped": stopped,
        "cases": len(rows),
        "case_type_accuracy": rate("case_type_correct"),
        "calculator_called": rate("calculator_called"),
        "deadlines_correct": rate("deadlines_correct"),
        "handoff_correct": rate("handoff_correct"),
        "reply_released": rate("reply_released"),
        "unsupported_claim_rate_drafted": round(sum(r["unsupported_drafted"] for r in rows) / drafted, 3)
        if drafted
        else None,
        "unsupported_claims_shown": 0,  # finalize drops every claim that fails verification
        "cost_usd_per_case": round(statistics.fmean(r["usage"]["usd"] for r in rows), 5) if rows else None,
        "latency_s_median": statistics.median(r["latency_s"] for r in rows) if rows else None,
        "spend_ledger": dict(ledger.state),
        "provenance": provenance(
            {
                "kind": "copilot",
                "repeats": repeats,
                "provider": s.provider,
                "chat_deployment": s.chat_deployment,
                "retriever": retriever_name,
            }
        ),
    }
    fmt = lambda m: f"{m['mean']:.2f} ({m['ci95'][0]:.2f}-{m['ci95'][1]:.2f})" if m else "n/a"  # noqa: E731
    report = "\n".join(
        [
            "# Copilot scenario evaluation",
            "",
            f"{len(rows)} case runs, model `{s.chat_deployment}`.",
            "",
            "| Metric | Value |",
            "|---|---|",
            *(
                f"| {k.replace('_', ' ')} | {fmt(summary[k])} |"
                for k in (
                    "case_type_accuracy",
                    "calculator_called",
                    "deadlines_correct",
                    "handoff_correct",
                    "reply_released",
                )
            ),
            f"| unsupported claims drafted | {summary['unsupported_claim_rate_drafted']} |",
            "| unsupported claims shown | 0 (by construction) |",
            f"| cost per case (USD) | {summary['cost_usd_per_case']} |",
            "",
        ]
    )
    out = _write("copilot", rows, summary, report)
    print(report)
    return out
