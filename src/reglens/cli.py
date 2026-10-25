"""reglens CLI.

reglens build [--refetch]                       fetch, parse and chunk the pinned corpus
reglens check-law                               has any in-scope regulation been amended since the snapshot?
reglens ask "question" [--retriever dense_bge]  answer one question through the governed pipeline
reglens eval retrieval --retrievers bm25 dense_bge hybrid_bge dense_bge+rerank
reglens eval answers --configs A_closed_book B_rag_no_verify C_rag_verified D_copilot
reglens eval copilot [--dry-run]                10 member cases: case type, calculator arguments, handoff
"""

from __future__ import annotations

import argparse
import json

from reglens.config import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="reglens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--as-of", default="2026-10-02")
    b.add_argument("--refetch", action="store_true")
    sub.add_parser("load-stores", help="load the citation graph into Neo4j and the vectors into Qdrant")
    sub.add_parser("check-law", help="exit 1 if eCFR shows in-scope amendments after the pinned snapshot")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--retriever", default="dense_bge")
    e = sub.add_parser("eval")
    e.add_argument("kind", choices=["retrieval", "answers", "stores", "copilot"])
    e.add_argument("--retrievers", nargs="+", default=["bm25", "dense_bge", "hybrid_bge", "dense_bge+rerank"])
    e.add_argument("--configs", nargs="+", default=["A_closed_book", "B_rag_no_verify", "C_rag_verified", "D_copilot"])
    e.add_argument("--retriever", default="dense_bge")
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--dry-run", action="store_true", help="answers: print the upper-bound cost, call no model")
    args = ap.parse_args()

    if args.cmd == "build":
        from reglens.corpus.build import build

        build(args.as_of, args.refetch)
    elif args.cmd == "check-law":
        import sys

        from reglens.corpus.freshness import check

        report = check(ROOT / "data" / "raw" / "manifest.json")
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["current"] else 1)
    elif args.cmd == "load-stores":
        from reglens.eval.run import make_retriever
        from reglens.kg import build_graph
        from reglens.kg.neo4j_store import Neo4jGraph
        from reglens.retrieval import load_chunks

        chunks = load_chunks(ROOT / "data" / "processed" / "chunks.jsonl")
        neo = Neo4jGraph.from_env()
        loaded = neo.load(build_graph(chunks))
        neo.close()
        q = make_retriever("qdrant_bge", chunks)
        points = q.client.count(q.collection, exact=True).count
        print(json.dumps({"neo4j": loaded, "qdrant": {"collection": q.collection, "points": points}}))
    elif args.cmd == "ask":
        from reglens.config import ModelSettings
        from reglens.eval.run import make_retriever
        from reglens.generation.generator import ChatGenerator
        from reglens.graph import build_pipeline
        from reglens.retrieval import load_chunks
        from reglens.telemetry import shutdown, tracer

        tracer(ROOT / "traces" / "spans.jsonl")
        chunks = load_chunks(ROOT / "data" / "processed" / "chunks.jsonl")
        corpus = {c.id: c for c in chunks}
        from reglens.eval.run import budget_from_env
        from reglens.generation.generator import Budgeted

        s = ModelSettings.from_env()
        raw = ChatGenerator(s)
        gen = Budgeted(raw, budget_from_env(s), s.price_input_per_m, s.price_output_per_m, raw.max_output_tokens)
        app = build_pipeline(make_retriever(args.retriever, chunks), gen, corpus)
        state = app.invoke({"question": args.question})
        final = state["final"]
        for c in final["claims"]:
            c.setdefault("citation", corpus[c["chunk_id"]].citation)  # unverified runs (verify=False) only
        print(json.dumps({"final": final, "events": state["events"], "usage": state["usage"]}, indent=2))
        shutdown()
    elif args.cmd == "eval":
        from reglens.eval.run import eval_answers, eval_retrieval

        if args.kind == "retrieval":
            eval_retrieval(args.retrievers)
        elif args.kind == "copilot":
            from reglens.eval.run import estimate_copilot, eval_copilot

            if args.dry_run:
                print(json.dumps(estimate_copilot(args.repeats), indent=2))
            else:
                eval_copilot(args.repeats)
        elif args.kind == "stores":
            from reglens.eval.run import eval_stores

            eval_stores()
        else:
            if args.dry_run:
                from reglens.eval.run import estimate_answers

                print(json.dumps(estimate_answers(args.configs, args.retriever, args.repeats), indent=2))
            else:
                eval_answers(args.configs, args.retriever, args.repeats)


if __name__ == "__main__":
    main()
