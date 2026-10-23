"""reglens CLI.

  reglens build [--refetch]                       fetch, parse and chunk the pinned corpus
  reglens index                                   embed chunks and create/upload the Azure AI Search index
  reglens ask "question" [--retriever hybrid]     answer one question through the governed pipeline
  reglens eval retrieval --retrievers bm25 hybrid
  reglens eval answers --configs A_closed_book B_rag_no_verify C_rag_verified
"""

from __future__ import annotations

import argparse
import json

from reglens.config import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(prog="reglens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--as-of", default="2026-10-02")
    b.add_argument("--refetch", action="store_true")
    sub.add_parser("index")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--retriever", default="hybrid")
    e = sub.add_parser("eval")
    e.add_argument("kind", choices=["retrieval", "answers"])
    e.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid"])
    e.add_argument("--configs", nargs="+", default=["A_closed_book", "B_rag_no_verify", "C_rag_verified"])
    e.add_argument("--retriever", default="hybrid")
    e.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    if args.cmd == "build":
        from reglens.corpus.build import build
        build(args.as_of, args.refetch)
    elif args.cmd == "index":
        from reglens.config import AzureOpenAISettings, AzureSearchSettings
        from reglens.retrieval import Dense, load_chunks
        from reglens.retrieval.azure import AzureEmbedder, AzureHybrid
        chunks = load_chunks(ROOT / "data" / "processed" / "chunks.jsonl")
        emb = AzureEmbedder(AzureOpenAISettings.from_env())
        dense = Dense(chunks, emb, ROOT / "data" / "processed" / f"embeddings_{emb.deployment}.npz")
        az = AzureHybrid(AzureSearchSettings.from_env(), emb, chunks)
        az.create_index(dense.matrix.shape[1])
        print(json.dumps({"uploaded": az.upload(chunks, dense.matrix), "dims": int(dense.matrix.shape[1]),
                          "embedding_tokens": emb.tokens_used}))
    elif args.cmd == "ask":
        from reglens.config import AzureOpenAISettings
        from reglens.eval.run import make_retriever
        from reglens.generation.generator import AzureGenerator
        from reglens.graph import build_pipeline
        from reglens.retrieval import load_chunks
        from reglens.telemetry import shutdown, tracer
        tracer(ROOT / "traces" / "spans.jsonl")
        chunks = load_chunks(ROOT / "data" / "processed" / "chunks.jsonl")
        corpus = {c.id: c for c in chunks}
        app = build_pipeline(make_retriever(args.retriever, chunks), AzureGenerator(AzureOpenAISettings.from_env()), corpus)
        state = app.invoke({"question": args.question})
        final = state["final"]
        for c in final["claims"]:
            c["citation"] = corpus[c["chunk_id"]].citation
        print(json.dumps({"final": final, "events": state["events"], "usage": state["usage"]}, indent=2))
        shutdown()
    elif args.cmd == "eval":
        from reglens.eval.run import eval_answers, eval_retrieval
        if args.kind == "retrieval":
            eval_retrieval(args.retrievers)
        else:
            eval_answers(args.configs, args.retriever, args.repeats)


if __name__ == "__main__":
    main()
