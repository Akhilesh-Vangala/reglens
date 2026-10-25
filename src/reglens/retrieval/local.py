"""Open-weight retrieval components that run on CPU with no API key (ONNX via fastembed).

BGE models are trained with an instruction prefix on queries only; documents are embedded as-is.
The cross-encoder rescores the fused top candidates jointly with the query, which a bi-encoder cannot.
"""

from __future__ import annotations

import os

import numpy as np

from reglens.retrieval import Hit

if os.getenv("VERCEL"):  # model weights download to the only writable path on cold start
    os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch: int = 64):
        from fastembed import TextEmbedding

        self.model, self.batch = TextEmbedding(model), batch
        self.name = self.deployment = model.split("/")[-1]

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts, batch_size=self.batch)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([BGE_QUERY_PREFIX + text])[0]


class Reranked:
    """Wrap any retriever: take its top `depth`, rescore with a cross-encoder, return the top k."""

    def __init__(self, base, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", depth: int = 30):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.base, self.depth, self.encoder = base, depth, TextCrossEncoder(model)
        self.name = f"{base.name}+rerank"

    def search(self, query: str, k: int = 8) -> list[Hit]:
        cands = self.base.search(query, self.depth)
        scores = list(self.encoder.rerank(query, [h.chunk.index_text() for h in cands]))
        order = np.argsort(-np.asarray(scores))[:k]
        return [Hit(cands[i].chunk, float(scores[i]), {**cands[i].ranks, "rerank": r + 1}) for r, i in enumerate(order)]
