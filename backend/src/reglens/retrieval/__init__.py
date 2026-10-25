"""Retrieval primitives behind one `Retriever` interface: BM25 (implemented here), dense cosine search over any
`Embedder`, and reciprocal rank fusion. Production retrievers built on these live in local.py (bge, cross-encoder),
graph.py (citation-graph expansion) and qdrant.py (vector database); eval/run.py:make_retriever names them all."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from reglens.corpus.chunk import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "for",
        "on",
        "or",
        "and",
        "is",
        "are",
        "be",
        "by",
        "with",
        "as",
        "at",
        "that",
        "this",
        "which",
        "any",
        "from",
        "its",
        "it",
        "may",
        "shall",
        "not",
        "if",
        "under",
        "such",
        "than",
        "other",
        "must",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # per-retriever rank, 1-based


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 8) -> list[Hit]: ...


class BM25:
    """Okapi BM25 over chunk.index_text()."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.index_text())) for c in chunks]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = float(self.lens.mean())
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = d.get(t)
                if tf:
                    s += (
                        self.idf[t]
                        * tf
                        * (self.k1 + 1)
                        / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
                    )
            out[i] = s
        return out

    def search(self, query: str, k: int = 8) -> list[Hit]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i]), {"bm25": r + 1}) for r, i in enumerate(top) if s[i] > 0]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class Dense:
    """Cosine similarity over L2-normalized embeddings, cached to disk per embedder."""

    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache: Path | None = None):
        self.chunks, self.embedder = chunks, embedder
        ids = [c.id for c in chunks]
        if cache and cache.exists():
            data = np.load(cache, allow_pickle=False)
            if list(data["ids"]) == ids:
                self.matrix = data["vectors"]
                return
        vecs = embedder.embed([c.index_text() for c in chunks])
        self.matrix = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, ids=np.array(ids), vectors=self.matrix)

    def search(self, query: str, k: int = 8) -> list[Hit]:
        embed_query = getattr(self.embedder, "embed_query", None)  # asymmetric models prefix queries
        q = embed_query(query) if embed_query else self.embedder.embed([query])[0]
        s = self.matrix @ (q / np.linalg.norm(q))
        top = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i]), {"dense": r + 1}) for r, i in enumerate(top)]


def rrf(result_lists: list[list[Hit]], k: int = 8, c: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score = sum 1 / (c + rank). c=60 as in Cormack et al. (2009)."""
    fused: dict[str, Hit] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits, 1):
            cur = fused.setdefault(h.chunk.id, Hit(h.chunk, 0.0, {}))
            cur.score += 1.0 / (c + rank)
            cur.ranks.update(h.ranks)
    return sorted(fused.values(), key=lambda h: -h.score)[:k]


class LocalHybrid:
    name = "hybrid"

    def __init__(self, bm25: BM25, dense: Dense, depth: int = 50):
        self.bm25, self.dense, self.depth = bm25, dense, depth

    def search(self, query: str, k: int = 8) -> list[Hit]:
        return rrf([self.bm25.search(query, self.depth), self.dense.search(query, self.depth)], k=k)
