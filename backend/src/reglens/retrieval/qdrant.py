"""Qdrant vector database backend for the dense retriever.

Same embeddings and query encoding as `Dense`; Qdrant adds an HNSW index, payload filtering (restrict a search to
one regulation part or to rules vs. interpretations) and a server other services can share. `exact=True` runs a
brute-force scan, which the parity test uses to prove the index returns what the in-memory baseline returns.

QDRANT_URL selects a server (docker compose / Kubernetes); without it the client runs embedded in-process.
"""

from __future__ import annotations

import os

import numpy as np

from reglens.corpus.chunk import Chunk
from reglens.retrieval import Dense, Hit


class QdrantDense:
    def __init__(
        self, dense: Dense, collection: str, url: str | None = None, path: str | None = None, exact: bool = False
    ):
        from qdrant_client import QdrantClient

        self.dense, self.collection, self.exact = dense, collection, exact
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path) if path else QdrantClient(":memory:")
        self.remote = bool(url)
        self.by_id = {c.id: c for c in dense.chunks}
        self.name = "qdrant_bge_exact" if exact else "qdrant_bge"
        self.supports_filters = True
        self._ensure()

    def _ensure(self) -> None:
        from qdrant_client import models as m

        n, dims = self.dense.matrix.shape
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            # reuse only an index built from the same vectors: same count and same dimensionality
            if info.config.params.vectors.size == dims and self.client.count(self.collection, exact=True).count == n:
                return
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=m.VectorParams(size=dims, distance=m.Distance.COSINE),
            hnsw_config=m.HnswConfigDiff(m=16, ef_construct=128),
        )
        if self.remote:  # payload indexes exist only on a server; embedded mode filters by scan
            for field in ("part", "kind"):
                self.client.create_payload_index(self.collection, field, m.PayloadSchemaType.KEYWORD)
        for start in range(0, n, 512):
            chunk_slice = self.dense.chunks[start : start + 512]
            self.client.upsert(
                self.collection,
                points=[
                    m.PointStruct(
                        id=start + i,
                        vector=self.dense.matrix[start + i].tolist(),
                        payload={"chunk_id": c.id, "part": c.part, "kind": c.kind, "citation": c.citation},
                    )
                    for i, c in enumerate(chunk_slice)
                ],
                wait=True,
            )

    def search(self, query: str, k: int = 8, part: str | None = None, kind: str | None = None) -> list[Hit]:
        from qdrant_client import models as m

        eq = getattr(self.dense.embedder, "embed_query", None)
        q = eq(query) if eq else self.dense.embedder.embed([query])[0]
        must = [m.FieldCondition(key=f, match=m.MatchValue(value=v)) for f, v in (("part", part), ("kind", kind)) if v]
        res = self.client.query_points(
            self.collection,
            query=(q / np.linalg.norm(q)).tolist(),
            limit=k,
            query_filter=m.Filter(must=must) if must else None,
            search_params=m.SearchParams(exact=self.exact, hnsw_ef=128) if self.remote else None,  # embedded is exact
        ).points
        return [Hit(self.by_id[p.payload["chunk_id"]], float(p.score), {"qdrant": r + 1}) for r, p in enumerate(res)]


def from_env(dense: Dense, chunks: list[Chunk], exact: bool = False) -> QdrantDense:
    from reglens.config import STATE_DIR

    url = os.getenv("QDRANT_URL")
    return QdrantDense(
        dense,
        os.getenv("QDRANT_COLLECTION", "reglens_bge_2026_10_02"),
        url=url,
        path=None if url else str(STATE_DIR / "qdrant"),
        exact=exact,
    )
