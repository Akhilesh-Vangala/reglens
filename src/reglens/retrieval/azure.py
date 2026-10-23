"""Azure OpenAI embeddings and Azure AI Search hybrid retrieval.

Azure AI Search runs BM25 and HNSW vector search in one request and fuses them with RRF
inside the service, so `AzureHybrid` is the production counterpart of `LocalHybrid`.
"""

from __future__ import annotations

import time

import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI, OpenAI

from reglens.config import AzureSearchSettings, ModelSettings
from reglens.corpus.chunk import Chunk
from reglens.retrieval import Hit


class ModelEmbedder:
    def __init__(self, s: ModelSettings, batch: int = 64):
        if s.provider == "azure":
            self.client = AzureOpenAI(
                azure_endpoint=s.endpoint, api_key=s.api_key, api_version=s.api_version, max_retries=5
            )
        else:
            self.client = OpenAI(api_key=s.api_key, base_url=s.endpoint or None, max_retries=5)
        self.deployment, self.batch = s.embedding_deployment, batch
        self.name = f"{s.provider}:{s.embedding_deployment}"
        self.tokens_used = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=self.deployment, input=texts[i : i + self.batch])
            self.tokens_used += resp.usage.total_tokens
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return np.asarray(out, dtype=np.float32)


class AzureHybrid:
    """Hybrid query against an Azure AI Search index built from the same chunks."""

    name = "azure_hybrid"

    def __init__(self, s: AzureSearchSettings, embedder: ModelEmbedder, chunks: list[Chunk]):
        cred = AzureKeyCredential(s.api_key)
        self.index_client = SearchIndexClient(s.endpoint, cred)
        self.client = SearchClient(s.endpoint, s.index, cred)
        self.index, self.embedder = s.index, embedder
        self.by_id = {c.id: c for c in chunks}

    def create_index(self, dims: int) -> None:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="part", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="kind", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="citation", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dims,
                vector_search_profile_name="hnsw",
            ),
        ]
        vs = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
            profiles=[VectorSearchProfile(name="hnsw", algorithm_configuration_name="hnsw-algo")],
        )
        self.index_client.create_or_update_index(SearchIndex(name=self.index, fields=fields, vector_search=vs))

    def upload(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 500) -> int:
        docs = [
            {
                "id": c.id,
                "part": c.part,
                "kind": c.kind,
                "citation": c.citation,
                "content": c.index_text(),
                "embedding": v.tolist(),
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        ok = 0
        for i in range(0, len(docs), batch):
            ok += sum(r.succeeded for r in self.client.upload_documents(docs[i : i + batch]))
            time.sleep(0.2)
        return ok

    def search(self, query: str, k: int = 8) -> list[Hit]:
        qv = self.embedder.embed([query])[0]
        results = self.client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=qv.tolist(), k_nearest_neighbors=50, fields="embedding")],
            select=["id"],
            top=k,
        )
        hits = []
        for rank, r in enumerate(results, 1):
            hits.append(Hit(self.by_id[r["id"]], float(r["@search.score"]), {"azure_hybrid": rank}))
        return hits
