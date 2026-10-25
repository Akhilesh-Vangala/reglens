"""Citation graph built from the pinned corpus, plus the Qdrant and Neo4j backends held to parity with the
in-memory implementations. Server tests run when QDRANT_URL / NEO4J_URI are set (CI starts both as services)."""

import os
import zlib

import numpy as np
import pytest

from reglens.config import ROOT
from reglens.kg import build_graph
from reglens.retrieval import Dense, load_chunks
from reglens.retrieval.qdrant import QdrantDense

CHUNKS = load_chunks(ROOT / "data" / "processed" / "chunks.jsonl")


@pytest.fixture(scope="module")
def graph():
    return build_graph(CHUNKS)


def test_interpretation_links_to_the_rule_it_explains(graph):
    assert ("INTERPRETS", "in", "Supplement I to Part 1005, comment 6(b)(1)-1") in graph.neighbors(
        "12 CFR 1005.6(b)(1)"
    )


def test_heading_only_paragraph_resolves_to_its_first_child(graph):
    # comment 11(c)-2 explains 1005.11(c), whose heading lives in (c)(1)
    assert graph.neighbors("Supplement I to Part 1005, comment 11(c)-2") == [
        ("INTERPRETS", "out", "12 CFR 1005.11(c)(1)")
    ]


def test_section_cross_reference_is_an_edge(graph):
    assert ("CITES", "out", "12 CFR 1005.6(a)") in graph.neighbors("12 CFR 1005.11(c)(2)(i)")


def test_references_outside_the_corpus_are_counted_not_guessed(graph):
    s = graph.stats()
    assert s["edges"] > 2000 and 0 < s["unresolved_refs"] < 150
    assert graph.unresolved["12 CFR 1026.61"] > 0  # Reg Z prepaid rules are not in this corpus


class HashEmbedder:
    """Deterministic bag of hashed random token vectors: continuous scores, so no ties to break differently."""

    name = "hash64"

    def embed(self, texts):
        def vec(tok):
            return np.random.default_rng(zlib.crc32(tok.encode())).standard_normal(64)

        return np.array([sum((vec(t) for t in x.lower().split()), np.zeros(64)) for x in texts], dtype=np.float32)


def _toy_dense():
    sub = [c for c in CHUNKS if c.part == "1005"][:300]
    return Dense(sub, HashEmbedder())


def test_qdrant_exact_search_matches_in_memory_dense_and_filters_by_payload():
    dense = _toy_dense()
    q = QdrantDense(dense, "t", exact=True)
    for query in ("investigate an error notice", "stolen card liability days"):
        assert [h.chunk.id for h in q.search(query, 8)] == [h.chunk.id for h in dense.search(query, 8)]
    assert all(h.chunk.kind == "rule" for h in q.search("error", 8, kind="rule"))


@pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="needs a Qdrant server")
def test_qdrant_server_hnsw_matches_exact():
    dense = _toy_dense()
    hnsw = QdrantDense(dense, "ci_hnsw", url=os.environ["QDRANT_URL"])
    exact = QdrantDense(dense, "ci_hnsw", url=os.environ["QDRANT_URL"], exact=True)
    assert [h.chunk.id for h in hnsw.search("error notice", 8)] == [h.chunk.id for h in exact.search("error notice", 8)]


@pytest.mark.skipif(not os.getenv("NEO4J_URI"), reason="needs a Neo4j server")
def test_neo4j_neighbors_identical_to_in_memory_for_every_node(graph):
    from reglens.kg.neo4j_store import Neo4jGraph

    n = Neo4jGraph.from_env()
    assert n.load(graph) == {"nodes": graph.stats()["nodes"], "edges": graph.stats()["edges"]}
    assert [c for c in graph.chunks_by_citation if n.neighbors(c) != graph.neighbors(c)] == []
    n.close()
