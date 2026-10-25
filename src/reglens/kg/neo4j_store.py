"""Neo4j backend for the citation graph: (:Paragraph {citation, part, kind, heading}) nodes joined by
[:CITES] and [:INTERPRETS] relationships. `neighbors` answers the same question as CitationGraph.neighbors, and the
parity test holds the two to identical results for every node. Cypher also enables questions the in-memory form
doesn't index, e.g. every interpretation within two hops of a rule paragraph.

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD select the server (docker compose / Kubernetes).
"""

from __future__ import annotations

import os

from reglens.kg import CitationGraph

NEIGHBORS = """
MATCH (p:Paragraph {citation: $c})-[r]-(n:Paragraph)
RETURN type(r) AS rel, CASE WHEN startNode(r) = p THEN 'out' ELSE 'in' END AS dir, n.citation AS citation
ORDER BY rel, dir, citation
"""

TWO_HOP_INTERPRETATIONS = """
MATCH (p:Paragraph {citation: $c})-[*1..2]-(n:Paragraph {kind: 'interpretation'})
RETURN DISTINCT n.citation AS citation ORDER BY citation
"""


class Neo4jGraph:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase

        self.driver, self.db = GraphDatabase.driver(uri, auth=(user, password)), database

    @classmethod
    def from_env(cls) -> Neo4jGraph:
        return cls(os.environ["NEO4J_URI"], os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])

    def _q(self, cypher: str, **params):
        return self.driver.execute_query(cypher, params, database_=self.db).records

    def load(self, g: CitationGraph, batch: int = 1000) -> dict:
        self._q("CREATE CONSTRAINT paragraph_citation IF NOT EXISTS FOR (p:Paragraph) REQUIRE p.citation IS UNIQUE")
        self._q("MATCH (n:Paragraph) DETACH DELETE n")
        nodes = [
            {"citation": c, "part": cs[0].part, "kind": cs[0].kind, "heading": cs[0].heading}
            for c, cs in g.chunks_by_citation.items()
        ]
        for i in range(0, len(nodes), batch):
            self._q(
                "UNWIND $rows AS r CREATE (:Paragraph {citation: r.citation, part: r.part, kind: r.kind, "
                "heading: r.heading})",
                rows=nodes[i : i + batch],
            )
        for rel in ("CITES", "INTERPRETS"):  # relationship types can't be parameters in Cypher
            rows = [{"s": s, "d": d} for s, k, d in g.edges() if k == rel]
            for i in range(0, len(rows), batch):
                self._q(
                    f"UNWIND $rows AS r MATCH (a:Paragraph {{citation: r.s}}), (b:Paragraph {{citation: r.d}}) "
                    f"CREATE (a)-[:{rel}]->(b)",
                    rows=rows[i : i + batch],
                )
        n = self._q("MATCH (n:Paragraph) RETURN count(n) AS n")[0]["n"]
        e = self._q("MATCH ()-[r]->() RETURN count(r) AS e")[0]["e"]
        return {"nodes": n, "edges": e}

    def neighbors(self, citation: str) -> list[tuple[str, str, str]]:
        return [(r["rel"], r["dir"], r["citation"]) for r in self._q(NEIGHBORS, c=citation)]

    def interpretations_within_two_hops(self, citation: str) -> list[str]:
        return [r["citation"] for r in self._q(TWO_HOP_INTERPRETATIONS, c=citation)]

    def close(self) -> None:
        self.driver.close()


class GraphService:
    """What the server and copilot use: neighbor queries go to Neo4j when NEO4J_URI is set (Kubernetes /
    docker compose), otherwise to the in-memory graph; chunk lookups always come from the in-memory corpus."""

    def __init__(self, graph: CitationGraph):
        self.graph, self.chunks_by_citation = graph, graph.chunks_by_citation
        self.neo = Neo4jGraph.from_env() if os.getenv("NEO4J_URI") else None
        self.backend = "neo4j" if self.neo else "memory"

    def neighbors(self, citation: str) -> list[tuple[str, str, str]]:
        return self.neo.neighbors(citation) if self.neo else self.graph.neighbors(citation)

    def stats(self) -> dict:
        return {**self.graph.stats(), "backend": self.backend}
