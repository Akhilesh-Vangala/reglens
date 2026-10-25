"""Graph-expanded retrieval: dense search finds entry points, the citation graph brings in what a compliance
analyst would open next (the Official Interpretation of a rule paragraph, the rule a comment explains, the
paragraphs it cross-references). Neighbors are ranked by their own similarity to the query, so expansion adds
connected text without trusting the graph blindly."""

from __future__ import annotations

import numpy as np

from reglens.kg import CitationGraph
from reglens.retrieval import Dense, Hit


class GraphExpanded:
    def __init__(self, dense: Dense, graph: CitationGraph, seeds: int = 3, per_seed: int = 1):
        self.dense, self.graph, self.seeds, self.per_seed = dense, graph, seeds, per_seed
        self.index = {c.id: i for i, c in enumerate(dense.chunks)}
        self.name = f"graph_bge_s{seeds}"

    def _scores(self, query: str) -> np.ndarray:
        eq = getattr(self.dense.embedder, "embed_query", None)
        q = eq(query) if eq else self.dense.embedder.embed([query])[0]
        return self.dense.matrix @ (q / np.linalg.norm(q))

    def search(self, query: str, k: int = 8) -> list[Hit]:
        s = self._scores(query)
        order = [int(i) for i in np.argsort(-s)]
        out: list[Hit] = []
        seen: set[str] = set()

        def push(i: int, ranks: dict) -> None:
            c = self.dense.chunks[i]
            if c.id not in seen and len(out) < k:
                seen.add(c.id)
                out.append(Hit(c, float(s[i]), ranks))

        for rank, i in enumerate(order):
            if len(out) >= k:
                break
            push(i, {"dense": rank + 1})
            if rank < self.seeds:
                cit = self.dense.chunks[i].citation
                nbr = [n for _, _, n in self.graph.neighbors(cit)]
                cand = [self.index[c.id] for n in nbr for c in self.graph.chunks_by_citation.get(n, [])]
                for j in sorted(cand, key=lambda j: -s[j])[: self.per_seed]:
                    push(j, {"graph_from": rank + 1})
        return out
