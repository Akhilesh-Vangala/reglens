"""Regulation citation graph.

Nodes are paragraph citations. Edges come only from the text and the citation scheme itself, never from a model:

  INTERPRETS  Official Interpretation comment -> the rule paragraph it explains
              ("Supplement I to Part 1005, comment 6(b)(1)-1" -> "12 CFR 1005.6(b)(1)")
  CITES       explicit cross-reference in a paragraph's text: "§ 1005.11(c)(2)", "paragraph (b)(1) of this
              section" (resolved against the citing section), "comment 18(b)(2)-2" (resolved within the part)

A reference to a paragraph with no node of its own resolves to the first paragraph under it in document order (the
parser stores a heading-only paragraph such as "(c) Time limits" inside its first child, (c)(1)), else to the nearest
existing ancestor. A reference to text outside the corpus (e.g. Regulation Z sections other than 1026.12-13) is
counted as unresolved, never guessed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from reglens.corpus.chunk import Chunk

SECTION_REF = re.compile(r"§§?\s?(\d{3,4})\.(\d+[a-z]?)((?:\([a-zA-Z0-9]+\))*)")
PARAGRAPH_REF = re.compile(r"paragraphs? ((?:\([a-zA-Z0-9]+\))+) of this section")
COMMENT_REF = re.compile(r"comments? (\d+[a-z]?)((?:\([a-zA-Z0-9]+\))*)-(\d+)")
OWN_COMMENT = re.compile(r"^Supplement I to Part (\d+), comment (\d+[a-z]?)((?:\([a-zA-Z0-9]+\))*)-\d+")
LABELS = re.compile(r"\([a-zA-Z0-9]+\)")


@dataclass
class CitationGraph:
    chunks_by_citation: dict[str, list[Chunk]]
    out: dict[str, set[tuple[str, str]]] = field(default_factory=lambda: defaultdict(set))  # src -> {(type, dst)}
    inc: dict[str, set[tuple[str, str]]] = field(default_factory=lambda: defaultdict(set))  # dst -> {(type, src)}
    unresolved: Counter = field(default_factory=Counter)

    def add(self, src: str, kind: str, dst: str) -> None:
        if src != dst:
            self.out[src].add((kind, dst))
            self.inc[dst].add((kind, src))

    def edges(self) -> list[tuple[str, str, str]]:
        return sorted((s, k, d) for s, es in self.out.items() for k, d in es)

    def neighbors(self, citation: str) -> list[tuple[str, str, str]]:
        """(relation, direction, citation) for every edge touching `citation`, both directions, sorted."""
        n = [(k, "out", d) for k, d in self.out.get(citation, ())]
        n += [(k, "in", s) for k, s in self.inc.get(citation, ())]
        return sorted(n)

    def stats(self) -> dict:
        kinds = Counter(k for _, k, _ in self.edges())
        return {
            "nodes": len(self.chunks_by_citation),
            "edges": sum(kinds.values()),
            "by_type": dict(kinds),
            "unresolved_refs": sum(self.unresolved.values()),
        }


def _resolve(target: str, nodes: set[str], order: list[str]) -> str | None:
    """Exact node, else first descendant in document order, else nearest existing ancestor, else None."""
    if target in nodes:
        return target
    first_child = next((c for c in order if c.startswith(target + "(")), None)
    if first_child:
        return first_child
    while True:
        trimmed = re.sub(r"\([a-zA-Z0-9]+\)$", "", target)
        if trimmed == target:
            return None
        target = trimmed
        if target in nodes:
            return target


def _rule_target_of_comment(citation: str) -> str | None:
    m = OWN_COMMENT.match(citation)
    if not m:
        return None
    part, sec, labels = m.groups()
    return f"12 CFR {part}.{sec}{labels}"


def build_graph(chunks: list[Chunk]) -> CitationGraph:
    by_cit: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_cit[c.citation].append(c)
    nodes, order = set(by_cit), list(by_cit)  # dicts keep insertion (document) order
    g = CitationGraph(dict(by_cit))
    for cit, cs in by_cit.items():
        part = cs[0].part
        section_num = cs[0].section.split(".", 1)[1] if "." in cs[0].section else ""
        if cs[0].kind == "interpretation":
            tgt = _rule_target_of_comment(cit)
            if tgt:
                r = _resolve(tgt, nodes, order)
                if r:
                    g.add(cit, "INTERPRETS", r)
                else:
                    g.unresolved["interprets"] += 1
        text = " ".join(c.text for c in cs)
        refs = [f"12 CFR {p}.{s}{lab}" for p, s, lab in SECTION_REF.findall(text)]
        if cs[0].kind == "rule" and section_num:
            refs += [f"12 CFR {part}.{section_num}{lab}" for lab in PARAGRAPH_REF.findall(text)]
        refs += [f"Supplement I to Part {part}, comment {s}{lab}-{n}" for s, lab, n in COMMENT_REF.findall(text)]
        for ref in refs:
            r = _resolve(ref, nodes, order)
            if r:
                g.add(cit, "CITES", r)
            else:
                g.unresolved[ref.split("(")[0]] += 1
    return g
