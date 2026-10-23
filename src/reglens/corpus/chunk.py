"""Turn parsed passages into retrievable, verifiable chunks.

A chunk keeps two verbatim pieces of regulation text:
  context  the parent paragraph's lead-in, so "(i) An unauthorized transfer;" is
           retrievable and readable on its own
  text     the passage itself
Quotes are verified against context + text, both of which are the law's own words.
Long passages are split on sentence boundaries with one sentence of overlap; every
piece keeps the passage citation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from reglens.corpus.parse import Passage

MAX_CHARS = 1600
CONTEXT_CHARS = 400
SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    id: str
    part: str
    section: str
    kind: str
    citation: str
    heading: str
    context: str
    text: str

    def source_text(self) -> str:
        """What the model sees and what quotes are verified against."""
        return f"{self.context} {self.text}".strip()

    def index_text(self) -> str:
        """What retrieval indexes: citation and heading improve keyword matching."""
        return f"{self.citation}. {self.heading}. {self.source_text()}"

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(citation: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", citation).strip("_")


def _split(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    sents = SENT.split(text)
    pieces, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s])) > MAX_CHARS:
            pieces.append(" ".join(cur))
            cur = [cur[-1]]  # one-sentence overlap keeps a split claim quotable
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_passages(passages: list[Passage]) -> list[Chunk]:
    by_label = {(p.part, p.section, p.kind, p.label): p for p in passages}
    chunks: list[Chunk] = []
    for p in passages:
        context = ""
        if p.kind == "rule":
            parent_label = re.sub(r"\([^()]+\)$", "", p.label)
            parent = by_label.get((p.part, p.section, "rule", parent_label)) if parent_label != p.label else None
            if parent:
                context = parent.text[:CONTEXT_CHARS]
        pieces = _split(p.text)
        for i, piece in enumerate(pieces, 1):
            cid = _slug(p.citation) + (f"__{i}" if len(pieces) > 1 else "")
            chunks.append(
                Chunk(
                    id=cid,
                    part=p.part,
                    section=p.section,
                    kind=p.kind,
                    citation=p.citation,
                    heading=p.heading,
                    context=context,
                    text=piece,
                )
            )
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    return chunks
