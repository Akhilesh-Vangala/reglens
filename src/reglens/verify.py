"""Deterministic grounding checks. No model is involved in deciding whether a claim is supported.

A claim is verified only if:
  1. its chunk_id is one of the passages retrieved for THIS question (no citing unseen text), and
  2. its quote, after whitespace/quote-mark normalization, is a substring of that passage's
     verbatim source text (context + text), and
  3. the quote is long enough to carry meaning (>= MIN_QUOTE characters).

Each chunk carries its parent paragraph's lead-in as context ("The term error means: ..."). A quote found only in
that context is attributed to the parent paragraph, so the citation names where the words actually are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reglens.corpus.chunk import Chunk

MIN_QUOTE = 25
_WS = re.compile(r"\s+")
_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "§": "§"})


def normalize(text: str) -> str:
    return _WS.sub(" ", text.translate(_MAP)).strip().lower()


@dataclass
class ClaimCheck:
    chunk_id: str
    citation: str | None
    verified: bool
    reason: str  # ok | unknown_chunk | not_retrieved | quote_too_short | quote_not_in_source
    matched_in: str | None = None  # text | context (parent lead-in)


def check_claims(claims: list[dict], retrieved: dict[str, Chunk], corpus: dict[str, Chunk]) -> list[ClaimCheck]:
    out = []
    for c in claims:
        cid = str(c.get("chunk_id", "")).strip()
        quote = normalize(str(c.get("quote", "")))
        chunk = corpus.get(cid)
        if chunk is None:
            reason = "unknown_chunk"
        elif cid not in retrieved:
            reason = "not_retrieved"
        elif len(quote) < MIN_QUOTE:
            reason = "quote_too_short"
        elif quote not in normalize(chunk.source_text()):
            reason = "quote_not_in_source"
        else:
            reason = "ok"
        citation, where = (chunk.citation if chunk else None), None
        if reason == "ok":
            where = "text" if quote in normalize(chunk.text) else "context"
            if where == "context":
                citation = parent_citation(chunk.citation)
        out.append(ClaimCheck(cid, citation, reason == "ok", reason, where))
    return out


def parent_citation(citation: str) -> str:
    """'12 CFR 1005.6(b)(2)(ii)' -> '12 CFR 1005.6(b)(2)'. Comments and section roots are returned unchanged."""
    trimmed = re.sub(r"\([a-zA-Z0-9]+\)$", "", citation)
    return trimmed if "(" in trimmed else citation


def feedback(checks: list[ClaimCheck]) -> str:
    """Model-facing correction message: says which claims failed and why, never what the answer is."""
    bad = [f"- claim citing {c.chunk_id!r}: {c.reason}" for c in checks if not c.verified]
    return (
        (
            "These claims failed verification:\n"
            + "\n".join(bad)
            + "\nRe-answer using only the passages provided. Copy quotes exactly from the cited passage. "
            "Drop any claim you cannot support with an exact quote."
        )
        if bad
        else ""
    )
