"""Scoring. Everything here is deterministic: regexes, citation prefixes and the verifier. No LLM judge."""

from __future__ import annotations

import random
import re
import statistics

from reglens.corpus.chunk import Chunk
from reglens.verify import check_claims


def cites_gold(citation: str | None, gold: list[str]) -> bool:
    return bool(citation) and any(citation.startswith(g) for g in gold)


def retrieval_metrics(retrieved_citations: list[str], gold: list[str], ks=(1, 3, 5, 8)) -> dict:
    first = next((i for i, c in enumerate(retrieved_citations, 1) if cites_gold(c, gold)), None)
    out = {f"recall@{k}": float(first is not None and first <= k) for k in ks}
    out["mrr"] = 1.0 / first if first else 0.0
    return out


_CITE = re.compile(r"(?:12\s*CFR\s*)?(\d{3,4}\.\d+[a-z]?)((?:\([a-zA-Z0-9]+\))*)")


def normalize_citation(text: str) -> str | None:
    m = _CITE.search(text or "")
    return f"12 CFR {m.group(1)}{m.group(2)}" if m else None


def score_rag(
    item: dict, final: dict, draft: dict | None, retrieved: dict[str, Chunk], corpus: dict[str, Chunk]
) -> dict:
    """final = what the member sees; draft = model output before the verifier (same as final when off)."""
    status = final.get("status")
    out = {"status_correct": status == item["expected_status"]}
    if item.get("expect_review") is not None:
        out["review_correct"] = bool(final.get("needs_human_review")) == item["expect_review"]
    shown = check_claims(final.get("claims", []), retrieved, corpus)
    drafted = check_claims((draft or {}).get("claims", []), retrieved, corpus)
    out.update(
        n_claims_shown=len(shown),
        n_unsupported_shown=sum(not c.verified for c in shown),
        n_claims_drafted=len(drafted),
        n_unsupported_drafted=sum(not c.verified for c in drafted),
        unsupported_reasons=[c.reason for c in drafted if not c.verified],
    )
    if item["expected_status"] == "answered":
        facts = all(re.search(f, final.get("answer", ""), re.IGNORECASE) for f in item["must_include"])
        grounded = any(c.verified and cites_gold(c.citation, item["gold"]) for c in shown)
        out.update(
            facts_present=facts, grounded_in_gold=grounded, answer_correct=status == "answered" and facts and grounded
        )
    return out


def score_closed_book(item: dict, final: dict, corpus_citations: set[str]) -> dict:
    status = final.get("status")
    out = {"status_correct": status == item["expected_status"]}
    if item.get("expect_review") is not None:
        out["review_correct"] = bool(final.get("needs_human_review")) == item["expect_review"]
    cites = [normalize_citation(c.get("citation", "")) for c in final.get("claims", [])]
    # a citation is fabricated when no paragraph in the pinned corpus starts with it
    fabricated = [c for c in cites if c is None or not any(k.startswith(c) for k in corpus_citations)]
    out.update(n_claims_shown=len(cites), n_fabricated_citations=len(fabricated))
    if item["expected_status"] == "answered":
        facts = all(re.search(f, final.get("answer", ""), re.IGNORECASE) for f in item["must_include"])
        out.update(
            facts_present=facts,
            answer_correct=status == "answered" and facts,
            cites_gold=any(
                c
                and any(
                    c.startswith(g.removeprefix("Supplement I to Part ").split(",")[0]) or g.startswith(c)
                    for g in item["gold"]
                )
                for c in cites
            ),
        )
    return out


def bootstrap(values: list[float], n: int = 4000, seed: int = 7) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return statistics.fmean(values), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def paired_delta(a: dict[str, float], b: dict[str, float], n: int = 4000, seed: int = 7) -> dict:
    shared = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in shared]
    mean, lo, hi = bootstrap(d, n, seed)
    return {
        "n": len(shared),
        "delta": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "significant": not (lo <= 0 <= hi),
        "improved": [k for k in shared if b[k] > a[k]],
        "regressed": [k for k in shared if b[k] < a[k]],
    }
