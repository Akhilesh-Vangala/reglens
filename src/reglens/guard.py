"""Pre-model input guard and post-model spend guard.

PII: member questions can contain account, card or Social Security numbers. They are masked before the
text reaches retrieval, the model or traces; only the counts of what was masked are recorded.

Spend: every model call is checked against a per-request token cap and a cumulative USD ledger. A call
that would exceed either is refused before it is sent (AI FinOps: cost is a control, not a report).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Order matters: card numbers before generic account numbers.
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("card_number", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")),
    ("account_number", re.compile(r"\b(?:account|acct|member)\s*(?:number|no\.?|#)?\s*:?\s*\d{5,17}\b", re.I)),
]


@dataclass
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.counts)


def redact(text: str) -> Redaction:
    counts: dict[str, int] = {}
    for name, pat in PII_PATTERNS:
        text, n = pat.subn(f"[{name.upper()} REDACTED]", text)
        if n:
            counts[name] = n
    return Redaction(text, counts)


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    """Thread-safe cumulative spend ledger persisted as JSON. Prices are operator supplied; with no prices
    the ledger still enforces the token cap and records tokens, but cannot enforce a dollar cap."""

    def __init__(self, path: Path, usd_cap: float | None, max_tokens_per_request: int = 20_000):
        self.path, self.usd_cap, self.max_tokens = path, usd_cap, max_tokens_per_request
        self.lock = threading.Lock()
        self.state = (
            json.loads(path.read_text())
            if path.exists()
            else {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        )

    def admit(self, estimated_input_tokens: int, estimated_usd: float | None) -> None:
        if estimated_input_tokens > self.max_tokens:
            raise BudgetExceeded(f"request of ~{estimated_input_tokens} tokens exceeds cap {self.max_tokens}")
        if self.usd_cap is not None and estimated_usd is not None and self.state["usd"] + estimated_usd > self.usd_cap:
            raise BudgetExceeded(f"cumulative spend cap ${self.usd_cap:.2f} would be exceeded")

    def record(self, input_tokens: int, output_tokens: int, usd: float | None) -> None:
        with self.lock:
            self.state["input_tokens"] += input_tokens
            self.state["output_tokens"] += output_tokens
            self.state["calls"] += 1
            if usd is not None:
                self.state["usd"] = round(self.state["usd"] + usd, 6)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=2))
