"""Model clients behind one protocol. Azure OpenAI (or OpenAI) via LangChain structured output in production;
scripted fakes in tests. Every call returns the parsed answer plus measured token usage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from reglens.config import ModelSettings


@dataclass
class Generation:
    parsed: dict | None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None
    model: str = ""


class Generator(Protocol):
    model: str

    def generate(self, prompt: str, schema: type[BaseModel]) -> Generation: ...


class ChatGenerator:
    """LangChain AzureChatOpenAI / ChatOpenAI / ChatAnthropic with JSON-schema structured output."""

    def __init__(self, s: ModelSettings, temperature: float = 0.0):
        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        common = dict(max_retries=3, timeout=60, api_key=s.api_key)
        if s.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # The Claude API rejects sampling parameters (temperature/top_p/top_k) since 2026-10-21. Room is left
            # for adaptive thinking, which counts toward max_tokens.
            self.max_output_tokens = 8000
            self.llm = ChatAnthropic(model=s.chat_deployment, max_tokens=self.max_output_tokens, **common)
        elif s.provider == "azure":
            self.max_output_tokens = 4000
            common["temperature"] = temperature
            self.llm = AzureChatOpenAI(
                azure_endpoint=s.endpoint, api_version=s.api_version, azure_deployment=s.chat_deployment, **common
            )
        else:
            self.max_output_tokens = 4000
            common["temperature"] = temperature
            self.llm = ChatOpenAI(model=s.chat_deployment, base_url=s.endpoint or None, **common)
        self.model = f"{s.provider}:{s.chat_deployment}"

    def generate(self, prompt: str, schema: type[BaseModel]) -> Generation:
        # Claude's json_schema mode is constrained decoding already; OpenAI needs strict=True to get the same
        strict = {} if self.model.startswith("anthropic:") else {"strict": True}
        runnable = self.llm.with_structured_output(schema, method="json_schema", include_raw=True, **strict)
        start = time.perf_counter()
        try:
            out = runnable.invoke(prompt)
        except Exception as exc:  # noqa: BLE001  (provider errors are recorded, not raised, in evaluation)
            return Generation(None, seconds=time.perf_counter() - start, error=repr(exc)[:500], model=self.model)
        usage = getattr(out["raw"], "usage_metadata", None) or {}
        parsed = out.get("parsed")
        return Generation(
            parsed=parsed.model_dump() if parsed is not None else None,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            seconds=time.perf_counter() - start,
            error=None if parsed is not None else f"parse_error: {out.get('parsing_error')}"[:500],
            model=self.model,
        )


class Budgeted:
    """Wraps a generator with a hard spend cap. Every call is admitted against its worst case (estimated input
    plus the full output allowance) before it is sent, and billed at measured usage after. Exceeding the cap
    raises BudgetExceeded, which stops an evaluation run instead of quietly spending past the limit."""

    def __init__(self, inner: Generator, ledger, price_in_per_m: float, price_out_per_m: float, max_out: int):
        self.inner, self.ledger, self.p_in, self.p_out, self.max_out = (
            inner,
            ledger,
            price_in_per_m,
            price_out_per_m,
            max_out,
        )
        self.model = inner.model

    def generate(self, prompt: str, schema: type[BaseModel]) -> Generation:
        est_in = len(prompt) // 3  # conservative: ~3 chars/token for legal English
        self.ledger.admit(est_in, (est_in * self.p_in + self.max_out * self.p_out) / 1e6)
        g = self.inner.generate(prompt, schema)
        self.ledger.record(
            g.input_tokens, g.output_tokens, (g.input_tokens * self.p_in + g.output_tokens * self.p_out) / 1e6
        )
        return g


@dataclass
class ScriptedGenerator:
    """Deterministic test double: returns queued answers in order."""

    answers: list[dict]
    model: str = "scripted"
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str, schema: type[BaseModel]) -> Generation:
        self.prompts.append(prompt)
        parsed = schema.model_validate(self.answers.pop(0)).model_dump()
        return Generation(parsed, input_tokens=len(prompt) // 4, output_tokens=50, seconds=0.0, model=self.model)
