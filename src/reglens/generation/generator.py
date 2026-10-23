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
    """LangChain AzureChatOpenAI / ChatOpenAI with strict JSON-schema structured output, temperature 0."""

    def __init__(self, s: ModelSettings, temperature: float = 0.0):
        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        common = dict(temperature=temperature, max_retries=3, timeout=60, api_key=s.api_key)
        if s.provider == "azure":
            self.llm = AzureChatOpenAI(
                azure_endpoint=s.endpoint, api_version=s.api_version, azure_deployment=s.chat_deployment, **common
            )
        else:
            self.llm = ChatOpenAI(model=s.chat_deployment, base_url=s.endpoint or None, **common)
        self.model = f"{s.provider}:{s.chat_deployment}"

    def generate(self, prompt: str, schema: type[BaseModel]) -> Generation:
        runnable = self.llm.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)
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
