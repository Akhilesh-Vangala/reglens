"""Settings from environment (.env supported). Nothing here has a default price or model:
cost figures are computed only from prices the operator supplies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.getenv("REGLENS_ROOT") or Path(__file__).resolve().parents[2])  # override when pip-installed
load_dotenv(ROOT / ".env")
# Writable state (spend ledger, traces). Serverless filesystems are read-only except /tmp.
STATE_DIR = Path(os.getenv("REGLENS_STATE_DIR") or ("/tmp/reglens" if os.getenv("VERCEL") else ROOT / "traces"))


def _req(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required setting {name}; see .env.example")
    return value


@dataclass(frozen=True)
class ModelSettings:
    """LLM_PROVIDER=azure (Azure OpenAI deployments) or openai (OpenAI API, or any OpenAI-compatible base URL).
    The pipeline, prompts, schema and verifier are identical for both; only the client differs."""

    provider: str
    endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str
    # USD per 1M tokens, operator supplied from the Azure pricing page; used only for reporting
    price_input_per_m: float | None
    price_output_per_m: float | None
    price_embedding_per_m: float | None

    @classmethod
    def from_env(cls) -> ModelSettings:
        f = lambda k: float(os.environ[k]) if os.getenv(k) else None
        prices = dict(
            price_input_per_m=f("PRICE_INPUT_PER_M"),
            price_output_per_m=f("PRICE_OUTPUT_PER_M"),
            price_embedding_per_m=f("PRICE_EMBEDDING_PER_M"),
        )
        provider = os.getenv("LLM_PROVIDER", "azure").lower()
        if provider == "openai":
            return cls(
                provider="openai",
                endpoint=os.getenv("OPENAI_BASE_URL", ""),
                api_key=_req("OPENAI_API_KEY"),
                api_version="",
                chat_deployment=_req("OPENAI_CHAT_MODEL"),
                embedding_deployment=_req("OPENAI_EMBEDDING_MODEL"),
                **prices,
            )
        if provider != "azure":
            raise RuntimeError(f"LLM_PROVIDER must be azure or openai, got {provider!r}")
        return cls(
            provider="azure",
            endpoint=_req("AZURE_OPENAI_ENDPOINT"),
            api_key=_req("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            chat_deployment=_req("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            embedding_deployment=_req("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            **prices,
        )


@dataclass(frozen=True)
class AzureSearchSettings:
    endpoint: str
    api_key: str
    index: str

    @classmethod
    def from_env(cls) -> AzureSearchSettings:
        return cls(
            endpoint=_req("AZURE_SEARCH_ENDPOINT"),
            api_key=_req("AZURE_SEARCH_API_KEY"),
            index=os.getenv("AZURE_SEARCH_INDEX", "reglens-2026-10-02"),
        )
