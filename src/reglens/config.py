"""Settings from environment (.env supported). Nothing here has a default price or model:
cost figures are computed only from prices the operator supplies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _req(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required setting {name}; see .env.example")
    return value


@dataclass(frozen=True)
class AzureOpenAISettings:
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
    def from_env(cls) -> AzureOpenAISettings:
        f = lambda k: float(os.environ[k]) if os.getenv(k) else None
        return cls(
            endpoint=_req("AZURE_OPENAI_ENDPOINT"),
            api_key=_req("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            chat_deployment=_req("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            embedding_deployment=_req("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            price_input_per_m=f("PRICE_INPUT_PER_M"),
            price_output_per_m=f("PRICE_OUTPUT_PER_M"),
            price_embedding_per_m=f("PRICE_EMBEDDING_PER_M"),
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
