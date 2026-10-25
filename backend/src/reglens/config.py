"""Settings from environment (.env supported). Nothing here has a default price or model:
cost figures are computed only from prices the operator supplies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.getenv("REGLENS_ROOT") or Path(__file__).resolve().parents[3])  # override when pip-installed
if not os.getenv("REGLENS_NO_DOTENV"):  # tests set this so a real key in .env can never be used by a test
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
    """Chat model settings. LLM_PROVIDER=anthropic (Claude API), azure (Azure OpenAI deployment) or openai (OpenAI
    API or any OpenAI-compatible base URL). Prompts, schemas, tools and verification are identical for all three;
    only the LangChain client differs. Embeddings are always the local bge model, so no provider needs an
    embeddings endpoint."""

    provider: str
    endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str
    # USD per 1M tokens from the provider's pricing page; drive the spend caps and cost reporting
    price_input_per_m: float | None
    price_output_per_m: float | None
    price_cache_read_per_m: float | None = None  # Anthropic cache reads (0.1x input); unset = billed at input price

    @classmethod
    def from_env(cls) -> ModelSettings:
        f = lambda k: float(os.environ[k]) if os.getenv(k) else None
        prices = dict(
            price_input_per_m=f("PRICE_INPUT_PER_M"),
            price_output_per_m=f("PRICE_OUTPUT_PER_M"),
            price_cache_read_per_m=f("PRICE_CACHE_READ_PER_M"),
        )
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        if provider == "openai":
            return cls(
                provider="openai",
                endpoint=os.getenv("OPENAI_BASE_URL", ""),
                api_key=_req("OPENAI_API_KEY"),
                api_version="",
                chat_deployment=_req("OPENAI_CHAT_MODEL"),
                **prices,
            )
        if provider == "anthropic":
            return cls(
                provider="anthropic",
                endpoint="",
                api_key=_req("ANTHROPIC_API_KEY"),
                api_version="",
                chat_deployment=_req("ANTHROPIC_CHAT_MODEL"),
                **prices,
            )
        if provider != "azure":
            raise RuntimeError(f"LLM_PROVIDER must be azure, openai or anthropic, got {provider!r}")
        return cls(
            provider="azure",
            endpoint=_req("AZURE_OPENAI_ENDPOINT"),
            api_key=_req("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            chat_deployment=_req("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            **prices,
        )
