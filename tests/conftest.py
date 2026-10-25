"""Isolation for every test: no real model credentials, and a throwaway state directory, so tests can never call
a paid API or write to the real spend ledger (traces/spend.json)."""

import os
import tempfile

os.environ["REGLENS_NO_DOTENV"] = "1"
os.environ["REGLENS_STATE_DIR"] = tempfile.mkdtemp(prefix="reglens-test-")
for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "LLM_PROVIDER", "REGLENS_USD_CAP"):
    os.environ.pop(var, None)
