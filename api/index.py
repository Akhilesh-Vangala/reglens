"""Vercel entry point: the same FastAPI app as `reglens-serve`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from reglens.server.app import app  # noqa: E402,F401
