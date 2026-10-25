"""Is the pinned corpus still the law? Compares eCFR amendment history with the snapshot date.

For every part in data/raw/manifest.json, the eCFR versioner lists each amendment by section (and Supplement I).
Any in-scope amendment dated after the snapshot means answers may cite superseded text: rebuild the corpus, re-run
the golden tests and evaluations, and redeploy. Runs weekly in CI (.github/workflows/law-watch.yml).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx

VERSIONS = "https://www.ecfr.gov/api/versioner/v1/versions/title-12.json"


def _fetch(part: str) -> list[dict]:
    r = httpx.get(VERSIONS, params={"part": part}, headers={"Accept-Encoding": "gzip"}, timeout=60)
    r.raise_for_status()
    return r.json()["content_versions"]


def in_scope(version: dict, sections: list[str] | None) -> bool:
    """Whole part, or only the listed sections (plus the part's Official Interpretations, which explain them)."""
    if sections is None:
        return True
    return version["identifier"] in sections or version["identifier"].startswith("Supplement I")


def check(manifest_path: Path, fetch: Callable[[str], list[dict]] = _fetch) -> dict:
    manifest = json.loads(manifest_path.read_text())
    as_of = manifest["as_of"]
    parts = {}
    for part, meta in manifest["parts"].items():
        versions = [v for v in fetch(part) if in_scope(v, meta["sections"])]
        newer = sorted((v for v in versions if v["amendment_date"] > as_of), key=lambda v: v["amendment_date"])
        parts[part] = {
            "latest_amendment": max((v["amendment_date"] for v in versions), default=None),
            "amended_since_snapshot": [
                {"identifier": v["identifier"], "amendment_date": v["amendment_date"], "substantive": v["substantive"]}
                for v in newer
            ],
        }
    return {"as_of": as_of, "current": not any(p["amended_since_snapshot"] for p in parts.values()), "parts": parts}
