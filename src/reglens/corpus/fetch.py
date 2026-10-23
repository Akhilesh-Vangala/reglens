"""Download pinned regulation text from the official eCFR versioner API.

Every file is stored with its SHA-256 in a manifest, so an evaluation run can state
exactly which version of the law it was measured against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

ECFR = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-12.xml"

# part -> sections to keep (None = whole part). Scoped to member-service topics.
CORPUS: dict[str, dict] = {
    "1005": {"name": "Regulation E (Electronic Fund Transfers)", "sections": None},
    "707": {"name": "NCUA Part 707 (Truth in Savings for credit unions)", "sections": None},
    "229": {"name": "Regulation CC (Availability of Funds and Collection of Checks)", "sections": None},
    "1026": {"name": "Regulation Z (Truth in Lending), credit card liability and billing errors",
             "sections": ["1026.12", "1026.13"]},
}


def fetch(raw_dir: Path, as_of: str) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "eCFR versioner API", "as_of": as_of, "parts": {}}
    with httpx.Client(timeout=60, headers={"Accept-Encoding": "gzip"}) as client:
        for part, meta in CORPUS.items():
            resp = client.get(ECFR.format(date=as_of), params={"part": part})
            resp.raise_for_status()
            path = raw_dir / f"title12_part{part}_{as_of}.xml"
            path.write_bytes(resp.content)
            manifest["parts"][part] = {
                **meta,
                "file": path.name,
                "sha256": hashlib.sha256(resp.content).hexdigest(),
                "bytes": len(resp.content),
                "url": str(resp.url),
            }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
