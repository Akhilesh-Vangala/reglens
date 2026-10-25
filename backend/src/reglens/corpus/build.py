"""Fetch (optional), parse and chunk the corpus into data/processed/chunks.jsonl."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from reglens.corpus.chunk import chunk_passages
from reglens.corpus.fetch import fetch
from reglens.corpus.parse import parse_part

ROOT = Path(__file__).resolve().parents[4]


def build(as_of: str, refetch: bool) -> Path:
    raw = ROOT / "data" / "raw"
    manifest_path = raw / "manifest.json"
    manifest = fetch(raw, as_of) if refetch or not manifest_path.exists() else json.loads(manifest_path.read_text())
    passages, warnings = [], []
    for part, meta in manifest["parts"].items():
        secs = set(meta["sections"]) if meta.get("sections") else None
        passages += parse_part((raw / meta["file"]).read_bytes(), part, secs, warnings)
    chunks = chunk_passages(passages)
    out = ROOT / "data" / "processed" / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict()) + "\n")
    stats = {
        "as_of": manifest["as_of"],
        "passages": len(passages),
        "chunks": len(chunks),
        "by_part": dict(Counter(c.part for c in chunks)),
        "by_kind": dict(Counter(c.kind for c in chunks)),
        "parse_warnings": warnings,
    }
    (out.parent / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return out
