"""Record real copilot runs for the demo's replay panel: python backend/scripts/record_copilot_runs.py C01 C04 C06 C09

Each run is the exact event sequence /api/copilot/stream emits (start, one step per agent turn or tool call,
result), with the model, date and git commit, written to data/demo/copilot_runs.jsonl. Spend goes through the same
capped ledger as every other model run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date

from reglens.config import ROOT, ModelSettings
from reglens.copilot import MAX_STEPS, build_copilot, chat_model
from reglens.eval.run import budget_from_env, load_copilot_cases, make_retriever
from reglens.kg import build_graph
from reglens.retrieval import load_chunks
from reglens.server.app import _span

OUT = ROOT / "data" / "demo" / "copilot_runs.jsonl"


def main(ids: list[str]) -> None:
    s = ModelSettings.from_env()
    ledger = budget_from_env(s)
    chunks = load_chunks(ROOT / "data" / "processed" / "chunks.jsonl")
    corpus, graph, retriever = {c.id: c for c in chunks}, build_graph(chunks), make_retriever("dense_bge", chunks)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout
    cases = {c["id"]: c for c in load_copilot_cases()}
    runs = []
    for cid in ids:
        case = cases[cid]
        app = build_copilot(
            retriever,
            graph,
            corpus,
            chat_model(s),
            today=date.fromisoformat(case["today"]),
            ledger=ledger,
            prices=(s.price_input_per_m, s.price_output_per_m, s.price_cache_read_per_m),
        )
        events, state, emitted = [{"type": "start", "max_steps": MAX_STEPS}], {}, 0
        for update in app.stream({"case": case["case"]}, stream_mode="updates"):
            for delta in update.values():
                state.update(delta or {})
            events += [{"type": "step", "step": st} for st in state.get("steps", [])[emitted:]]
            emitted = len(state.get("steps", []))
        r = state["result"]
        for c in r["claims"]:
            c["span"] = _span(c["source"], c["quote"])
        events.append({"type": "result", "result": r, "usage": state["usage"], "redactions": state["redactions"]})
        runs.append(
            {
                "id": cid,
                "title": case["title"],
                "case": case["case"],
                "model": f"{s.provider}:{s.chat_deployment}",
                "recorded_at": date.today().isoformat(),
                "commit": commit.strip(),
                "events": events,
            }
        )
        print(cid, r["case_type"], "withheld" if r["draft_withheld"] else "released", f"${state['usage']['usd']:.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x) + "\n" for x in runs))


if __name__ == "__main__":
    main(sys.argv[1:] or ["C01", "C04", "C06", "C09"])
