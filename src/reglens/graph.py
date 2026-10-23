"""LangGraph pipeline: retrieve -> generate -> verify -> (retry once with feedback | finalize).

Finalize never lets an unverified claim through: unverified claims are removed, and an answer left with
no verified claim becomes "not_in_corpus" with human review requested. The model proposes; deterministic
code decides what reaches the member.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from reglens.corpus.chunk import Chunk
from reglens.generation.generator import Generator
from reglens.generation.prompts import POLICY, PROMPT_VERSION, RAG_TEMPLATE, format_passages
from reglens.generation.schema import Answer
from reglens.retrieval import Retriever
from reglens.telemetry import tracer
from reglens.verify import check_claims, feedback

MAX_RETRIES = 1


class State(TypedDict, total=False):
    question: str
    retrieved: list[str]
    draft: dict | None
    checks: list[dict]
    attempts: int
    usage: dict
    final: dict
    events: list[str]


def build_pipeline(
    retriever: Retriever, generator: Generator, corpus: dict[str, Chunk], k: int = 8, verify: bool = True
):
    t = tracer()

    def retrieve(state: State) -> State:
        with t.start_as_current_span("retrieve") as span:
            hits = retriever.search(state["question"], k)
            span.set_attribute("reglens.retriever", retriever.name)
            span.set_attribute("reglens.k", k)
            span.set_attribute("reglens.retrieved_citations", [h.chunk.citation for h in hits])
        return {
            "retrieved": [h.chunk.id for h in hits],
            "attempts": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "calls": 0},
            "events": [],
        }

    def generate(state: State) -> State:
        passages = [corpus[i] for i in state["retrieved"]]
        prompt = RAG_TEMPLATE.format(policy=POLICY, passages=format_passages(passages), question=state["question"])
        if state.get("checks") and state.get("draft"):
            prompt += "\n\n" + feedback(_checks(state["checks"])) + "\n\nYOUR PREVIOUS ANSWER:\n" + str(state["draft"])
        with t.start_as_current_span("generate") as span:
            g = generator.generate(prompt, Answer)
            span.set_attribute("gen_ai.request.model", g.model)
            span.set_attribute("gen_ai.usage.input_tokens", g.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", g.output_tokens)
            span.set_attribute("reglens.prompt_version", PROMPT_VERSION)
            span.set_attribute("reglens.attempt", state["attempts"] + 1)
            if g.error:
                span.set_attribute("error.type", g.error[:200])
        u = dict(state["usage"])
        u.update(
            input_tokens=u["input_tokens"] + g.input_tokens,
            output_tokens=u["output_tokens"] + g.output_tokens,
            seconds=u["seconds"] + g.seconds,
            calls=u["calls"] + 1,
        )
        events = state["events"] + ([f"generation_error: {g.error}"] if g.error else [])
        return {"draft": g.parsed, "attempts": state["attempts"] + 1, "usage": u, "events": events}

    def check(state: State) -> State:
        draft = state.get("draft") or {}
        retrieved = {i: corpus[i] for i in state["retrieved"]}
        checks = check_claims(draft.get("claims", []), retrieved, corpus)
        with t.start_as_current_span("verify") as span:
            span.set_attribute("reglens.claims", len(checks))
            span.set_attribute("reglens.claims_verified", sum(c.verified for c in checks))
        return {"checks": [c.__dict__ for c in checks]}

    def route(state: State) -> str:
        failed = any(not c["verified"] for c in state["checks"])
        if verify and failed and state["attempts"] <= MAX_RETRIES and state.get("draft") is not None:
            return "generate"
        return "finalize"

    def finalize(state: State) -> State:
        draft = dict(
            state.get("draft")
            or {
                "status": "not_in_corpus",
                "answer": "",
                "claims": [],
                "needs_human_review": True,
                "review_reason": "generation failed",
            }
        )
        events = list(state["events"])
        if verify:
            ok = [c for c, chk in zip(draft["claims"], state["checks"], strict=True) if chk["verified"]]
            dropped = len(draft["claims"]) - len(ok)
            if dropped:
                events.append(f"dropped_unverified_claims: {dropped}")
            draft["claims"] = ok
            if draft["status"] == "answered" and not ok:
                draft.update(
                    status="not_in_corpus",
                    needs_human_review=True,
                    review_reason="no claim could be verified against the retrieved regulation text",
                )
                events.append("downgraded_to_not_in_corpus")
        return {"final": draft, "events": events}

    g = StateGraph(State)
    for name, fn in (("retrieve", retrieve), ("generate", generate), ("verify", check), ("finalize", finalize)):
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route, {"generate": "generate", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def _checks(rows: list[dict]):
    from reglens.verify import ClaimCheck

    return [ClaimCheck(**r) for r in rows]
