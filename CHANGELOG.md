# Changelog

Every measured number in this project is tied to the commit that produced it (see `evals/reports/*/summary.json`),
so history is kept as written rather than rewritten. This file is the organized view.

## 1.1.0 — 2026-10-25

**Web console**
- Every view is a link: `#copilot?run=C01` replays a recorded run, `#checker?chunk=…&quote=…` re-runs a check,
  `#explore?node=…&q=…&ret=…` reopens a search and graph neighborhood, `#deadlines?preset=…` a scenario.
- Verified citations link to their paragraph in the citation graph and back into the checker.
- Copilot runs and replays can be stopped mid-run (AbortController), with an elapsed timer.
- WAI-ARIA tabs (arrows, Home, End, roving focus), skip link, keyboard-walkable graph, live regions,
  loading skeletons, error states with retry, copy-link and copy-reply toasts.
- Light and dark themes (system default, persisted choice applied before first paint, CSP unchanged).
- Playwright suite against the real FastAPI app, with an axe WCAG 2 A/AA scan of every tab in both themes and a
  mobile overflow check; runs in CI.

## 1.0.0 — 2026-10-25

First evaluated release: the copilot and pipeline measured with a real model (Claude Haiku 5.5).

**Agent and API**
- Member-service copilot: LangGraph tool-calling agent (search, citation graph, deadline calculator, structured
  submit) with strict JSON-schema tools; Anthropic prompt caching on every agent turn.
- Deterministic finalize: verbatim-quote verification against passages a tool returned, dates and dollar amounts
  only from the calculator, a verified quote or the member's own message (sums of member amounts allowed),
  mandatory handoff for disputes, withheld replies for anything unsupported.
- Server-sent-event stream of agent steps; recorded real runs replayed in the demo.
- Copilot v3: `not_covered` (benign, redirect) separated from `out_of_scope` (refuse); prompt caching on every turn
  (64–74% of input from cache) with cache-aware, never-under-counting cost accounting.

**Evaluation (Claude Haiku 5.5; $0.47 for everything)**
- Copilot scenarios, 30 case runs: deadlines and liability exactly right 1.00, case type 0.97, handoff 0.93,
  replies released 0.93, unsupported claims shown 0, $0.0033 per case.
- Answer evaluation, 43 questions: right fact closed book 0.82 / RAG + verifier 0.88 / copilot 0.94; answers backed
  by a verified quote from the gold paragraph 0 / 0.88 / 0.97; every out-of-scope question declined; no unsupported
  claim shown. Copilot vs closed book on right fact is not significant (+0.12, CI [0.00, +0.27]).
- Like-for-like metrics and `reglens eval rescore` (recompute a report from stored rows, no model calls).

**Fixes found by real-model runs**
- Tools were exposed under Pydantic class names; nested arguments arrived stringified; submissions failed
  validation. Fixed with named strict tools and argument decoding.
- Two verifier false positives (a total of member-given amounts; a member's date written without a year).
- Quotes found only in a chunk's parent lead-in are attributed to the parent paragraph.
- Tests could write scripted usage into the real spend ledger; tests are now isolated from credentials and state.

## 0.4.0 — Production hardening

- Request ids, JSON access logs that never contain member text, strict CSP without inline script, JSON errors.
- Retriever allowlist and query limits; per-IP rate limit and daily USD cap in Redis that fail closed; public spend
  reservations settle to measured cost even when a run fails.
- Law-change monitor (`reglens check-law`) and a weekly workflow against eCFR amendment history.
- Demo rebuilt: live pipeline rail, case timeline, quote-divergence view, radial citation-graph explorer.
- Removed code paths that were never exercised (Azure AI Search, API-embedding retrievers).

## 0.3.0 — Copilot, stores and Kubernetes

- Reg E / Reg Z deadline and liability calculator on a Federal Reserve business-day calendar; every rule quote
  tested verbatim against the pinned regulation.
- Citation graph (2,276 edges parsed from the text) with a Neo4j backend; Qdrant vector store with payload filters.
  Both held to parity with in-memory references (43/43 identical top-8; 0 of 2,371 neighbor mismatches).
- Graph-expanded retrieval measured and rejected as a ranking signal (recall@5 −0.06, CI includes 0).
- Kubernetes manifests (StatefulSets, loader Job, hardened Deployment, HPA, PDB, NetworkPolicy) with a smoke test in
  CI on kind; docker compose stack.

## 0.2.0 — Service and deployment

- FastAPI service and demo; PII masking before retrieval and the model; spend ledger with worst-case admission.
- Open-weight bge-small dense retrieval chosen by ablation: recall@5 0.76 vs 0.46 for RRF hybrid and 0.24 for BM25.
- Provider-agnostic chat client (Anthropic, Azure OpenAI, OpenAI); Docker image; Vercel deployment.

## 0.1.0 — Governed RAG baseline

- Pinned eCFR corpus with SHA-256 manifest; six-level CFR paragraph parser with Official Interpretations;
  citation-addressed chunks.
- BM25, dense and RRF retrieval; verbatim-quote verifier; LangGraph pipeline with one verifier-guided retry.
- 43-question golden set validated against the regulation text; bootstrap CIs and paired deltas; no LLM judge.
