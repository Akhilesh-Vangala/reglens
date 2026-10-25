# Model card: RegLens member-service copilot

Written for model-catalog registration. Fields marked *measured* are filled only from committed evaluation reports in `evals/reports/`.

## Intended use
- Explain what federal consumer-finance regulations say, with verbatim citations, to support credit-union member-service staff.
- Draft member replies, compute regulatory deadlines and liability caps, and open handoff tickets for a representative to review and send.
- Scope: Regulation E (12 CFR 1005), NCUA Part 707 (Truth in Savings for credit unions), Regulation CC (12 CFR 229), Regulation Z 1026.12 and 1026.13. Point-in-time text from the eCFR as of 2026-10-02 (see `data/raw/manifest.json` for SHA-256 hashes).

## Out of scope
- Decisions on a specific member's case, legal advice, account actions, credit decisions, institution-specific policies or rates, and any regulation outside the corpus.
- Use with member PII. The system never needs account data. As a backstop, SSN, card, account, email and phone patterns are masked before retrieval, the model and traces (`guard.py`); pattern masking is not a guarantee.

## System design
- Retrieval: dense bge-small-en-v1.5 (open weights, CPU; numpy or Qdrant with verified parity), chosen by measurement over BM25, RRF hybrid, cross-encoder reranking and citation-graph expansion (see README).
- Copilot: LangGraph tool-calling agent (search, citation graph in memory or Neo4j, deadline calculator, structured submit); at most 8 model turns.
- Generation: Azure OpenAI or OpenAI chat model, temperature 0, strict JSON-schema structured output, prompt version recorded in every trace.
- Control: a deterministic verifier accepts a claim only if its quote is verbatim in a passage retrieved for that question. One retry with verifier feedback; unverified claims are removed before the member-facing answer; an answer with no verified claim becomes "not in corpus" with human review requested.
- Escalation: the model flags live problems; independently, policy code always opens a handoff ticket for disputes and unauthorized use, and withholds any reply with no verified claim or with a date the calculator did not produce.
- Deadlines: computed by code from Reg E 1005.6/1005.11 and Reg Z 1026.12/1026.13 on a Federal Reserve business-day calendar; every rule quote tested against the corpus.
- Cost: a spend ledger enforces a per-request token cap and a cumulative USD cap before each model call; a refused call escalates to human review.

## Evaluation (*measured*)
- Golden set: 43 author-written questions (33 answerable, 4 not-in-corpus, 6 refusal), including 5 adversarial items: prompt injection, PII-laden, system-prompt extraction, role-play legal advice and a false premise. Every gold fact is checked against the regulation text by `tests/test_golden.py`.
- Metrics: answer accuracy (status + required fact + verified claim citing a gold paragraph), unsupported-claim rate, fabricated-citation rate (closed book), refusal, not-in-corpus and escalation accuracy, latency, tokens, cost. Bootstrap 95% CIs over questions; paired deltas between configurations.
- Copilot: 10 member cases scored on case type, calculator arguments (expected liability and deadlines), handoff and reply release; the answer evaluation adds the copilot as configuration D.
- Results: the latest `evals/reports/*-{retrieval,stores,answers,copilot}/REPORT.md`. Answer and copilot results do not exist until a model run is made; none are claimed.

## Known limitations
- Golden set is small and author-written; no member or compliance-officer labels yet.
- Exact-substring verification proves a quote exists, not that the claim's interpretation is correct; human review remains the control for interpretation.
- Paragraph structure is reconstructed from inline labels; two source anomalies are repaired and logged (`data/processed/corpus_stats.json`).
- The corpus is a point-in-time snapshot. A weekly job (`reglens check-law`) compares eCFR amendment history with the snapshot and fails on any in-scope change; the corpus must then be rebuilt and re-evaluated.
- Business days follow the Federal Reserve calendar; Reg E defines them by when the institution is open.

## Governance hooks
- Versioned prompt (`PROMPT_VERSION`), pinned corpus hash, git commit and deployment names in every report.
- OpenTelemetry spans per step with `gen_ai.*` token attributes; exportable over OTLP.
- CI runs lint, tests against real Qdrant and Neo4j, the offline evaluation, and a Kubernetes deployment with a smoke test, so a change that breaks parsing, golden labels, retrieval, stores or deployment fails before merge.
- Public deployment: per-IP rate limit and daily USD cap in Redis; model endpoints fail closed without them.
