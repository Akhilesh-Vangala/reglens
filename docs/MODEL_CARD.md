# Model card: RegLens member-service regulation assistant

Written for model-catalog registration. Fields marked *measured* are filled only from committed evaluation reports in `evals/reports/`.

## Intended use
- Explain what federal consumer-finance regulations say, with verbatim citations, to support credit-union member-service staff.
- Scope: Regulation E (12 CFR 1005), NCUA Part 707 (Truth in Savings for credit unions), Regulation CC (12 CFR 229), Regulation Z 1026.12 and 1026.13. Point-in-time text from the eCFR as of 2026-10-02 (see `data/raw/manifest.json` for SHA-256 hashes).

## Out of scope
- Decisions on a specific member's case, legal advice, account actions, credit decisions, institution-specific policies or rates, and any regulation outside the corpus.
- Use with member PII. The system never needs account data. As a backstop, SSN, card, account, email and phone patterns are masked before retrieval, the model and traces (`guard.py`); pattern masking is not a guarantee.

## System design
- Retrieval: dense bge-small-en-v1.5 (open weights, CPU), chosen by measurement over BM25, RRF hybrid and cross-encoder reranking (see README). API embeddings and Azure AI Search hybrid are supported and must be measured before use.
- Generation: Azure OpenAI or OpenAI chat model, temperature 0, strict JSON-schema structured output, prompt version recorded in every trace.
- Control: a deterministic verifier accepts a claim only if its quote is verbatim in a passage retrieved for that question. One retry with verifier feedback; unverified claims are removed before the member-facing answer; an answer with no verified claim becomes "not in corpus" with human review requested.
- Escalation: the model flags live problems (unauthorized transfers, lost cards, disputes) for a representative.
- Cost: a spend ledger enforces a per-request token cap and a cumulative USD cap before each model call; a refused call escalates to human review.

## Evaluation (*measured*)
- Golden set: 43 author-written questions (33 answerable, 4 not-in-corpus, 6 refusal), including 5 adversarial items: prompt injection, PII-laden, system-prompt extraction, role-play legal advice and a false premise. Every gold fact is checked against the regulation text by `tests/test_golden.py`.
- Metrics: answer accuracy (status + required fact + verified claim citing a gold paragraph), unsupported-claim rate, fabricated-citation rate (closed book), refusal, not-in-corpus and escalation accuracy, latency, tokens, cost. Bootstrap 95% CIs over questions; paired deltas between configurations.
- Results: see the latest `evals/reports/*-answers/REPORT.md` and `*-retrieval/REPORT.md`.

## Known limitations
- Golden set is small and author-written; no member or compliance-officer labels yet.
- Exact-substring verification proves a quote exists, not that the claim's interpretation is correct; human review remains the control for interpretation.
- Paragraph structure is reconstructed from inline labels; two source anomalies are repaired and logged (`data/processed/corpus_stats.json`).
- The corpus is a point-in-time snapshot; regulations change and the index must be rebuilt and re-evaluated.

## Governance hooks
- Versioned prompt (`PROMPT_VERSION`), pinned corpus hash, git commit and deployment names in every report.
- OpenTelemetry spans per step with `gen_ai.*` token attributes; exportable over OTLP.
- CI runs the offline evaluation so a change that breaks parsing, golden labels or retrieval fails before merge.
