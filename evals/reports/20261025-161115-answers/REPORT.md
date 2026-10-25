# Answer evaluation

Model `claude-haiku-5-5`, embeddings `bge-small-en-v1.5 (local)`, retriever `dense_bge` (k=12), prompt `rag-v2`, repeats 1.
Corpus manifest sha256 `d2b04018c227`, golden sha256 `990e2287b30b`, commit `7bb9a9c`.

| Config | Answer accuracy (95% CI) | Status acc. | Refusal | Not-in-corpus | Escalation | Unsupported claims shown | Fabricated citations | p50 s | Cost/question |
|---|---|---|---|---|---|---|---|---|---|
| C_rag_verified | 0.82 (0.67-0.94) | 0.93 (0.84-1.00) | 1.00 (1.00-1.00) | 0.75 (0.25-1.00) | 1.00 (1.00-1.00) | 0.0% | n/a | 5.6 | $0.00103 |
| D_copilot | 0.94 (0.85-1.00) | 0.95 (0.88-1.00) | 0.83 (0.50-1.00) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 0.0% | n/a | 6.7 | $0.00132 |

## Like for like

Answer accuracy above is stricter for retrieval configurations (it also requires a verified quote from a gold paragraph). These columns use one definition for every configuration.

| Config | Right fact | Verifiably grounded | Declined when it should |
|---|---|---|---|
| C_rag_verified | 0.88 (0.76-0.97) | 0.88 (0.76-0.97) | 1.00 (1.00-1.00) |
| D_copilot | 0.94 (0.85-1.00) | 0.97 (0.91-1.00) | 1.00 (1.00-1.00) |

## Paired differences in answer accuracy (same questions)

| Contrast | Delta (95% CI) | Significant | Improved | Regressed |
|---|---|---|---|---|

Unsupported claims are claims whose quote is not verbatim in a passage retrieved for that question. Fabricated citations (closed book) are cited paragraphs that do not exist in the pinned corpus. Grading is deterministic: regexes, citation prefixes and the verifier; no LLM judge.
