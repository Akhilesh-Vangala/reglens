# Answer evaluation

Model `claude-haiku-5-5`, embeddings `bge-small-en-v1.5 (local)`, retriever `dense_bge` (k=8), prompt `rag-v2`, repeats 1.
Corpus manifest sha256 `d2b04018c227`, golden sha256 `990e2287b30b`, commit `117d234`.

| Config | Answer accuracy (95% CI) | Status acc. | Refusal | Not-in-corpus | Escalation | Unsupported claims shown | Fabricated citations | p50 s | Cost/question |
|---|---|---|---|---|---|---|---|---|---|
| A_closed_book | 0.82 (0.67-0.94) | 0.95 (0.88-1.00) | 1.00 (1.00-1.00) | 0.50 (0.00-1.00) | 1.00 (1.00-1.00) | n/a | 0.9% | 6.1 | $0.00063 |
| B_rag_no_verify | 0.76 (0.61-0.91) | 0.91 (0.81-0.98) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 0.0% | n/a | 4.7 | $0.00087 |
| C_rag_verified | 0.79 (0.64-0.91) | 0.91 (0.81-0.98) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 0.0% | n/a | 5.4 | $0.00089 |
| D_copilot | 0.85 (0.73-0.97) | 0.81 (0.70-0.93) | 1.00 (1.00-1.00) | 0.00 (0.00-0.00) | 1.00 (1.00-1.00) | 0.0% | n/a | 6.7 | $0.00162 |

## Like for like

Answer accuracy above is stricter for retrieval configurations (it also requires a verified quote from a gold paragraph). These columns use one definition for every configuration.

| Config | Right fact | Verifiably grounded | Declined when it should |
|---|---|---|---|
| A_closed_book | 0.82 (0.67-0.94) | n/a (no sources) | 1.00 (1.00-1.00) |
| B_rag_no_verify | 0.82 (0.70-0.94) | 0.79 (0.64-0.91) | 1.00 (1.00-1.00) |
| C_rag_verified | 0.85 (0.73-0.97) | 0.82 (0.67-0.94) | 1.00 (1.00-1.00) |
| D_copilot | 0.85 (0.73-0.97) | 0.88 (0.76-0.97) | 1.00 (1.00-1.00) |

## Paired differences in answer accuracy (same questions)

| Contrast | Delta (95% CI) | Significant | Improved | Regressed |
|---|---|---|---|---|
| A_closed_book->B_rag_no_verify | -0.06 (-0.24 to +0.12) | no | 4 | E03, E04, O05, T04, X01, Z01 |
| B_rag_no_verify->C_rag_verified | +0.03 (+0.00 to +0.09) | no | 1 | - |
| A_closed_book->C_rag_verified | -0.03 (-0.21 to +0.15) | no | 5 | E03, E04, O05, T04, X01, Z01 |

Unsupported claims are claims whose quote is not verbatim in a passage retrieved for that question. Fabricated citations (closed book) are cited paragraphs that do not exist in the pinned corpus. Grading is deterministic: regexes, citation prefixes and the verifier; no LLM judge.
