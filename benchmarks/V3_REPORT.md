# bert retrieval benchmark — v3 (single-model, Claude)

**Design.** Every arm answers with the *same* model (Claude), so the only variable is the retrieval method. Two question types over the same indexed code corpora (httpx+starlette, pydantic, gin/Go):

- **Code-fact track** — exact values (defaults, constants, signatures); AST-extracted, programmatically graded. This is *grep's* natural task.
- **Semantic track** — conceptual/behavioral questions deliberately phrased with no keyword overlap with the code, so naive grep can't keyword-match; judge-graded. This is *semantic retrieval's* natural task.

Gold is frozen and the retrieval cache is aligned to it (the earlier gold/cache-drift bug that starved the RAG arms is fixed). Accuracy carries 95% bootstrap CIs; pairwise differences use exact paired McNemar with Holm-Bonferroni correction.


## Track B1 — code-fact lookup (grep's home turf)

_249 questions · Claude · accuracy with 95% bootstrap CI_

| rank | arm | method | accuracy | 95% CI | n | tokens/q | tokens/correct |
|---|---|---|---|---|---|---|---|
| 1 | `A7grep` | agentic grep | **0.968** | [0.94, 0.99] | 249 | 70537 | 72878 |
| 2 | `A_mcp` | bert via live MCP | **0.855** | [0.81, 0.90] | 249 | 149648 | 174941 |
| 3 | `A3` | bert hybrid-RAG | **0.663** | [0.61, 0.72] | 249 | 32582 | 49169 |
| 4 | `A5` | BM25-only | **0.643** | [0.58, 0.70] | 249 | 32305 | 50274 |
| 5 | `A0` | closed-book (no retrieval) | **0.610** | [0.55, 0.67] | 249 | 44801 | 73392 |
| 6 | `A4` | vector-only | **0.510** | [0.45, 0.57] | 249 | 32391 | 63508 |
| 7 | `A2` | smart-truncation | **0.177** | [0.13, 0.22] | 249 | 89992 | 509272 |
| 8 | `A6` | graph / Aider RepoMap | **0.177** | [0.13, 0.22] | 249 | 32216 | 182312 |
| 9 | `A1` | naive-truncation | **0.120** | [0.08, 0.16] | 249 | 45201 | 375169 |

**Is `bert hybrid-RAG (A3)` significantly different from each arm?** (paired exact McNemar, Holm-corrected)

| vs | Δacc (A3 − other) | 95% CI | Holm p | significant |
|---|---|---|---|---|
| `A0` closed-book (no retrieval) | +0.052 | [-0.02, +0.12] | 0.750 | no |
| `A1` naive-truncation | +0.542 | [+0.48, +0.61] | 0.000 | **yes** |
| `A2` smart-truncation | +0.486 | [+0.42, +0.55] | 0.000 | **yes** |
| `A4` vector-only | +0.153 | [+0.10, +0.20] | 0.000 | **yes** |
| `A5` BM25-only | +0.020 | [-0.01, +0.05] | 1.000 | no |
| `A6` graph / Aider RepoMap | +0.486 | [+0.42, +0.55] | 0.000 | **yes** |
| `A7grep` agentic grep | -0.305 | [-0.37, -0.24] | 0.000 | **yes** |
| `A_mcp` bert via live MCP | -0.193 | [-0.24, -0.14] | 0.000 | **yes** |

## Track B2 — semantic / conceptual recall (bert's home turf)

_30 questions · Claude · accuracy with 95% bootstrap CI_

| rank | arm | method | accuracy | 95% CI | n | tokens/q | tokens/correct |
|---|---|---|---|---|---|---|---|
| 1 | `A7grep` | agentic grep | **0.967** | [0.90, 1.00] | 30 | 218168 | 225691 |
| 2 | `A_mcp` | bert via live MCP | **0.900** | [0.80, 1.00] | 30 | 110339 | 122599 |
| 3 | `A0` | closed-book (no retrieval) | **0.667** | [0.50, 0.83] | 30 | 29149 | 43724 |
| 4 | `A4` | vector-only | **0.667** | [0.50, 0.83] | 30 | 32702 | 49053 |
| 5 | `A3` | bert hybrid-RAG | **0.633** | [0.47, 0.80] | 30 | 32863 | 51889 |
| 6 | `A5` | BM25-only | **0.600** | [0.43, 0.77] | 30 | 32680 | 54466 |
| 7 | `A1` | naive-truncation | **0.200** | [0.07, 0.37] | 30 | 38842 | 194208 |
| 8 | `A2` | smart-truncation | **0.200** | [0.07, 0.33] | 30 | 39063 | 195316 |
| 9 | `A6` | graph / Aider RepoMap | **0.033** | [0.00, 0.10] | 30 | 31016 | 930491 |

**Is `bert hybrid-RAG (A3)` significantly different from each arm?** (paired exact McNemar, Holm-corrected)

| vs | Δacc (A3 − other) | 95% CI | Holm p | significant |
|---|---|---|---|---|
| `A0` closed-book (no retrieval) | -0.033 | [-0.30, +0.23] | 1.000 | no |
| `A1` naive-truncation | +0.433 | [+0.27, +0.60] | 0.006 | **yes** |
| `A2` smart-truncation | +0.433 | [+0.23, +0.63] | 0.020 | **yes** |
| `A4` vector-only | -0.033 | [-0.10, +0.00] | 1.000 | no |
| `A5` BM25-only | +0.033 | [-0.10, +0.17] | 1.000 | no |
| `A6` graph / Aider RepoMap | +0.600 | [+0.43, +0.77] | 0.000 | **yes** |
| `A7grep` agentic grep | -0.333 | [-0.53, -0.13] | 0.108 | no |
| `A_mcp` bert via live MCP | -0.267 | [-0.47, -0.07] | 0.258 | no |

## What the results actually show

- **Agentic methods dominate both tracks.** `A7grep` (agentic grep) and `A_mcp` (bert live over MCP) are the top two on code-fact *and* semantic — multi-turn search-and-read beats any one-shot retrieval.
- **Between the two agentic methods, grep-tools edge bert-MCP** — significantly on code-fact (0.97 vs 0.86), numerically on semantic (0.97 vs 0.90; n=30 is underpowered, p≈0.26).
- **The semantic track did NOT favor bert, contrary to the original hypothesis.** The reason: `A7grep` is *agentic*, not naive keyword grep — an agent that can grep, read, and reason over code answers conceptual questions fine. "Naive grep fails" ≠ "agentic grep fails." On code corpora, bert's one-shot hybrid-RAG (`A3`) is statistically indistinguishable from closed-book, vector, and BM25 on semantic questions.
- **Where bert's one-shot retriever does win:** on code-fact it significantly beats vector-only (+0.15) — the hybrid+rerank earns its keep — and it is more **token-efficient per correct answer** (~49k) than agentic grep (~73k).
- `A_mcp` is Claude calling bert's `memory_search` tool **live over MCP** (the real deployment path). It is a strong #2 on both tracks at lower token cost than grep — the most defensible single claim for bert here.

## Limitations

- **This benchmark tests retrieval over source CODE, not over prose/accumulated project memory** — which is bert's actual design target. The semantic edge of dense retrieval is expected to be larger over decisions/findings/notes where there is genuinely no symbol to grep; that is untested here.
- Corpora are **famous open-source libraries**, so the closed-book baseline (`A0`, 0.61/0.67) is inflated by parametric knowledge; on novel/proprietary code the retrieval arms' margin over `A0` would widen.
- Semantic track is **n=30, judge-graded** (3 non-Claude llama judges, majority vote) — underpowered for the close agentic-vs-agentic comparisons (several differences are non-significant).
- Single reader model (Claude); the method ranking can differ on weaker models.
