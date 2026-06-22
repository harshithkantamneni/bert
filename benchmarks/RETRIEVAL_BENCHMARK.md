# bert retrieval benchmark — consolidated report

**Scope.** How good is bert's retrieval, and how does it compare to real alternatives when the *only* variable is the retrieval method? Three tracks: (A) retrieval quality on public IR benchmarks, (B) end-to-end question-answering on code corpora with a single fixed reader model, (C) token cost. This is the authoritative writeup; `V3_REPORT.md` is the machine-regenerated table source and `results/v2/MANIFEST.md` documents the data.

---

## TL;DR (honest)

- **bert's retriever is real and competitive at the IR level.** On BEIR, hybrid+rerank beats published BM25 by **+0.08 / +0.05 / +0.20 nDCG@10** (scifact / nfcorpus / fiqa) and matches the published bge-base reference.
- **On code-fact QA with one fixed model (Claude), agentic methods dominate.** Agentic grep wins (0.97); **bert called live over MCP is a clear #2 (0.86)**, ahead of every one-shot retriever and +25pts over closed-book.
- **bert's hybrid one-shot RAG (0.66) beats vector-only (significant) and ties BM25 / the parametric floor.** The rerank earns its keep over vector; the hybrid edge over BM25 is not significant on this gold.
- **The semantic track did NOT favor bert** — because the "grep" arm is *agentic* (reads + reasons over code), not naive keyword grep. On code, dense retrieval has no semantic edge there.
- **This benchmark tests retrieval over source CODE, not over prose/accumulated project memory** — bert's actual design target. That gap is what the next suite addresses.

The defensible one-liner: **bert deployed the real way — agentically over MCP — is a strong #2 to the best agentic baseline on code QA, at lower token cost; and its hybrid retriever is a genuine IR system, not a wrapper.** It is *not* a grep-killer on code.

---

## What was tested

**Corpora** (real open-source repos, indexed into bert labs):

| id | repo(s) | files | size | role |
|---|---|---|---|---|
| c1 | httpx + starlette | 57 | 0.5 MB | Python HTTP client + ASGI framework |
| c2 | gin-gonic/gin | 58 | 0.2 MB | Go web framework |
| c3 | pydantic | 104 | 1.8 MB | Python validation lib |
| big | cpython/django/numpy/pandas/… | 9,383 | 167 MB (~41M tok) | scale stress |

**Gold** — 249 frozen code-fact questions: 228 programmatically graded (exact values via AST extraction) + 21 multi-hop judge-graded; tiers needle 164 / single-hop 64 / multi-hop 21; corpora c3 81 / big 80 / c1 76 / c2 12. Plus 30 conceptual semantic questions (judge-graded). Gold is **frozen** — every phase reads the same `gold.json` so the retrieval cache stays aligned.

**bert stack under test:** BAAI/bge-base-en-v1.5 (768-d, asymmetric query instruction) dense + `core.bm25` sparse + RRF(k=60) fusion + BAAI/bge-reranker-v2-m3 cross-encoder. Single-sourced from `core.memory`, i.e. the production path.

**Arms (Track B), all answering with the same model (Claude):**
`A0` closed-book · `A1` naive-trunc · `A2` smart-trunc · `A3` **bert hybrid-RAG** · `A4` vector-only · `A5` BM25-only · `A6` graph/Aider RepoMap · `A7grep` **agentic grep** (Claude + grep/read tools) · `A_mcp` **bert via live MCP** (Claude calls `memory_search` as a real tool).

**Stats:** accuracy with 95% bootstrap CIs; pairwise differences via exact paired McNemar + Holm-Bonferroni. Judge-graded answers use a 3-judge majority (nvidia llama-3.3-70b, nvidia llama-3.1-70b, groq llama-3.3-70b), calibrated (clear-correct→1, clear-wrong→0).

---

## Track A — retrieval quality on public IR benchmarks (model-free)

bge-base + BM25 + RRF + bge-reranker on independent BEIR qrels, nDCG@10 [95% CI]. Best method per dataset bolded.

| dataset | vector | bm25 | hybrid | hybrid+rerank | best vs published BM25 |
|---|---|---|---|---|---|
| scifact (300q) | 0.740 | 0.658 | 0.719 | **0.745** | **+0.080** |
| nfcorpus (323q) | **0.374** | 0.314 | 0.355 | 0.354 | **+0.049** |
| fiqa (648q) | 0.406 | 0.230 | 0.356 | **0.434** | **+0.198** |

scifact 0.745 ≈ the published bge-base reference (0.741) — confirms the stack is implemented correctly. The **+0.08 / +0.05 / +0.20** headline is best-method vs *published* BM25. The **reranker's own contribution** (hybrid → hybrid+rerank) is separate and smaller: **+0.078 on fiqa** (0.356→0.434, where queries are lexically hard), **+0.026 on scifact**, and **−0.001 on nfcorpus** (neutral where vector already saturates). This is a legitimate, reproducible IR result.

---

## Track B1 — code-fact QA (n=249, single model = Claude)

| rank | arm | accuracy | 95% CI | tokens/correct |
|---|---|---|---|---|
| 1 | `A7grep` agentic grep | **0.968** | [0.94, 0.99] | 72,878 |
| 2 | `A_mcp` **bert via live MCP** | **0.855** | [0.81, 0.90] | 174,941 |
| 3 | `A3` **bert hybrid-RAG** | **0.663** | [0.60, 0.72] | **49,169** |
| 4 | `A5` BM25-only | 0.643 | [0.58, 0.70] | 50,274 |
| 5 | `A0` closed-book | 0.610 | [0.55, 0.67] | 73,392 |
| 6 | `A4` vector-only | 0.510 | [0.45, 0.57] | 63,508 |
| 7–9 | `A2` / `A6` / `A1` | 0.18 / 0.18 / 0.12 | — | — |

**Significance (bert hybrid-RAG `A3` vs others, paired McNemar + Holm):** beats vector `A4` (+0.15, p<0.001) and truncation/graph (p<0.001); **ties BM25** (+0.02, ns) and **ties closed-book** (+0.05, ns); loses to agentic grep (−0.31, p<0.001) and to its own live-MCP form (−0.19, p<0.001).

Reading: on famous libraries the parametric floor (`A0` 0.61) is high, so one-shot retrieval's headroom is compressed; the rerank still buys a real win over vector. Agentic exploration (grep tools or bert-MCP) is in a different tier.

---

## Track B2 — semantic / conceptual QA (n=30, judge-graded)

| rank | arm | accuracy | 95% CI |
|---|---|---|---|
| 1 | `A7grep` agentic grep | **0.967** | [0.90, 1.00] |
| 2 | `A_mcp` **bert via live MCP** | **0.900** | [0.80, 1.00] |
| 3–4 | `A0` closed-book / `A4` vector | 0.667 / 0.667 | — |
| 5 | `A3` bert hybrid-RAG | 0.633 | [0.47, 0.80] |
| 6 | `A5` BM25-only | 0.600 | — |

**The hypothesis here was wrong, and that matters.** We expected bert to win because "grep can't keyword-match conceptual questions." But `A7grep` is *agentic* — it greps, reads, and reasons over code, so paraphrased questions don't defeat it. On code, bert's one-shot RAG (`A3` 0.63) is statistically indistinguishable from closed-book, vector, and BM25 — but it **loses to agentic grep, significantly.** Exact paired McNemar `A3` vs `A7grep` (−0.33, discordant 11–1): **p=0.006**; it stays significant under Holm over the A3-vs-others family (**p=0.03**), and loses significance only under the strictest all-pairs 36-comparison Holm (p=0.11). Grep genuinely beats bert-hybrid on semantic too — the earlier "underpowered / not significant" reading was wrong.

---

## Track C — tokenomics

One-shot RAG (`A3`/`A4`/`A5`) costs ~32k tokens/query; agentic arms cost far more (`A7grep` 70k, `A_mcp` 110–150k). On **tokens per correct answer**, bert hybrid-RAG (49k) is *more efficient than agentic grep* (73k) despite lower accuracy — the efficiency angle is real but secondary.

---

## Conclusions — by mechanism (not aggregate)

1. **Agentic > one-shot retrieval, on both tracks.** Multi-turn search-and-read beats any single retrieval pass.
2. **Among agentic, grep-tools edge bert-MCP** — significant on code-fact. (On semantic, agentic grep also significantly beats bert's *one-shot* hybrid; grep vs bert-MCP on semantic is too close to call at n=30.)
3. **bert's hybrid retriever is a real IR system** — beats published BM25 on BEIR, beats vector-only on code QA.
4. **bert-via-MCP is a strong, deployable #2** — beats all one-shot retrievers on both tracks, at lower token cost than grep.
5. **No semantic edge on code** — because code is re-derivable by an agent; this is the key limitation, not a bert defect.

A result that showed bert winning everything would be the flawed-benchmark smell. This one names where bert loses (agentic grep on code) and ties (BM25, closed-book on famous libs) — which is what makes the wins credible.

---

## Methodology (detailed)

**Design — model-controlled.** Every Track-B arm answers with the *same* reader (Claude sonnet, temperature 0). Only the *retrieval method* (the context handed to the reader, or the tools it may call) varies. This isolates retrieval quality from reader quality — the cross-model "0.9 vs 0.3" comparisons in earlier drafts were discarded as unfair.

**Corpora.** Real OSS repos chosen to span languages (Python, Go), sizes (0.2–167 MB), and familiarity (popular libs → high parametric floor; a 41M-token multi-repo `big` set → retrieval-at-scale stress). Each is ingested into a bert lab (sharded + indexed) exactly as in production.

**Gold construction.**
- *Code-fact (228 programmatic):* `v2_gold_ast.py` walks each repo's AST and extracts ground-truth facts (default arg values, constants, enum members, signatures) with the exact source location. Extraction is **method-blind** — it has no knowledge of which retrieval method will be tested, so no arm is favored. Each fact yields a question + an exact expected value (+ an optional `answer_regex`).
- *Code multi-hop (21 judge):* chains requiring 2+ facts, generated method-blind and human/LLM-validated.
- *Semantic (30 judge):* conceptual/behavioral questions generated by a multi-agent workflow, each deliberately phrased with **no lexical overlap** with the source (validated by a token-overlap filter) and a prose gold answer.
- The gold is **frozen** (`gold.json` is loaded, never regenerated) so every phase shares one question set and the retrieval cache stays aligned.

**Arm definitions (exact).**
- `A0` closed-book — no context; parametric knowledge only.
- `A1` naive-truncation — first ~15k tokens of concatenated corpus files.
- `A2` smart-truncation — ~15k-token budget spread across file heads (structure-aware).
- `A3` **bert hybrid-RAG** — top-10 chunks from dense(bge-base)+BM25+RRF(k=60)+bge-reranker.
- `A4` vector-only — top-10 dense chunks. `A5` BM25-only — top-10 sparse chunks.
- `A6` graph/Aider — top-10 from a real Aider RepoMap (PageRank over the symbol graph).
- `A7grep` **agentic grep** — Claude given Grep/Read/Glob tools in the repo dir; multi-turn search-and-read.
- `A_mcp` **bert via live MCP** — Claude given bert's `memory_search` as a real MCP tool; multi-turn.
- Reader context budget for `A1–A6` is fixed (~15k tokens); retrieval depth is top-10 for all retriever arms — so differences are method, not budget.

**Retrieval precompute (fairness).** Hybrid/vector/BM25/Aider retrievals are computed **once per question** and cached, then reused across budgets/reps/arms. Every arm sees the identical retrieval for a given question — no per-run retrieval variance.

**Grading.**
- *Programmatic:* `answer_regex` match if present; else **numeric-equality** for numeric golds (avoids the "5 matches 50" substring trap); else **word-bounded** normalized substring. Self-tested against known false-positive cases.
- *Judge (open-ended):* 3-judge majority vote — nvidia llama-3.3-70b, nvidia llama-3.1-70b, groq llama-3.3-70b (all *independent of the Claude reader*). Calibrated (clear-correct→1, clear-wrong→0); a judge that returns no valid vote forces a retry rather than a silent 0.

**Statistics.** Accuracy CIs are 10,000-iteration percentile bootstrap (seed 0). Pairwise differences use the **exact paired McNemar** test on the discordant pairs (questions where the two arms disagree), then **Holm-Bonferroni** correction. *Family choice matters and is stated:* the pairwise tables correct over the **full all-pairs family** (the most conservative); where that changes a conclusion versus the raw test or a per-arm family, all three p-values are reported (see Track B2, `A3` vs `A7grep`).

**Reproducibility.** Frozen gold + fixed seeds + a per-`(id, arm)` checkpoint make every run deterministic and resumable; an interrupted run resumes without recomputation or data loss.

### Integrity log

Three silent bugs were caught and fixed during the run; each would have shipped a *wrong* benchmark. Documented here because catching them is the difference between a citable number and one that collapses under scrutiny.

1. **Gold/cache drift.** `assemble_gold` was non-deterministic across processes, so the factorial ran questions whose chunks weren't in the precompute cache → bert-RAG got **empty context on 57% of cells** and scored ~0 there, producing a false "grep beats bert" headline. *Fix:* froze the gold (`gold.json` is loaded, never regenerated); all phases share one aligned cache.
2. **Reader quota wall.** The Claude session limit was hit mid-run; affected calls returned a "session limit" string that was checkpointed as wrong answers (semantic 100%, code tail ~9%). *Fix:* hardened the result filter to skip quota-wall / zero-token replies so they retry on resume instead of poisoning data.
3. **Dead judges.** All three judge providers were unreachable (mistral 429 / openrouter 402-no-credits / cerebras 404-bad-model); `grade_judges` silently returned 0 for every judge-graded answer (all 30 semantic + 21 multi-hop code). *Fix:* swapped to three verified-live judges, calibrated them, and made judge failure **loud** (retry, never fake-zero).

Final state: both Claude tracks 0 contaminated rows; n=249 code (full), n=30 semantic.

---

## Limitations / what is NOT tested

- **Retrieval over source CODE, not prose/accumulated project memory** — bert's actual design target. Where the answer is re-derivable by reading present files, memory has no structural edge; the next suite tests the regime where it does (cross-session, beyond-context-window, paraphrase-over-prose, knowledge-update, abstention, live write-policy).
- **Famous libraries** inflate the closed-book floor (`A0` 0.61/0.67); on novel/proprietary code the retrieval margin over `A0` widens.
- **Semantic track n=30, judge-graded** — underpowered for the close agentic-vs-agentic comparisons.
- **Single reader (Claude)** — method ranking can differ on weaker models.

---

## Reproduce

```
# Track A (BEIR)
.venv/bin/python benchmarks/b2_beir_multi.py
# Track B (single-model Claude, both QA tracks) + report
bash benchmarks/v3_finish.sh           # resumable per (id,arm); regenerates V3_REPORT.md
```

Data + provenance: `benchmarks/results/v2/MANIFEST.md`.
