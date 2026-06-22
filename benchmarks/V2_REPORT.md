> **⚠️ SUPERSEDED — do not cite.** This is the v2 two-tier run, invalidated by the gold/cache-drift bug (bert-RAG was starved of context on 57% of cells). See [`RETRIEVAL_BENCHMARK.md`](RETRIEVAL_BENCHMARK.md) for the corrected, authoritative results.

# bert retrieval benchmark — v2 (research-grade)

A from-scratch redo of the v1 pilot with the controls v1 lacked: a closed-book
baseline, deterministic + programmatically-graded gold, paired statistics with
confidence intervals, multiple corpora (incl. a large-scale one), an established
independent-qrels retrieval track, and the real alternatives (graph/Aider,
agentic-grep with both a weak AND a frontier agent).

## Methodology (what makes this defensible)
- **Closed-book control (A0):** the reader answers with NO context, so accuracy
  attributable to retrieval is separated from parametric knowledge (popular libs
  leak into pretraining).
- **Deterministic, method-blind gold:** 228 questions extracted from code by
  AST (default values, constants, regexes, enum members) and graded by exact /
  numeric / regex match — no LLM judge, no authoring bias. Plus 21
  method-blind multi-hop questions (judge-graded, non-Claude judges).
- **Paired statistics:** per-arm bootstrap 95% CIs; arm-vs-arm via exact
  McNemar on discordant pairs; Holm-Bonferroni family-wise correction. Fine
  differences are reported as significant ONLY when they clear that bar.
- **Multiple corpora** of varied language/size + a large-scale corpus that
  exceeds any context window. **Pinned reader provider** (recorded per call),
  k=3 repeats for reader variance. Cost + latency tracked per arm.

## Corpora
| corpus | lang | files | ~tokens |
|---|---|---|---|
| c1 | python | 57 | 131,582 |
| c2 | go | 58 | 55,015 |
| c3 | python | 104 | 442,004 |
| big | python | 9383 | 41,884,569 |

## Track A — retrieval quality on established datasets (independent qrels)
Retrieval quality on established datasets with INDEPENDENT qrels (driven by bert's production embedder bge-base-en-v1.5; RRF k=60; cross-encoder = bge-reranker-v2-m3). Metric: nDCG@10 with bootstrap 95% CI.


**arguana**

| method | nDCG@10 [95% CI] | R@10 | MRR@10 | p95 lat |
|---|---|---|---|---|
| bm25_only | 0.335 [0.301,0.367] | 0.719 | 0.215 | 193ms |
| vector_only | 0.431 [0.400,0.459] | 0.849 | 0.296 | 46ms |
| hybrid_no_rerank | 0.412 [0.381,0.442] | 0.824 | 0.281 | 257ms |

**fiqa**

| method | nDCG@10 [95% CI] | R@10 | MRR@10 | p95 lat |
|---|---|---|---|---|
| bm25_only | 0.298 [0.253,0.350] | 0.364 | 0.347 | 30ms |
| vector_only | 0.497 [0.448,0.551] | 0.571 | 0.559 | 38ms |
| hybrid_no_rerank | 0.433 [0.383,0.488] | 0.524 | 0.489 | 63ms |
| hybrid_with_rerank | 0.496 [0.450,0.546] | 0.576 | 0.575 | 9682ms |

**nfcorpus**

| method | nDCG@10 [95% CI] | R@10 | MRR@10 | p95 lat |
|---|---|---|---|---|
| bm25_only | 0.349 [0.307,0.397] | 0.152 | 0.557 | 3ms |
| vector_only | 0.403 [0.355,0.451] | 0.187 | 0.596 | 24ms |
| hybrid_no_rerank | 0.381 [0.339,0.430] | 0.176 | 0.590 | 16ms |
| hybrid_with_rerank | 0.393 [0.349,0.439] | 0.189 | 0.590 | 3482ms |

**scidocs**

| method | nDCG@10 [95% CI] | R@10 | MRR@10 | p95 lat |
|---|---|---|---|---|
| bm25_only | 0.154 [0.127,0.185] | 0.164 | 0.266 | 39ms |
| vector_only | 0.235 [0.206,0.269] | 0.247 | 0.401 | 42ms |
| hybrid_no_rerank | 0.215 [0.184,0.248] | 0.226 | 0.370 | 71ms |

**scifact**

| method | nDCG@10 [95% CI] | R@10 | MRR@10 | p95 lat |
|---|---|---|---|---|
| bm25_only | 0.684 [0.631,0.741] | 0.815 | 0.647 | 11ms |
| vector_only | 0.733 [0.686,0.781] | 0.876 | 0.696 | 26ms |
| hybrid_no_rerank | 0.749 [0.703,0.801] | 0.871 | 0.717 | 27ms |
| hybrid_with_rerank | 0.744 [0.698,0.797] | 0.855 | 0.715 | 3582ms |

## Track B — end-to-end QA accuracy — TIER 1: free-tier reader (llama-3.3-70b, bert's runtime)
End-to-end answer accuracy (60-249 questions per arm; A7f frontier runs on a subset, so pairwise tests align per-pair on shared questions) (deterministic tier graded programmatically; multi-hop tier by a non-Claude judge). Reader pinned to llama-3.3-70b; k=3 repeats collapsed per item. Accuracy with bootstrap 95% CI.


| arm | description | accuracy [95% CI] | p50 lat | mean $ |
|---|---|---|---|---|
| A7f | agentic-grep (frontier Claude agent) | 0.883 [0.800,0.950] | 42922ms | $0.1022 |
| A7w | agentic-grep (weak/free agent) | 0.639 [0.578,0.699] | 20140ms | $0.0000 |
| A3 | hybrid RAG (vec+bm25+RRF+rerank) | 0.301 [0.245,0.361] | 2411ms | $0.0000 |
| A5 | bm25 RAG | 0.301 [0.245,0.357] | 1730ms | $0.0000 |
| A0 | closed-book (no context) | 0.269 [0.217,0.325] | 2333ms | $0.0000 |
| A2 | smart truncation | 0.269 [0.213,0.325] | 5444ms | $0.0000 |
| A4 | vector RAG | 0.265 [0.209,0.321] | 2264ms | $0.0000 |
| A6 | graph RAG (real Aider RepoMap) | 0.097 [0.043,0.161] | 2135ms | $0.0000 |
| A1 | naive truncation | 0.096 [0.060,0.133] | 2837ms | $0.0000 |

**Significant pairwise differences (McNemar, Holm-corrected p<.05):**

- **A6 vs A7f**: Δacc=-0.730 CI=[-0.865,-0.568], Holm p=0.0000
- **A1 vs A7f**: Δacc=-0.700 CI=[-0.833,-0.567], Holm p=0.0000
- **A2 vs A7f**: Δacc=-0.667 CI=[-0.783,-0.533], Holm p=0.0000
- **A4 vs A7f**: Δacc=-0.617 CI=[-0.750,-0.483], Holm p=0.0000
- **A3 vs A6**: Δacc=+0.613 CI=[+0.505,+0.720], Holm p=0.0000
- **A5 vs A6**: Δacc=+0.613 CI=[+0.505,+0.720], Holm p=0.0000
- **A0 vs A7f**: Δacc=-0.600 CI=[-0.750,-0.450], Holm p=0.0000
- **A5 vs A7f**: Δacc=-0.600 CI=[-0.733,-0.467], Holm p=0.0000
- **A3 vs A7f**: Δacc=-0.567 CI=[-0.700,-0.433], Holm p=0.0000
- **A6 vs A7w**: Δacc=-0.548 CI=[-0.656,-0.441], Holm p=0.0000
- **A1 vs A7w**: Δacc=-0.542 CI=[-0.610,-0.474], Holm p=0.0000
- **A4 vs A6**: Δacc=+0.516 CI=[+0.409,+0.624], Holm p=0.0000
- **A4 vs A7w**: Δacc=-0.373 CI=[-0.450,-0.297], Holm p=0.0000
- **A0 vs A7w**: Δacc=-0.369 CI=[-0.438,-0.301], Holm p=0.0000
- **A2 vs A7w**: Δacc=-0.369 CI=[-0.442,-0.297], Holm p=0.0000
- **A3 vs A7w**: Δacc=-0.337 CI=[-0.414,-0.261], Holm p=0.0000
- **A5 vs A7w**: Δacc=-0.337 CI=[-0.410,-0.261], Holm p=0.0000
- **A7f vs A7w**: Δacc=+0.300 CI=[+0.150,+0.450], Holm p=0.0069
- **A2 vs A6**: Δacc=+0.258 CI=[+0.161,+0.355], Holm p=0.0000
- **A1 vs A3**: Δacc=-0.205 CI=[-0.273,-0.141], Holm p=0.0000
- **A1 vs A5**: Δacc=-0.205 CI=[-0.273,-0.137], Holm p=0.0000
- **A0 vs A1**: Δacc=+0.173 CI=[+0.108,+0.241], Holm p=0.0000
- **A1 vs A2**: Δacc=-0.173 CI=[-0.241,-0.104], Holm p=0.0000
- **A0 vs A6**: Δacc=+0.172 CI=[+0.075,+0.269], Holm p=0.0185
- **A1 vs A4**: Δacc=-0.169 CI=[-0.237,-0.100], Holm p=0.0001

**Notable NON-significant pairs (cannot distinguish at this n):**

- A3 vs A4: Δacc=+0.036 CI=[-0.004,+0.076], Holm p=1.000
- A3 vs A5: Δacc=+0.000 CI=[-0.020,+0.020], Holm p=1.000
- A4 vs A5: Δacc=-0.036 CI=[-0.072,+0.000], Holm p=0.862

**Accuracy by tier:**

| arm | multi_hop | needle | single_hop |
|---|---|---|---|
| A0 | 0.206 | 0.165 | 0.525 |
| A1 | 0.206 | 0.081 | 0.083 |
| A2 | 0.159 | 0.21 | 0.422 |
| A3 | 0.365 | 0.285 | 0.299 |
| A4 | 0.222 | 0.246 | 0.309 |
| A5 | 0.317 | 0.292 | 0.294 |
| A6 | 0.0 | 0.102 | 0.174 |
| A7f | 0.714 | 0.962 | 1.0 |
| A7w | 0.27 | 0.627 | 0.647 |

**Accuracy by corpus:**

| arm | big | c1 | c2 | c3 |
|---|---|---|---|---|
| A0 | 0.35 | 0.474 | 0.0 | 0.029 |
| A1 | 0.0 | 0.11 | 0.0 | 0.181 |
| A2 | 0.167 | 0.548 | 0.0 | 0.132 |
| A3 | 0.087 | 0.491 | 0.278 | 0.321 |
| A4 | 0.087 | 0.452 | 0.306 | 0.247 |
| A5 | 0.096 | 0.487 | 0.25 | 0.317 |
| A6 | - | 0.1 | 0.0 | 0.129 |
| A7f | 1.0 | 0.778 | 0.833 | 0.933 |
| A7w | 0.592 | 0.737 | 0.0 | 0.576 |

**Truncation budget sweep (accuracy vs token window):**

| arm | 5K | 15K | 30K | 60K |
|---|---|---|---|---|
| A1 | 0.023 | 0.092 | 0.173 | 0.361 |
| A2 | 0.096 | 0.264 | 0.384 | 0.474 |

## Track B — TIER 2: all-Claude (model held constant; bert-as-MCP, deployment-realistic)
_(all-Claude tier not yet run)_

## Track C — tokenomics (efficiency: how much each method burns)
_(tokenomics pass not yet run)_

## Limitations
- Multi-hop gold (judge tier) is LLM-generated (method-blind) + LLM-judged;
  softer than the deterministic tier. The deterministic tier is the rigorous core.
- BEIR datasets use a gold-preserving doc-pool subsample (max-docs) for tractable
  encoding on an 18 GB M3 Pro; codesearchnet (2M docs) was excluded for
  tractability — cqadupstack/programmers is the programming-domain proxy.
- Reader is a single free model (llama-3.3-70b); absolute accuracies would shift
  with a stronger reader, but the BETWEEN-ARM comparison holds the reader fixed.
- Agentic-grep frontier arm (A7f) runs on a question subset (Max-plan cost).
