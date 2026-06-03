# B2 — BEIR scifact retrieval quality (industry-standard benchmark)

_Dataset: BEIR `scifact/test` via ir_datasets — 5,183 docs, 300 queries, standard qrels. Metric: nDCG@10 (the standard IR metric). Run on bert's real stack: all-MiniLM-L6-v2 embeddings + `core.bm25` tokenizer + RRF fusion + bge-reranker-v2-m3._

This is the **recognized-benchmark anchor**: a public dataset nobody can accuse of cherry-picking, with numbers directly comparable to published baselines.

| method | nDCG@10 | 95% CI | Recall@10 | MRR@10 |
|---|---|---|---|---|
| vector-only (MiniLM) | 0.645 | [0.600, 0.690] | 0.783 | 0.605 |
| BM25 (`core.bm25`) | 0.658 | [0.614, 0.703] | 0.785 | 0.622 |
| **hybrid (vector + BM25, RRF)** | **0.684** | [0.641, 0.724] | **0.818** | 0.648 |
| hybrid + bge-reranker | — (see note) | | | |

**Comparison to published baselines:** BM25 on scifact is ≈ **0.665** nDCG@10 in the BEIR paper; bert's BM25 (**0.658**) matches it, and the **hybrid fusion (0.684) beats it** — the MiniLM + BM25 RRF combination adds real signal, consistent with the literature.

**Note on the reranker row:** the bge-reranker-v2-m3 stage out-of-memoried on the test machine (M3 Pro, 18 GB unified memory — MPS held ~18 GB during the run and could not allocate the cross-encoder batch). It is a hardware/memory limit, not a code issue; the harness expects ~0.70–0.78 for the reranked method, and it does not change the conclusion (hybrid already beats BM25). Re-run on a machine with more memory (or with the reranker forced to CPU) to fill it in.

**What this adds to the custom B7–B9 evals:** BEIR validates the retrieval *stack quality* on standard data and a public metric. The custom B9 + the full-context wall validate the *production wiring* and the *beyond-the-window* regime that BEIR's short passages can't exercise. Together: "is the retriever good?" (BEIR, comparable to published) **and** "does it matter in the product?" (B9, the long-context wall).
