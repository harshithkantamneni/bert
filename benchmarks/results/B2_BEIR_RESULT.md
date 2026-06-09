# B2 — BEIR scifact retrieval quality (industry-standard benchmark)

_Dataset: BEIR `scifact/test` via ir_datasets — 5,183 docs, 300 queries, standard qrels. Metric: nDCG@10 (the standard IR metric), with bootstrap 95% CI. Run on bert's real stack, with the embedder single-sourced from `core.memory` so the benchmark measures the production encoder, not a hardcoded copy: **BAAI/bge-base-en-v1.5** (768-dim, asymmetric query instruction) + `core.bm25` tokenizer + RRF(k=60) fusion + **BAAI/bge-reranker-v2-m3** cross-encoder._

This is the **recognized-benchmark anchor**: a public dataset nobody can accuse of cherry-picking, with numbers directly comparable to published baselines.

| method | nDCG@10 | 95% CI | Recall@10 | MRR@10 | mean lat |
|---|---|---|---|---|---|
| vector-only (bge-base) | 0.740 | [0.696, 0.779] | 0.874 | 0.703 | 17ms |
| BM25 (`core.bm25`) | 0.658 | [0.614, 0.703] | 0.785 | 0.622 | 8ms |
| hybrid (vector + BM25, RRF) | 0.719 | [0.676, 0.759] | 0.841 | 0.686 | 23ms |
| **hybrid + bge-reranker** | **0.745** | [0.707, 0.783] | 0.864 | 0.713 | 3376ms |

**Comparison to published baselines (BEIR scifact, nDCG@10):** BM25 ≈ 0.665 (BEIR paper), ColBERTv2 ≈ 0.694, BM25+cross-encoder ≈ 0.688, bge-base-en-v1.5 ≈ 0.741 (reference), E5-mistral ≈ 0.760. bert's BM25 (0.658) matches the published BM25; the full **hybrid + cross-encoder stack (0.745) is +0.080 over BM25** and in line with strong 2024 dense retrievers.

## Two honest findings

1. **The embedder swap is the bulk of the win.** Replacing the 2020 all-MiniLM-L6-v2 (384-dim) with bge-base-en-v1.5 (768-dim, asymmetric query instruction) moved vector-only from **0.645 → 0.740** — now matching the published bge-base reference (0.741). The 2020 encoder, not the fusion logic, was the ceiling.

2. **On scifact, naive RRF slightly drags the strong dense signal.** bge-base alone (0.740) beats the vector+BM25 RRF fusion (0.719) because BM25 (0.658) is much weaker here and RRF weights both signals equally. The cross-encoder rerank is what recovers it: reranking the fused top-30 pool lifts the full stack to **0.745**, edging past vector-only. This is a real, dataset-dependent result — fusion helps where the sparse and dense signals are closer in strength; the multi-dataset companion (`b2_beir_multi.py`, scifact + nfcorpus + fiqa) shows where.

## The reranker now actually runs

In the previous run the bge-reranker-v2-m3 stage **OOM'd** on 18 GB unified memory (MPS held ~18 GB; the cross-encoder defaults to an 8192-token window and the un-batched `predict()` allocated a huge attention buffer). `predict()` raised, `rerank()` returned `[]`, and `hybrid_with_rerank` silently degraded to `hybrid_no_rerank`. Fixed in `core/reranker.py` by bounding `max_length` (512) + `batch_size` (32), both env-tunable, with a clear-MPS-cache + `batch_size=1` retry on OOM. The reranked row above is now real. Note its latency (~3.4s/query): the cross-encoder is the accuracy/latency knob, not free.

**What this adds to the custom B7–B9 evals:** BEIR validates the retrieval *stack quality* on standard data and a public metric. The custom B9 + the full-context wall validate the *production wiring* and the *beyond-the-window* regime that BEIR's short passages can't exercise. Together: "is the retriever good?" (BEIR, comparable to published) **and** "does it matter in the product?" (B9, the long-context wall).
