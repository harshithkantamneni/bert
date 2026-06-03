# B9 — Long-context RAG benchmark result

_Platform: M3 Pro 18GB. Corpus: httpx-0.28.1 + starlette (vendored), ~131K tokens, 57 files → 438 chunks._
_Reader: llama-3.3-70b (free, bert's runtime). Grader: mistral-large + deepseek-v4-pro (non-Claude). 20 span-validated gold questions._

## Headline

Over a 131K-token corpus with a 15K truncation window (so the corpus is ~8.7× the window), retrieval holds answer quality flat at **~4.6× fewer input tokens** than truncation:

| arm | accuracy | input tokens | retrieval recall@10 |
|---|---|---|---|
| A1 naive truncation (15K) | 0.10 | 15,000 | — |
| A2 smart truncation (manifest+heads, 15K) | 0.35 | 14,709 | — |
| A4 vector-RAG | 0.70 | 2,905 | 0.692 |
| **A3 hybrid-RAG (vector+BM25+rerank)** | **0.85** | **3,278** | **0.783** |

By tier (the needle tier is the thesis — one line buried in 131K tokens):

| tier | A1 | A2 | A4 vector | A3 hybrid |
|---|---|---|---|---|
| single-hop | 0.00 | 0.40 | 0.80 | 0.80 |
| multi-hop | 0.29 | 0.43 | 0.71 | **0.86** |
| **needle** | **0.00** | **0.25** | 0.63 | **0.88** |

**Conclusion:** when the relevant context exceeds the window, RAG is both *more accurate and far cheaper* than stuffing what fits. Truncation drops the needle (0.00–0.25); retrieval finds it (0.88). Hybrid (RRF + bge-reranker) beats plain vector on multi-hop (0.86 vs 0.71), as designed.

## The bug this benchmark caught (and fixed)

The first run scored bert's flagship **hybrid** retriever at **0.10 accuracy / 0.125 recall — 5× worse than plain vector**. Root cause (core/retrieval.py): the vector candidate builder read `r["text"]/["id"]/["score"]` but `memory.search` returns `{path, chunk_idx, content, distance}` — so vector candidates got **empty text + index ids**, silently zeroing the vector signal in RRF fusion; and both vector/BM25 candidates truncated content to **240 chars**, dropping the answer span on real (1500-char) code chunks. BEIR's short passages never exposed this, and the BEIR benchmark used a *separate* reimplemented retrieval path — so the production `hybrid_retrieve` was broken while the prior benchmark "passed." Fix: read the right keys + carry full chunk content. Re-measure: hybrid **0.10 → 0.85 accuracy, 0.125 → 0.783 recall**.

## Honest limitations
- Free llama reader caps absolute accuracy; the *contrast* (RAG vs truncation) is the result, not the absolute numbers. A stronger reader (Opus, Phase 2) would lift all arms; the cost advantage of RAG is reader-independent.
- n=20 (single-hop 5 / multi-hop 7 / needle 8) — directional, not significance-tested. Self-truncation window (15K) is a controllable stand-in for a real context limit.
- A0 full-context-stuffed (the ceiling + the >window infeasibility wall) is deferred to the paid Opus phase (131K exceeds the free reader's practical window).
- Gold set is blind-authored + adversarially reviewed + every span verbatim-validated against the corpus, but the corpus is two libraries; broader corpora would strengthen generality.

## The full-context wall (Max-only, 2026-06-02)

Same httpx/starlette gold needles, corpus padded with numpy+sympy to cross the 1M window. A0 reader = Max Opus 1M bridge (`claude -p`, free on subscription); RAG/truncation = free llama. Cost axis = tokens (Max $ is imputed).

| corpus | A0 full-context | A1 truncation (15K) | A3 bert-RAG |
|---|---|---|---|
| 132K (fits 1M window) | **acc 1.00** @ 132K tok | — | 0.50 @ 3.5K tok (n=2) |
| **3.0M (exceeds window)** | **INFEASIBLE** (3.04M > 1M) | **0.00** @ 15K tok (n=8) | **0.75** @ 3.3K tok (n=8) |

**Conclusion:** below the window full-context is perfect (RAG unnecessary); above it, full-context cannot run at all, truncation scores 0.00 (needle past the cut), and retrieval is the only working option — at a **flat ~3.3K input tokens regardless of corpus size (132K → 3M)**. A0 input-tokens-per-query climbs linearly then hits a vertical wall at 1M; RAG is flat. This is the regime (project > context window) where the product is uniquely necessary, not merely cheaper.
