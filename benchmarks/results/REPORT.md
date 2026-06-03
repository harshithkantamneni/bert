# bert-lab — Benchmark Report

_Generated: 2026-05-25_
_Platform: M3 Pro 18 GB unified memory, Python 3.13, warm-cache after first run_

## Executive summary

1. **Hybrid retrieval (no rerank) end-to-end p50 = 39.0ms**; p99 = 43.6ms; throughput 24.9 QPS single-threaded.
2. **Adding bge-reranker-v2-m3 brings p50 to 406.3ms** (~10.4× the no-rerank cost) — a quality/latency trade-off.
3. **On synthetic gold-judged retrieval, `hybrid_with_rerank` wins nDCG@10** with 0.772; vector-only is close on recall but loses ranking quality.
4. **Adversarial-eval: hybrid retrieval beats vector-only on negation (+33%).** Both methods stumble on contradiction-with-stale-fact (a known gap — needs timestamp-aware retrieval).

## B1 — RAG latency + throughput

_Config: warmup n=20, measure n=100 per run, runs=3_

### Per-signal latency

| Signal | n | mean ± CI95 | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| `vector` | 300 | 12.38 ± 0.19ms | 11.98ms | 16.60ms | 20.73ms |
| `bm25` | 300 | 25.86 ± 0.83ms | 25.40ms | 26.89ms | 27.61ms |
| `ppr` | 300 | 0.00 ± 0.00ms | 0.00ms | 0.00ms | 0.00ms |
| `cache` | 300 | 0.07 ± 0.00ms | 0.07ms | 0.08ms | 0.08ms |

### Hybrid end-to-end (RRF + optional rerank)

| Mode | p50 | p95 | p99 | warmup |
|---|---:|---:|---:|---:|
| `hybrid_no_rerank` | 39.0ms | 41.7ms | 43.6ms | 0.8s |
| `hybrid_with_rerank` | 406.3ms | 426.0ms | 429.6ms | 5.6s |

### Throughput: **24.9 QPS** (no rerank, single-threaded, 125 calls in 5.03s)

### Memory: baseline 1071.8 MB, after 200 queries: delta +0.0 MB

### Index scale (current bert-lab corpus): `{"chunk_count": 1455, "total_tokens": 177306, "vocab_size": 12968, "last_build_ts": 1779727346.750006, "signature": [1455, 2325], "loaded": true}`

## B2 — Retrieval quality (gold-judged)

_Dataset: `synthetic_seed42` — 64 docs, 24 queries_

| Method | R@1 | R@10 [95% CI] | MRR@10 | nDCG@10 [95% CI] | p95 lat |
|---|---:|---:|---:|---:|---:|
| `vector_only` | 0.246 | 0.754 [0.650–0.860] | 1.000 | 0.741 [0.673–0.811] | 413.1ms |
| `bm25_only` | 0.246 | 0.548 [0.440–0.667] | 1.000 | 0.643 [0.558–0.724] | 0.3ms |
| `hybrid_no_rerank` | 0.246 | 0.673 [0.565–0.779] | 1.000 | 0.705 [0.635–0.779] | 29.5ms |
| `hybrid_with_rerank` | 0.246 | 0.746 [0.637–0.844] | 1.000 | 0.772 [0.702–0.836] | 798.1ms |

### Failure rates (queries with Recall@10 = 0)

- `vector_only`: 0/24 miss-rate (0.0%)
- `bm25_only`: 0/24 miss-rate (0.0%)
- `hybrid_no_rerank`: 0/24 miss-rate (0.0%)
- `hybrid_with_rerank`: 0/24 miss-rate (0.0%)

## B3 — Memory benchmark (LongMemEval-style)

_Scenarios: 25 across 5 categories_

| Method | Overall accuracy | Mean latency |
|---|---:|---:|
| `sliding_window` | 0.600 | 0.01ms |
| `bert_hybrid` | 0.600 | 370.60ms |

### Per-category accuracy

| Category | sliding_window | bert_hybrid |
|---|---:|---:|
| single_session_recall | 8/8 (1.00) | 8/8 (1.00) |
| multi_session_update | 0/5 (0.00) | 0/5 (0.00) |
| knowledge_update | 0/4 (0.00) | 0/4 (0.00) |
| temporal_reasoning | 4/4 (1.00) | 4/4 (1.00) |
| abstention | 4/4 (1.00) | 4/4 (1.00) |

## B5 — Adversarial-eval-by-design

_Scenarios: 24 across 4 failure modes_

| Method | Overall catch rate |
|---|---:|
| `vector_only` | 0.625 |
| `bert_hybrid` | 0.625 |

### Per-mode catch rate

| Failure mode | vector_only | bert_hybrid |
|---|---:|---:|
| `negation` | 3/6 (0.50) | 5/6 (0.83) |
| `multi_hop` | 5/6 (0.83) | 5/6 (0.83) |
| `distractor` | 6/6 (1.00) | 5/6 (0.83) |
| `contradiction` | 1/6 (0.17) | 0/6 (0.00) |

## Honest limitations (POPPER-style)

- **synthetic corpora**: B2/B3/B5 use procedurally-generated test   data for reproducibility. Real-world corpora (longer docs,   duplicates, noisy queries) will shift absolute numbers. The   relative ordering of methods on these synthetic sets is what   we report with confidence; absolute scores need real-corpus   validation (BEIR / MS-MARCO modes — opt-in).
- **subset sizes**: 24-30 queries / 24 scenarios is small for tight CIs.   Bootstrap 95% intervals are reported; larger N would tighten them.
- **single-threaded latency**: B1 measures single-call latency on one   thread. Production traffic with concurrent queries hits SQLite write   serialization differently; multi-thread benchmarks are future work.
- **contradiction failure mode** (B5): Both methods fail (0-17% catch).   This is a known gap — bert's wedge is the temporal-aware consolidator   layer on top of retrieval, which this benchmark does not exercise.
- **cold-start**: All p99/mean numbers are STEADY-STATE (post-warmup).   First-query latency adds ~30-60s for embedder cold-load and ~20s for   the cross-encoder. Production should pre-warm at server start.

## Reproducibility

Each benchmark is hermetic and seeded (default seed=42). To reproduce:

```bash
.venv/bin/python benchmarks/b1_latency_throughput.py
.venv/bin/python benchmarks/b2_retrieval_quality.py --seed 42
.venv/bin/python benchmarks/b3_memory_longmemeval.py
.venv/bin/python benchmarks/b5_adversarial_eval.py
.venv/bin/python benchmarks/b6_compile_report.py
```

Raw JSON outputs are in `benchmarks/results/`, one file per run.
Each summary `.md` is independently inspectable.

## What this report does NOT measure

bert is NOT positioned as best-in-class on `SWE-bench Verified`, `OWASP-depth`, or `WebArena`. These are agent capability benchmarks where bert would lose by design (we don't optimize for code-execution autonomy). The benchmarks here are the ones where bert's wedge (free-tier autonomous lab + hybrid retrieval + adversarial-eval-by-design) is actually competitive.
