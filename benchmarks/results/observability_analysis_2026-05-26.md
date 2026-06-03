# bert observability analysis

_Generated from data in `/path/to/Desktop/bert-lab/state/observability` — 1775 total events analyzed_

## Event stream sizes

- retrieval.jsonl: 1285
- cycle_outcome.jsonl: 25
- background_invocation.jsonl: 2
- tool_call.jsonl: 297
- verdict.jsonl: 166

## Retrieval latency distribution

- Total calls: **1285**
- p50: **8.5ms**, p95: 44.7ms, p99: 431.5ms
- min: 5.4ms, max: 46691.0ms, mean: 75.8ms

### Per-stage breakdown (mean ms)

- vector_ms: 23.17ms (mean of 1285 calls)
- bm25_ms: 36.94ms (mean of 1285 calls)
- graph_ms: 0.00ms (mean of 1285 calls)
- rrf_ms: 0.03ms (mean of 1285 calls)
- rerank_ms: 14.90ms (mean of 1285 calls)

## Query frequency distribution (Zipfian check)

- Total queries: 1282
- Unique queries: 14
- Repeat rate: 98.9%
- top-1 query: 1069 times (83.4% of all calls)
- top-5 queries hold 89.7% of all calls
- top-10 queries hold 97.5% of all calls
- top-1 / top-5 ratio: 0.93 (> 0.4 = strong Zipfian → LFU/ARC preferred over LRU)

### Hottest 5 queries

- `mamba state space` — 1069 hits (83.4%)
- `cross-encoder rerank` — 21 hits (1.6%)
- `test` — 20 hits (1.6%)
- `attention transformer` — 20 hits (1.6%)
- `BM25 sparse` — 20 hits (1.6%)

## Signal contribution to final top-K

- Queries with results: 1282
- Queries where final top-K had multiple signals: 1282 (100.0%)

### How often each signal appears in final top-K

- vector: 3796 occurrences across all queries (59.3%)
- bm25: 2606 occurrences across all queries (40.7%)

## Tier 1 result cache potential

- Total queries: 1282
- Unique queries: 14

### Hit rate by cache size (LFU eviction)

- K=  5: hit rate **89.3%** (1145 hits of 1282 calls)
- K= 10: hit rate **96.7%** (1240 hits of 1282 calls)

## Latency by query characteristics

- short (≤20 chars): n=1204, p50=8.6ms, mean=80.2ms
- medium (21-60): n=81, p50=8.2ms, mean=10.6ms

## Cycle outcome correlation

- Total cycles graded: 25
- Successful: 18 (72.0%)
- With ≥1 artifact accepted: 11

### Retrieval activity by cycle outcome

- Cycles with ≥1 memory_search: 4/25 (16.0%)
- Success rate (cycles WITH retrieval): 75% (n=4)
- Success rate (cycles WITHOUT retrieval): 71% (n=21)

## Background tool footprint

- `falsifier_baseline`: 1 runs, avg duration 173.7s, 2 findings produced, success rate 100%
- `test_tool`: 1 runs, avg duration 0.0s, 1 findings produced, success rate 100%
