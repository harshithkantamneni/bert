# bert observability analysis

_Generated from data in `/path/to/Desktop/bert-lab/state/observability` — 26799 total events analyzed_

## Event stream sizes

- retrieval.jsonl: 26304
- cycle_outcome.jsonl: 25
- background_invocation.jsonl: 2
- tool_call.jsonl: 301
- verdict.jsonl: 167

## Retrieval latency distribution

- Total calls: **26304**
- p50: **0.8ms**, p95: 1.5ms, p99: 16.4ms
- min: 0.3ms, max: 46691.0ms, mean: 4.5ms

### Per-stage breakdown (mean ms)

- vector_ms: 1.13ms (mean of 26304 calls)
- bm25_ms: 2.55ms (mean of 26304 calls)
- graph_ms: 0.00ms (mean of 26304 calls)
- rrf_ms: 0.01ms (mean of 26304 calls)
- rerank_ms: 0.73ms (mean of 26304 calls)

## Query frequency distribution (Zipfian check)

- Total queries: 26297
- Unique queries: 334
- Repeat rate: 98.7%
- top-1 query: 1994 times (7.6% of all calls)
- top-5 queries hold 23.8% of all calls
- top-10 queries hold 32.8% of all calls
- top-1 / top-5 ratio: 0.32 (> 0.4 = strong Zipfian → LFU/ARC preferred over LRU)

### Hottest 5 queries

- `Velero backup restore` — 1994 hits (7.6%)
- `dense vector index quantization` — 1683 hits (6.4%)
- `mamba state space` — 1069 hits (4.1%)
- `Phoenix Arize tracing` — 790 hits (3.0%)
- `cache coherence invalidation` — 732 hits (2.8%)

## Signal contribution to final top-K

- Queries with results: 25236
- Queries where final top-K had multiple signals: 1282 (5.1%)

### How often each signal appears in final top-K

- bm25: 194564 occurrences across all queries (98.1%)
- vector: 3796 occurrences across all queries (1.9%)

## Tier 1 result cache potential

- Total queries: 26297
- Unique queries: 334

### Hit rate by cache size (LFU eviction)

- K=  5: hit rate **23.8%** (6263 hits of 26297 calls)
- K= 10: hit rate **32.7%** (8603 hits of 26297 calls)
- K= 20: hit rate **42.8%** (11253 hits of 26297 calls)
- K= 50: hit rate **56.9%** (14974 hits of 26297 calls)
- K=100: hit rate **70.7%** (18595 hits of 26297 calls)
- K=200: hit rate **86.5%** (22760 hits of 26297 calls)

## Latency by query characteristics

- medium (21-60): n=21784, p50=0.8ms, mean=0.8ms
- short (≤20 chars): n=4520, p50=0.8ms, mean=22.0ms

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
