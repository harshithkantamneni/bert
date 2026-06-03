# bert observability analysis

_Generated from data in `/path/to/Desktop/bert-lab/state/observability` — 11796 total events analyzed_

## Event stream sizes

- retrieval.jsonl: 11301
- cycle_outcome.jsonl: 25
- background_invocation.jsonl: 2
- tool_call.jsonl: 301
- verdict.jsonl: 167

## Retrieval latency distribution

- Total calls: **11301**
- p50: **0.8ms**, p95: 8.8ms, p99: 30.1ms
- min: 0.3ms, max: 46691.0ms, mean: 9.3ms

### Per-stage breakdown (mean ms)

- vector_ms: 2.64ms (mean of 11301 calls)
- bm25_ms: 4.91ms (mean of 11301 calls)
- graph_ms: 0.00ms (mean of 11301 calls)
- rrf_ms: 0.01ms (mean of 11301 calls)
- rerank_ms: 1.69ms (mean of 11301 calls)

## Query frequency distribution (Zipfian check)

- Total queries: 11294
- Unique queries: 333
- Repeat rate: 97.1%
- top-1 query: 1616 times (14.3% of all calls)
- top-5 queries hold 38.6% of all calls
- top-10 queries hold 49.2% of all calls
- top-1 / top-5 ratio: 0.37 (> 0.4 = strong Zipfian → LFU/ARC preferred over LRU)

### Hottest 5 queries

- `dense vector index quantization` — 1616 hits (14.3%)
- `mamba state space` — 1069 hits (9.5%)
- `Phoenix Arize tracing` — 763 hits (6.8%)
- `Kubernetes pod scheduling` — 503 hits (4.5%)
- `QuestDB high-cardinality TS` — 405 hits (3.6%)

## Signal contribution to final top-K

- Queries with results: 10689
- Queries where final top-K had multiple signals: 1282 (12.0%)

### How often each signal appears in final top-K

- bm25: 78539 occurrences across all queries (95.4%)
- vector: 3796 occurrences across all queries (4.6%)

## Tier 1 result cache potential

- Total queries: 11294
- Unique queries: 333

### Hit rate by cache size (LFU eviction)

- K=  5: hit rate **38.5%** (4351 hits of 11294 calls)
- K= 10: hit rate **49.1%** (5542 hits of 11294 calls)
- K= 20: hit rate **59.0%** (6658 hits of 11294 calls)
- K= 50: hit rate **71.8%** (8108 hits of 11294 calls)
- K=100: hit rate **81.7%** (9228 hits of 11294 calls)
- K=200: hit rate **91.9%** (10382 hits of 11294 calls)

## Latency by query characteristics

- medium (21-60): n=8823, p50=0.8ms, mean=0.9ms
- short (≤20 chars): n=2478, p50=0.9ms, mean=39.4ms

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
