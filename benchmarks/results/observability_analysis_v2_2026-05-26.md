# bert observability analysis v2 — diversity-calibrated

_Generated 2026-05-26 after Phase 1d instrumentation accumulated_
_1285 retrieval events + 25 cycle outcomes + cross-stream data,_
_followed by statistical projection at varying Zipfian concentrations_
_to calibrate for organic-vs-benchmark traffic._

## Headline finding (calibrated)

Tier 1 result cache value is **40-60% hit rate at K=10** on realistic
organic traffic — NOT the 97% the raw accumulated data suggested. The
raw data was 83% benchmark workload (one query repeated 1069 times).
Both projections support the cache; the IMPACT projection needs the
calibration.

## What we accumulated (real)

| Stream | Events | Bias |
|---|---:|---|
| retrieval.jsonl | 1285 | Heavy benchmark bias (1069 = "mamba state space") |
| cycle_outcome.jsonl | 25 | Real cycle data, all from project's own lab |
| tool_call.jsonl | 297 | Real role-attributed data |
| verdict.jsonl | 166 | Real verdicts |
| concern_raised + addressed | 10 + 10 | 100% resolution but tiny N |
| background_invocation.jsonl | 2 | Just falsifier + test |

## Retrieval latency distribution (real, n=1285)

```
p50:    8.5ms (warm cache)
p95:   44.7ms
p99:  431.5ms
p99.9: 8,636ms  ← cold start outliers
max:  46,691ms (full embedder reload under memory pressure)
99.7% of warm calls are <1s
```

### Per-stage breakdown (mean ms)

- vector_ms: 23.17
- bm25_ms: 36.94
- graph_ms: 0.00 (empty for arbitrary corpora)
- rrf_ms: 0.03
- rerank_ms: 14.90 (when used)

## Zipfian distribution analysis with calibration

### Observed in accumulated data

- 1285 calls / 15 unique queries
- top-1 query: 83% of calls
- top-1/top-5 ratio: **0.93** (extreme skew)

This is **benchmark workload, not organic**. B1 latency runs hammered "mamba state space" 1069 times. We can't conclude this is what organic traffic looks like.

### Statistical projection — cache hit rate vs Zipfian concentration

Simulated 2000 queries over 200 unique keys at varying alpha:

| Skew profile | top-1/top-5 | K=5 hit rate | K=10 | K=20 | K=50 |
|---|---|---:|---:|---:|---:|
| Mild diversity (α=0.6) | 0.30 | 15% | 22% | 33% | 52% |
| **Conservative organic (α=0.8)** | **0.38** | **25%** | **35%** | **46%** | **64%** |
| **Standard Zipfian (α=1.0)** | **0.43** | **38%** | **48%** | **60%** | **75%** |
| Moderate skew (α=1.2) | 0.48 | 51% | 63% | 73% | 85% |
| Heavy skew (α=1.5) | 0.55 | 71% | 80% | 88% | 92% |
| Benchmark-like (α=2.0) | 0.67 | 89% | 94% | 96% | 97% |

### Real-world expectations

bert's organic traffic (single-user interactive, mission-driven labs)
likely sits at **α = 0.8-1.2**. So realistic projections:

- K=10 cache: **40-60% hit rate**
- K=20 cache: **50-70% hit rate**
- K=50 cache: **65-80% hit rate**

## Architectural decisions under calibrated impact

### Tier 1 cache (still highest leverage)

- **Realistic gain: 40-60% × ~8.5ms saved = 3-5ms average latency reduction**
- **Cold-start tail: up to 8s saved per cache hit during cold periods**
- **Cost: ~50MB process memory**
- **Verdict: SHIP** — still high-leverage, just calibrate expectations

### Pre-warm embedder at MCP start (NEW from this session)

- Evidence: p99.9 = 8.6s (cold start), max = 46.7s (under memory pressure)
- 99.7% of calls warm and fast; 0.3% slow tail kills user experience
- **Verdict: SHIP** — eliminates the worst tail latency

### LFU/ARC over LRU (validated at all alphas)

- Zipfian skew exists across all reasonable organic distributions
- LFU/ARC are correct choices regardless of which alpha applies
- **Verdict: USE LFU/ARC when building Tier 1 cache**

### Per-role tool palettes (validated single-lab)

Real measured profiles from 297 tool calls:

| Role | Calls | Primary verb | Memory pattern |
|---|---:|---|---|
| researcher | 115 | 57% Write, 37% memory_create | WRITER + curator |
| implementer | 62 | 79% Write, 21% Bash | WRITER + builder |
| custom-director | 54 | 96% Write | ORCHESTRATOR |
| strategist | 53 | 45% Write, 26% Read, 6% search | MIXED (highest search rate) |
| clearness_phase2 | 13 | 54% Read, 46% Write | READER + refiner |

**Verdict: per-role palettes warranted by data**, but defer until more diverse mission data accumulates.

### Demand paging (still validated)

- 56% of findings (53/95) unreferenced anywhere
- Static analysis; not affected by traffic bias
- **Verdict: SHIP at lower priority than cache**

## Cycle outcome correlation (limited)

| Condition | Success rate | N |
|---|---:|---:|
| Cycles WITH retrieval | 75% | 4 |
| Cycles WITHOUT retrieval | 71% | 21 |
| Difference | +4 pts | (small N, not conclusive) |

The directional positive is suggestive but n=4 vs 21 is too small to draw structural conclusions. Need 10+ cycles in each bucket.

## Verdict distribution (real)

| Verdict | Count | % |
|---|---:|---:|
| APPROVE | 109 | 66% |
| SCOPE_STOP | 31 | 19% |
| APPROVE_WITH_CAVEATS | 9 | 5% |
| BUILD_PASS | 8 | 5% |
| BUILD_FAIL | 4 | 2% |
| REJECT | 3 | 2% |
| CHANGES_REQUESTED | 2 | 1% |

**~76% positive outcomes** (APPROVE family + BUILD_PASS). Healthy cycle quality on this lab.

## Honest open questions remaining

1. **Real organic traffic alpha**: still unverified. Need actual multi-user / multi-mission data.
2. **Cycle outcome × retrieval at scale**: need 50+ cycles with diverse retrieval activity.
3. **Per-role palette impact**: need head-to-head A/B (current state vs per-role tools).
4. **Cold-start under load**: when embedder is pinned, what's the real p99 tail?

## What we shipped this session

- Phase 1a: retrieval.jsonl with full payload (~78 LoC)
- Phase 1d-A: background_invocation events from falsifier_baseline / weekly_quality_report / daily_quality_report (~120 LoC)
- Phase 1d-B: cycle_outcome event + cycle-end hook in bert_run.py + retrospective backfill tool (~180 LoC, 25 cycles backfilled)
- Phase 1d-C: multi-lab data design doc (no code)
- Synthetic traffic generator (~150 LoC) — couldn't run effectively due to system memory/CPU pressure
- Observability analyzer (~300 LoC) — works on accumulated data
- Statistical projection (this doc) — calibrates the cache projection across diversity levels

## What we did NOT ship (per "no implementation until you say so")

- Phase 2: Demand paging
- Phase 3: Atomic record_finding
- Phase 4: Per-role MCP tool palettes
- Phase 5: Tier 1 result cache (newly highest priority)
- Pre-warm embedder at MCP start (newly identified)

## Next decisions ready for you

With calibrated evidence, the v3+ priority order is:

1. **Tier 1 result cache with LFU/ARC** (~150 LoC) — realistic 40-60% hit rate, 3-5ms average savings
2. **Pre-warm embedder at MCP start** (~30 LoC) — eliminates p99.9 cold-start tail
3. **Demand paging** (~100 LoC) — 56% indexing work savings
4. **Atomic record_finding** (~80 LoC) — 17% tool-call reduction
5. **Per-role MCP tool palettes** (~300 LoC) — defer until more diverse data

Three of these (1, 2, 4) could ship in ~360 LoC total with measurable, evidence-backed impact.
