# Tier-1 retrieval result cache — measure-first decision (2026-05-30)

**Decision: do NOT build the Tier-1 retrieval result cache now. Defer until (a)
real organic multi-lab traffic exists to re-measure, and (b) the `corpus_version`
coherence layer is built (a correctness prerequisite).**

The v3+ plan promoted a Tier-1 (LFU/ARC) retrieval result cache to priority 1 on
a projected **33–68%** hit rate. Per the design's own "measure before each phase"
rule, this is the measurement that gates the build — and it does not justify it.

## Data

`state/observability/retrieval.jsonl` — 7,445 retrieval calls with a `query`
field. Replayed the real query sequence through an LFU result cache.

## Finding 1 — the projection was benchmark-inflated

| Cache | Full sequence (incl. benchmark) | Organic (drop queries seen >20×) |
|---|---|---|
| LFU K=10 | 51.1% | **5.3%** |
| LFU K=20 | 57.0% | 10.6% |
| LFU K=50 | 66.0% | 24.4% |

The full-sequence numbers match the v3+ projection, but they are dominated by
benchmark/test traffic: `"test query"` ×3000, `"warmup"`, `"x"`, `"anything"`,
plus repeated BEIR-style eval suites. Stripping the obvious benchmark queries
collapses the K=10 hit rate from 51% to **5.3%**.

## Finding 2 — robust across thresholds + a realistic TTL model

Threshold sensitivity (LFU K=20): drop >1000× → 28% · >50× → 15% · >20× → 11% ·
>10× → 7%. Monotone with benchmark pollution.

Windowed/TTL model (a result cache expires entries — hit only if the same query
recurred within the last W calls), organic subset: **W=10 → 4.2%**, W=50 → 19%,
W=100 → 33%, W=200 → 53%. A meaningful organic hit rate needs a ~100–200-entry
cache.

## Finding 3 — organic repeats are sporadic, not cacheable at small sizes

Gaps between repeats of mid-frequency organic queries:
- `"ZeRO zero redundancy optimizer"` (4×): gaps [1812, 94, 707]
- `"LangGraph state machine"` (8×): gaps [915, 375, 760, 28, 282, 539, 453]
- `"Brooks's law mythical man-month"` (8×): gaps [1314, 321, 728, 41, 385, 878, 89]

Repeats recur across the whole sequence with huge, irregular gaps — not a tight
loop a small cache catches. Caching through an 1812-gap needs an 1812-entry cache.

## Why this kills the build (for now)

1. **Weak organic win.** 4–19% at reasonable cache sizes (K/W = 10–50), vs the
   48–97% projection. The projection rested on benchmark traffic.
2. **Correctness prerequisite missing.** A result cache returns stale results
   when the corpus changes (a finding is written). A correct cache needs
   `corpus_version` invalidation — still unbuilt (mtime coherence is per-file at
   index time, not a retrieval-result epoch). The proof-packet-determinism
   invariant (cached results must not silently diverge from a re-run) adds risk.
3. **The high-value latency wins are already in place.** Cold-start (the 8.6s
   p99.9 tail) is handled by embedder pre-warm (`dd5b680`); LLM-call dedup is
   handled by the existing `semantic_cache`. The retrieval result cache is the
   one item the data says skip.

## Re-open conditions

Re-measure when: real multi-lab organic retrieval traffic exists (today's log is
benchmark/eval-dominated, single-lab); AND `corpus_version` lands. If a re-measure
shows organic hit rate >30% at K≤50 AND the cache can be made coherence-correct,
build it then.

Cross-refs: `project_bert_memory_arch_v3.md` (priority list), the earlier
`observability_analysis_v3_2026-05-26.md` (which reported the benchmark-inflated
97%/46% numbers this measurement corrects).
