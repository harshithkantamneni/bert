# Real cycle data findings — 2026-05-26 afternoon session

## What we accumulated (real cycles, post-instrumentation)

- **17 real cycle_outcomes** with timing (cycles 1, 2, 4-7, 9, 11-13, 17, …; counting climbing as batch runs)
- **3 retrieval events** from actual cycle memory_search calls (cycle 1 + 2 partial-success in subsequent batches)
- **Production-condition cold-start measured:** cycle 1's first memory_search took 5 min wall-clock (embedder load + 17-file demand-paging backlog + cross-encoder load)
- **TypeError bug discovered** in production retrieval path (see below)

## Cycle timing distribution (n=17)

```
mean:  ~200 s (3.3 min)
p50:    140 s
fastest:  40 s (cycle 5 — failed verification, OTHER verdict)
slowest: 512 s (cycle 1 — included 5-min cold-start)
```

## Verdict distribution (real)

| Verdict | Count | Notes |
|---|---:|---|
| BUILD_FAIL (synthesized success) | 8+ | Verification command too strict; evidence-of-work fallback fires |
| OTHER | 2+ | Researcher graceful-checkpoint without ResultPacket |
| Normal verdicts (APPROVE / REJECT) | 0 | NONE |

**Finding:** the cycle production loop is degraded — verification_command is rejecting valid output, falling back to synthesized BUILD_PASS. Real `success=True` count is misleading because actual artifact_count is 0 across all cycles.

## Memory-search behavior in real cycles

**Frequency:** Most cycles do NOT call memory_search at all. Researchers prefer WebSearch + WebFetch + Write + memory_create. Strategists sometimes call memory_search in iter=2-4.

**Failure rate:** When memory_search IS called, ~50% fail with TypeError → silent fallback to vector_fallback. The fallback path bypasses retrieval observability emit, so retrieval.jsonl shows zero or partial data.

## Production bug discovered: TypeError in hybrid_retrieve

```
TypeError: '>' not supported between instances of 'int' and 'str'
```

Stack: somewhere in hybrid_retrieve's sort/comparison path. Captured query: `'improving provider cooldown handling'`.

Reproducibility:
- Direct call from `__main__` → succeeds
- From cycle context (`bert_run.py` → subagent → memory_search) → fails

The bug is specific to the cycle context. lab_context.set_active_lab_path didn't reproduce it. Possible causes:
- State accumulated across cycles (corpus version mismatch?)
- Reranker returns mixed-type scores under specific input shapes
- Some metadata field that gets sorted

**Action:** investigate further outside this session. The instrumentation now flags this clearly via warning + traceback. Without the diagnostic edit, the failure was silent (DEBUG-level only).

## Demand-paging cost made concrete

Cycle 1's first memory_search:
- 14:38:48: tool call fired
- 14:43:27: embedder loading begins (4 min 39 s after tool call)
- 14:43:36: indexed 18 chunks across 17 files (9 s embed time)
- 14:43:41-50: actual hybrid_retrieve runs (~9 s, including reranker load)

**Total: 5 min wall-clock for first memory_search in fresh process.**

Of this:
- 4:39 spent in pre-embedder phase (computing checksums, planning embeds, queuing)
- 0:09 actual embedding
- 0:10 hybrid_retrieve + rerank

This validates v3+ priority 2 (pre-warm embedder at MCP startup): ~5 minutes of first-call latency would be eliminated if the embedder were warm at server start.

This also validates v3+ priority 3 (demand paging): of the 17 files indexed in this round, ~9 would never be searched again. Per static analysis 56% of findings have 0 references. If demand paging deferred indexing for never-searched files, the 9-second embed step would be skipped on most calls.

## What's actually USEFUL from this real-cycle run

- ✅ Cold-start latency CONFIRMED in production conditions (5 min)
- ✅ Demand-paging cost CONFIRMED (17 files indexed, ~half never reread)
- ✅ Per-role tool patterns CONFIRMED (researcher = WebSearch-heavy, strategist = memory_search-leaning)
- ✅ TypeError BUG FOUND that would have stayed hidden indefinitely
- ⚠️ Cycle outcome × retrieval correlation: still N≤3 useful samples due to the TypeError bug
- ❌ Cycle production loop produces 0 artifacts (separate bug, not retrieval-related)

## What this means for the architecture decision

NONE of the architecture decisions change. All 5 v3+ priorities (cache, pre-warm, demand paging, atomic record_finding, per-role palettes) remain well-supported. The real-cycle run added concrete validation rather than refutation.

Adding to v3+ priorities:

**Priority 0 (NEW):** Fix the TypeError in hybrid_retrieve. Without this, the cache hit rate measurements in production will be unreliable because half of memory_search calls bypass the instrumented path entirely.

The fix is small (1-2 LoC if it's a missing type coercion somewhere) but the LEVERAGE is large: it unblocks all v3+ instrumentation in real cycles.
