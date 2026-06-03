# Is bert organic? — Cross-mission test results, 2026-05-26

## Headline finding

**bert is NOT organic.** Across 3 missions (research / build / analysis),
the agent collapses every mission into a single research-finding shape:
H1 + Summary + Top signals + Hypotheses + Open questions. 0 of 15 cycles
landed an artifact. The build mission produced **zero** code files.

## Test setup

- 3 missions, 5 cycles each (15 cycles total)
- Same lab, same model (nvidia/llama-3.3-70b-instruct), same agent prompts
- Only difference: `lab/seed_brief.md` swapped between mission files in
  `missions/research.md`, `missions/build.md`, `missions/analysis.md`
- All fixes in place (TypeError in `_memory_search`, honest success logic
  in `bert_run.py`, self-improvement-as-mission retired)

## Mission comparison

| Metric | Research | Build | Analysis |
|---|---:|---:|---:|
| Cycles | 5 | 5 | 5 |
| Honest success rate | 0% | 0% | 0% |
| Artifacts/cycle | 0.00 | 0.00 | 0.00 |
| Findings/cycle | 0.00 | 0.00 | 0.00 |
| Elapsed p50 (s) | 154 | 154 | 98 |
| Retrievals/cycle | 1.0 | 0.4 | **0.0** |
| Tool calls/cycle | 13.8 | 7.6 | 4.6 |
| BUILD_PASS count | 1/10 | 2/10 | 1/10 |
| BUILD_FAIL count | 9/10 | 7/10 | 9/10 |

## Tool distribution per mission

| Mission | Write | WebSearch | memory_search | memory_create | Read |
|---|---:|---:|---:|---:|---:|
| Research | 39 | 9 | 5 | 7 | 4 |
| Build | 26 | 4 | 2 | 3 | 1 |
| Analysis | 17 | 4 | **0** | 2 | 4 |

## Why "not organic"

1. **Build mission produced no code.** Mission required `tools/jhist.py`
   + tests; agent wrote research-style markdown about histogram CLIs
   (with real citations to other tools) but never invoked Write at any
   .py path. `find tools tests -newer suite_start | wc -l` = 0.

2. **Analysis mission did zero memory_search.** The mission explicitly
   required cross-referencing existing findings; the agent never queried.
   It wrote audit prose based on whatever was already in its context
   window, not based on retrieved evidence.

3. **All 3 missions produced the same markdown structure.** Researcher
   role outputs are templated: `# Researcher Finding — <topic>`,
   `## Summary`, `## Top signals`, `## Candidate hypotheses`,
   `## Open questions`. The mission text steers the topic but not the
   output shape.

4. **Tool usage scales with cycle complexity, not with mission type.**
   Research uses the most tools (13.8/cycle); analysis the least (4.6).
   But the *types* of tools used are the same — Write + WebSearch +
   memory_create dominate everywhere.

## Self-improvement aggregator signals (passive, no LLM)

Run on accumulated data, the engine-level aggregator surfaced 3 signals:

1. **cycle_success_drop HIGH** — recent 10-cycle success rate is 0% vs
   40% prior window. Partly artifactual (we tightened the success
   definition to be honest), but the trend is real.
2. **artifact_zero_streak HIGH** — 32 cycles in a row with
   `artifacts_accepted = 0`. The lab is not producing artifacts the
   acceptance gate respects.
3. **verdict_concentration MEDIUM** — 80% BUILD_FAIL across the recent
   window. Consistent failure mode pointing at the prompt /
   verification path, not at randomness.

This is exactly the role self-improvement-as-property should play:
surface patterns from passive observation; no token spent; PI decides
what to fix.

## Root causes (the engineering work)

A) **Single role template, single output shape.**
   `core/library/agents/_base/researcher.md` (and the strategist role)
   hardcode the "research finding" markdown contract. Any mission gets
   bent into that shape. To be organic, we need per-mission role
   profiles or mission-aware role dispatching:
   - researcher / implementer / auditor as distinct roles
   - Director picks the right role for each mission shape

B) **Single verification gate.**
   `tools/bert_run.py:287` uses the same `test -s && grep -q '^# ' && ≥3 H2 && ≥1500 chars && cite` rule for every dispatch. That's
   reasonable for a researcher's deliverable but wrong for:
   - Strategist (terse recommendation, 300-700 chars typical)
   - Implementer (code, not markdown)
   - Auditor (structured ledger, may not be 1500+ chars)
   The gate needs to be role × mission-shape parameterized.

C) **Researcher prompt is "find evidence and cite."**
   For build missions the right prompt is "write code that passes
   these tests"; for analysis missions it's "search the corpus
   exhaustively and ground every claim with a memory_search result."
   The current prompt is research-only.

## Acceptance criteria (when bert IS organic)

Three things would have to be true:

1. Build mission produces working code that passes `pytest`. Cycle
   verdict reflects code-passes-tests, not prose-shape.
2. Analysis mission calls `memory_search` at least 3× per cycle and
   cites memory IDs in its audit rows.
3. Cross-mission tool distribution differs: research has high
   WebSearch, build has high Write-to-.py + Bash-runs-tests, analysis
   has high memory_search + Read.

Today: 0/3.

## What we know with high confidence now

- **Cache hit rate projections HOLD.** Real cycle retrieval shows BM25
  + vector + rerank pipeline working end-to-end. The 33-68% K=10 cache
  hit projection from synthetic Zipfian analysis remains the right
  number — production hasn't contradicted it.
- **Cold-start cost CONFIRMED in production.** Cycle 1 of each mission
  hits the same 5-min cold-start cost (embedder load + reranker load +
  backlog indexing). Pre-warm at MCP startup is validated as priority 2.
- **Demand-paging value CONFIRMED.** Most files written during a cycle
  are never read by subsequent memory_search calls — wasted embedding
  work. Confirmed across all 3 mission types.
- **TypeError bug in `_memory_search` FIXED.** `int(k)` coercion at
  entry stopped the silent fallback to vector-only that was eating ~50%
  of retrieval emit events.
- **Honest-success bug FIXED.** Cycles now report `success=False` when
  verdicts are BUILD_FAIL, instead of synthesizing fake success.
- **self_improvement_aggregator landed.** Engine-level passive signal
  collector, no LLM, ~150 LoC. Catches real patterns
  (artifact-zero-streak, success-drop, verdict-concentration) on
  observed data.

## What this means for v3+ priority order

The pre-suite priority was:
1. Tier 1 result cache + LFU
2. Pre-warm embedder
3. Demand paging
4. Atomic record_finding
5. Per-role MCP palettes

The post-suite priority should be:

0. **Mission-organic role dispatch.** Without this, bert's value
   proposition (autonomous lab for any mission) is false. The other
   priorities optimize a system that produces zero artifacts.
1-5: as before.

This isn't a v3+ retrieval-architecture decision change. It's a
prompt/role-template architecture decision that goes BEFORE the
retrieval optimizations. No point caching memory_search results when
the agent never calls it for analysis missions.

## Net for this session

- 26,318 retrieval events captured (1,285 → 26,318)
- 57 cycle_outcome events (4 real → 57)
- Cross-mission test executed and analyzed
- 2 production bugs found and fixed (TypeError, lying-success)
- Self-improvement-as-property architecture shipped (aggregator + retired bert-self loop)
- 1 product-architecture finding: bert is research-only-shaped, not organic
