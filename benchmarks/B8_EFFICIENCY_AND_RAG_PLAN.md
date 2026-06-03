<!-- Designed via the bert-efficiency-and-rag-plan workflow (5 agents, code-verified) on 2026-06-02. -->
# B8 — Efficiency + Long-Context RAG program (sequenced plan)

Three gated phases. **Spend zero live-Opus dollars until every measurement-integrity fix is green** — a leaky benchmark is worse than none.

## Root cause of the 17–47× token waste (code-verified)
1. **No difficulty gate.** `router.resolve_model_for_dispatch` Tier-1b routes *every* role to host Opus/Sonnet unconditionally. Trivia goes through the full agentic loop.
2. **Fixed heavy ritual.** `_seed_to_research_task` hard-codes ≥1500 chars + Top-signals + Hypotheses + gaps.md on every dispatch; the verification gate then *punishes* a short correct answer and nudges it back up.
3. **Multi-role roster on trivia.** 2–4 roles each pay the full ritual.
4. **Measurement artifact.** Gross `tokens_in` counts cache-reads (agentic loop re-sends context each turn). The 40:1 is mostly re-counted cache reads, not fresh compute. *(bridge split: DONE.)*

## Phase 0 — measurement integrity ($0, local TDD) — DO FIRST
- **WS0a** cache-split honesty: bridge split (done) + `cost_ledger.record` prices fresh@full / cache-read@~10% + fix `cache_hit_rate` denominator. Scope the claim to the anthropic-cli bridge path only (free-tier providers never parse cache).
- **WS0b** RAG lab-scoping (BLOCKER for RAG): `core/memory.py` `DB_PATH`/`INDEX_DIRS` are import-bound to repo-root and ignore `lab_context` → RAG arm silently hits bert's *global* memory.db. Parameterize off `lab_context` + add `ingest_corpus(src_dir)`. Hard precondition gate before any RAG arm runs.
- **WS0c** effort-triage: `core/effort_triage.py` `classify()` (frozen lexicon committed *before* the eval set), `BERT_EFFORT_TRIAGE=off` ablation, gold-correctness guard on the trivial path. Wire at top of `_run_one_cycle`: trivial → ONE direct haiku answer (no ritual, no 2nd role, Read/Write only).
- **WS0d** blinding+determinism: extend `_SCRUB_REGEXES` to strip Sonnet/Opus/Claude/Anthropic (not just "bert"), pin temperature=0, disable semantic_cache, invalidate the 5s index TTL.

## Phase 1 — judge + RAG harness (mostly $0, free readers/judges)
- **WS1a** pairwise both-orders comparator (win-rate + order-flip rate as position-bias) — de-compresses better than absolute 0.85–0.95 scores.
- **WS1b** strong **non-Claude** judge panel (arms are all Claude-family, so judge must not be): Mistral-large / DeepSeek-v4-pro / Gemini-2.5-pro (spaced, free-tier 429s in hot loops) / GLM-5.1 — 4 labs. Forcing anti-clustering rubric via the existing `system_prompt_fn`. No-Anthropic guard + calibration vs human gold.
- **WS1c/1d** RAG harness arms A0 full-stuff / A1 naive-trunc / A2 smart-trunc / A2′ budget-matched-trunc / A3 bert-RAG-hybrid (thesis) / A4 vector-only / A5 BM25-only. Reader = **free llama-3.3-70b first** (bert's real thesis). Frozen 60–100 item gold Q&A (tiers: 40% single-hop / 40% multi-hop / 20% needle), gold_chunk_ids computed with bert's own chunker. **Public repo (httpx/fastapi) co-primary** with the self-corpus to kill self-benchmarking bias.

## Phase 2 — live-Opus runs (gated, costs money)
- **WS2c** harness-lift tier study: bert-Sonnet / bare-Opus / **bare-Sonnet** (the missing control). Decomposition: `(bert-Sonnet − bare-Sonnet)` = harness lift at fixed tier; `(bare-Opus − bare-Sonnet)` = raw model-tier gap. Win = bert-Sonnet ≥ bare-Opus at lower $.
- RAG A0 Opus arms (the full-context ceiling + the infeasibility wall past the window).

## Prices (verified `model_prices.yaml`, USD/1K tok)
Opus 0.015/0.075 · Sonnet 0.003/0.015 · Haiku 0.0008/0.004 · all free judges/readers $0. Use per-dispatch server-side `cost_usd` as true $ (cache split server-side), not raw token counts.

## Resume narrative (if results land as hypothesized)

> **UPDATE (results landed):** the harness-lift hypothesis was **DISPROVED** — bert-Sonnet 0.79 < bare-Sonnet 0.87 < bare-Opus 0.89, harness_lift −0.077, never beat Opus. See [`BENCHMARK_SYNTHESIS.md`](BENCHMARK_SYNTHESIS.md). The **Honest fallback** clause below is what actually held; the conditional draft is kept only to document the pre-registration discipline.

Diagnosed and corrected a 17–47× input-token inflation in an autonomous multi-agent system by separating cache-read from fresh-billable input end-to-end (bridge → cost ledger → reporting), proving most of the blow-up was re-counted prompt-cache reads, not real compute. Added a pre-registered effort-triage classifier that short-circuits trivial lookups to a single cheap-tier answer instead of a multi-role 1500-char ritual — collapsing trivia from ~300K tokens to a few K with gold-verified no accuracy loss. Built a falsifiable RAG-vs-full-context benchmark over a ~131K-token in-window corpus plus a 132K→3.0M-token wall sweep, showing retrieval holds answer quality flat (esp. needle-in-haystack) while input cost stays sub-linear and frontier full-context stuffing hits an infeasibility wall — plus a harness-lift study that *falsified* the cheaper-model-plus-harness hypothesis (bert-Sonnet 0.79 < bare-Sonnet 0.87 < bare-Opus 0.89). **Honest fallback if not:** the contribution is the rigorous, adversarially-reviewed evaluation that could falsify its own system, and a precise root-cause of where orchestration helps vs purely taxes.
