# bert benchmark program — honest synthesis (2026-06-02)

Three suites (B7 infra-value, B8 efficiency, B9 long-context RAG) run to answer: *where is bert actually good, and where is it not?* The discipline was to try to **falsify bert's own claims**, not confirm them.

## What the data DISPROVED (bert's hypotheses that didn't survive)

| claim | test | result |
|---|---|---|
| orchestration improves quality on a frontier model | B7: bert-Opus vs bare-Opus | **No** — ≈0 gain (within noise), *hurts* on trivia, at 17–47× tokens |
| the harness lets a cheaper model match the frontier | B7 harness-lift: bert-Sonnet vs bare-Opus/Sonnet | **No** — bert-Sonnet 0.79 < bare-Sonnet 0.87 < bare-Opus 0.89; harness_lift −0.077 (same sign all 3 tasks); never beat Opus (tie/tie/loss) |

**Conclusion:** bert's orchestration does *not* improve single-deliverable quality at any model tier. The decomposition/verification overhead slightly *degrades* a task a capable model handles fine in one shot. bert is **not a better reasoner**.

## What the data CONFIRMED (bert's one real, defensible value)

| claim | test | result |
|---|---|---|
| retrieval beats full-context/truncation when the corpus > window | B9 RAG (131K corpus, 15K window) | **Yes** — hybrid-RAG 0.85 vs naive-trunc 0.10 / smart-trunc 0.35, at ~5× less input; needle tier 0.88 vs 0.00–0.25 |
| above the context window, retrieval is the *only* option | B9 wall (3.0M corpus, 1M window, Max Opus) | **Yes** — full-context INFEASIBLE, truncation 0.00, RAG 0.75 at flat ~3.3K tok |
| effort should scale to task difficulty | B8 effort-triage | **Indicative** — trivia 8.8× cheaper / 7× faster, no accuracy loss (internal run; raw JSON not included in this public copy, unlike B7/B9/B2/B10) |

**Conclusion:** bert's defensible niche is the **long-context regime** — projects that exceed the model's context window, where full-context can't run and retrieval is the only thing that works, at constant cost. Narrow, but real.

## Standard-benchmark anchors (recognized, comparable to published baselines)

The custom B7–B9 evals are rigorous but bespoke. These two are recognized benchmarks anyone can cross-check against the literature:

| benchmark | what it is | bert's result |
|---|---|---|
| **BEIR** (`b2_beir_multi.py`: scifact + nfcorpus + fiqa) | the standard IR benchmark (nDCG@10) | bge-base matches the published ref on all 3 (scifact 0.740 / nfcorpus 0.374 / fiqa 0.406); full stack beats published BM25 everywhere (+0.08 / +0.05 / +0.20); scifact hybrid+rerank **0.745** |
| **Needle-in-a-Haystack** (`b10_niah.py`) | the de-facto context-window test | bert-RAG **25/25** across a 5×5 depth×length grid *including 2× the window*; full-context walls past 1M |

**Honest scope:** BEIR validates retrieval-*stack* quality on standard data (its short passages can't show the wall). NIAH is **single-needle** and its **full-context arm is quota-bounded** (one sample per cell — not the full-context heatmap, and not RULER-grade multi-needle). Single-needle NIAH is also *easy* for retrieval (one distinctive needle) — the *hard* retrieval test, with semantically-similar distractors, is B9 (hybrid-RAG 0.85, not 1.0). Together: BEIR + NIAH anchor "competitive on standard data + survives past the window"; B9 + the wall cover "holds under realistic difficulty + the production wiring is correct".

## Bugs the eval caught in bert's own system (the highest-value output)

1. **Hybrid retriever silently broken** — `_vector_candidates` read the wrong dict keys (vector signal zeroed in RRF fusion) + 240-char truncation. Scored 5× worse than plain vector until fixed: **0.10 → 0.85**. The old BEIR benchmark masked it by *reimplementing* retrieval instead of calling the production path.
2. **Gemini judge lane dead everywhere** — provider key is `gemini`, not `google`; the grader's `DEFAULT_CASCADE` and the "strong panel" both used `google`, so all grading had silently been Mistral + nvidia-llama.
3. **Token waste** — no difficulty gate; the full research ritual + multi-role roster ran on trivia (253K tokens for "what's the PostgreSQL port?").

## The honest positioning

bert is **long-context-project infrastructure**, not a better agent. A frontier model with a 1M window brute-forces anything that fits; nobody brute-forces a 10M-token project. That gap — project > window — is the niche, and it's the only place the data supports the product being *necessary* rather than overhead.

**Genuinely open:** bert-llama on *harder, decomposition-needing* tasks (multi-step, long-horizon). Single-deliverable orchestration is dead; whether orchestration helps a weak model on genuinely multi-step work is untested.

## Methodology notes (reusable)
- Judges must be **non-Claude** when arms are Claude-family (`assert_non_claude_cascade`); free-tier llama judges compress scores to 0.85–0.95 (use Mistral-large; Gemini-2.5-pro free tier 429s under load).
- Cost axis = **tokens + wall-clock**, not Max-plan dollars (imputed/fictional).
- Pre-register gold sets, span-validate against the corpus, blind-author + adversarially review (selection bias), pairwise both-orders grading to de-compress.
