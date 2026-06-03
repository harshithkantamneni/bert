# bert — Architecture

bert is **long-context project-memory and retrieval infrastructure** for AI coding hosts (Claude Code, Cursor, Codex, Claude Desktop). It installs as a local MCP server and gives the host a persistent, per-project memory + hybrid-retrieval layer plus a free-tier dispatch/verification harness. Its scope is deliberately narrow, and that scope is set by evidence, not ambition (see *What bert is / is NOT* and *Benchmark program*).

This document describes the real subsystems as they exist in the tree, with a data-flow overview, the complete benchmark metrics, and a "Where to look" index mapping every topic to the actual files so a reader can verify each claim.

---

## What bert is / is NOT

**bert IS:**
- A **local stdio MCP server** that turns an MCP host into a persistent, multi-lab orchestrator (`tools/mcp/bert_lab.py`), where "lab" == one project rooted at `~/.bert/labs/<name>/`.
- A **hybrid retrieval engine** (dense vector + BM25 + cross-encoder rerank, fused by Reciprocal Rank Fusion) over a per-project markdown corpus (`core/retrieval.py`, `core/memory.py`, `core/bm25.py`, `core/reranker.py`).
- A **free-tier dispatch + verification harness**: effort-sized routing across host / BYO-key / free-tier model lanes (`core/router.py`), a 9-step agent loop (`core/agent.py`), and Python-native artifact verification that overrides the agent's self-report (`core/verify_engine.py`).
- A program built to **falsify its own claims** and report the nulls (`benchmarks/`).

**bert is NOT:**
- **NOT a better agent / better reasoner.** B7 held the model constant: bert-Opus showed ≈0 quality gain over bare-Opus at 17–47× the tokens, and *hurt* on trivia.
- **NOT "a cheaper model + harness that matches the frontier."** That was the B7 hypothesis and it was disproved: bert-Sonnet 0.79 < bare-Sonnet 0.87 < bare-Opus 0.89; harness lift −0.077 (same negative sign on all three tasks); never beat Opus (tie/tie/loss).
- **NOT an autonomous lab that beats Opus.** The data supports only the long-context retrieval niche.
- **NOT a SaaS.** It spawns subprocesses, pins a resident embedder+reranker, and writes a persistent filesystem tree under `~/.bert/labs/` — inherently a single-tenant, per-user local process. There is no remote/HTTP transport; the only transport is stdio.

**The one confirmed, defensible value (B9):** the **long-context regime** — when a project exceeds the model's context window, full-context stuffing becomes infeasible and naive truncation drops the answer to 0.00; hybrid retrieval is the only thing that still works, at a flat ~3.3K input tokens regardless of corpus size (132K → 3M). A frontier model with a 1M window brute-forces anything that fits inside it; nobody brute-forces a 10M-token project. That gap is the niche, and it is the entirety of what bert claims.

**Genuinely open (do not claim either way):** whether orchestration helps a *weak* model on genuinely multi-step / long-horizon work is untested. Single-deliverable orchestration is shown dead; multi-step is unmeasured.

---

## System overview

```
            MCP host (Claude Code / Cursor / Codex / Claude Desktop)
                              │  stdio, JSON-RPC 2.0 (one msg/line)
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  MCP LAYER   tools/mcp/bert_lab.py  (the public surface)  │
        │  built on core/mcp_server.py (~366-LoC JSON-RPC framework)│
        │  11 tools · MCP resources (lab artifacts) · MCP prompts   │
        └───────────────┬───────────────────────┬──────────────────┘
            memory_search│                       │ lab_cycle (subprocess)
                         ▼                       ▼
        ┌────────────────────────┐   ┌────────────────────────────────┐
        │ MEMORY + RETRIEVAL     │   │ CORE DISPATCH  tools/bert_run.py │
        │ retrieval.hybrid_      │   │  effort_triage → router →        │
        │  retrieve (the value)  │   │  { claude -p bridge | subagent } │
        │ vector+BM25+rerank     │   │  → agent loop → verify_engine    │
        └────────────────────────┘   └────────────────────────────────┘
                         │                       │
                         ▼                       ▼
        ┌─────────────────────────────────────────────────────────┐
        │  PER-LAB FILESYSTEM STATE   ~/.bert/labs/<name>/          │
        │  memories/  findings/  sor/events.jsonl  state/  lab.yaml │
        └─────────────────────────────────────────────────────────┘
```

Per project, the host re-reads state via subsequent tool calls; nothing is held in server memory across requests except the prewarmed ML models.

---

## 1. MCP layer (the public surface)

**Shared framework — `core/mcp_server.py` (~366 LoC).** `MCPServer` is a JSON-RPC 2.0 server over stdio: `serve_stdio()` is a blocking `sys.stdin.readline` loop, one JSON message per line, JSON written to stdout. It implements `initialize` / `tools.list` / `tools.call` / `resources.list` / `resources.read` / `prompts.list` / `prompts.get`, plus `register_tool` / `register_resource` / `register_prompt`. The handshake reports `protocolVersion 2025-06-18` and advertises tools/resources/prompts capabilities. It supports optional **dotted-namespace qualification**: when `namespace` is set, `tools/list` emits qualified ids (`_qualified`) while `tools/call` still accepts the bare name (`_strip_ns`, lines 104–108) for back-compat. It also carries nonce-based replay protection (`_meta.nonce` → `core.mcp_replay`, `REPLAY_REJECTED -32004`, fail-open) and INVALID_PARAMS-vs-INTERNAL_ERROR mapping.

**The headline server — `tools/mcp/bert_lab.py` (~1056 LoC).** This is the one a real user installs and the README documents. It is a **multi-lab** orchestrator rooted at `~/.bert/labs/<name>/`. It is constructed with `namespace="bert.lab"` (line 671), so its tools are listed as `bert.lab.<tool>` (e.g. `bert.lab.lab_start`, `bert.lab.memory_search`) and callable by either the qualified or bare name.

`make_server()` issues **11 `register_tool` calls** (verified at lines 673, 693, 713, 739, 768, 788, 818, 892, 919, 945, 970):

| tool | purpose | core boundary |
|---|---|---|
| `lab_list` | enumerate known labs | filesystem walk of `~/.bert/labs` |
| `lab_status` | inspect a lab's state | `_lab_summary` |
| `lab_start` | create a new lab | `mission_profile.classify_mission` + `schema_synthesizer` (scaffolds dir tree, `seed_brief.md`, `lab.yaml`) |
| `lab_cycle` | run N autonomous cycles | **subprocess** → `tools/bert_run.py --lab <path> --max-cycles <budget> --autonomous` |
| `lab_reshape` | re-profile a drifting lab | `profile_drift` |
| `lab_resume` | resume a paused lab | `pause_resume` token verify |
| `lab_finalize` | grade + sign + ledger | `skill_runner.run_skill('finalize_project')` (4-judge grade) |
| `lab_approve` | PI-approve an organic proposal | `proposal_activate.activate` |
| `lab_synthesize_tool` | propose+sandbox a new tool | candidate code in `core.sandbox` |
| `memory_search` | hybrid search a lab's corpus | `core.memory.search` under `lab_context` |
| `packet_export` | build a signed proof-packet `.tar.gz` | `proof_packet.build_packet` |

Beyond tools, `bert_lab` exposes **MCP resources** (each lab's `seed_brief.md` + `lab.yaml` as `bert://lab/<name>/<artifact>`) and **MCP prompts** (one per feature `.md` under `core/library/features/`, fetchable via `prompts/get(name,{topic})`).

**`make_server()` vs `serve()`.** `make_server()` is deliberately **model-free** so tests can build a server without loading ML models. `serve()` is the `__main__` entry point: it first calls `core.prewarm.prewarm()` (background daemon pins embedder + reranker resident to move the cold-start tail off the first `memory_search`), then `make_server().serve_stdio()`.

**Legacy A2A inspection servers** — `bert_orchestrator`, `bert_memory`, `bert_queue`, `bert_mission`, `bert_search`, `bert_evaluator`, `bert_sandbox` — are a separate, older read/inspection surface scoped to a **single** lab at the repo root (`LAB_ROOT/memories`, `/findings`, `/lab/sor/events.jsonl`, `/work_queue`), launched via `lab.py mcp <name>` → `core.mcp_server.run(name)` dynamic-import. These are **not** what the README install wires up. Write/execute tools on them are permission-gated by a required `approver` field (P-005): `bert_queue.submit_pending`, and all `bert_sandbox` `run_python` / `run_shell`.

---

## 2. Core dispatch + model routing

`tools/bert_run.py` (~1444 LoC) is the orchestrator and the host bridge. A 3-layer routing stack feeds a 9-step agent loop.

**Stage 1 — effort triage (`core/effort_triage.py`, ~87 LoC).** A deterministic classifier over a frozen lexicon (`core/library/effort_lexicon.yaml`) returns `(effort: trivial|standard|deep, needs_grounding, confidence)`. `trivial` short-circuits to one cheap host-tier direct answer (this is the fix for the 253K-token-trivia waste B8 root-caused); `standard` trims the roster to one role; `deep` keeps the full roster. A quality-first guard: ALWAYS_DEEP judgment keywords (review/judge/propose/falsify/paper/decide) never down-triage.

**Stage 2 — the router (`core/router.py`, ~389 LoC).** `resolve_model_for_dispatch` (L270) is the canonical entry. It maps a role's cost tier (A/B/C from `role_registry`, after keyword overrides that force judgment asks to A) to a concrete `(provider, model)` with a strict **host > BYO > free** preference:
- **Tier-1b host default** (`_host_model_for_tier`, L253): when a Claude Code / Cursor host is attached, the host runs **every** reasoning role — **A→`claude-opus-4-7`, B→`claude-sonnet-4-6`, C→`claude-haiku-4-5`** via the `anthropic-cli/*` lane (verified `TIER_TO_PROVIDER_MODEL`, L139–151).
- **Tier-2 BYO keys** and **Tier-3 free-tier matrix** (`resolve_tier`, L181) engage only headless; default fallback `nvidia/llama-3.3-70b`.
- `BERT_FORCE_MODEL` is an escape hatch that pins a lane.
- `select_first_attempt_provider` (L75) is a **separate** RouteLLM heuristic stub, **distinct from and not yet wired into** the live tier path.

**Stage 3 — dispatch fork (`tools/bert_run._safe_dispatch`).** On the resolved model:
- `anthropic-cli/*` → `_dispatch_via_claude_cli`: runs `claude -p --model {opus|sonnet|haiku} --output-format json --max-budget-usd 2.0` (900s timeout) against the **user's own OAuth session**, with a cached `--append-system-prompt`; `_grade_bridge_artifact` then grades the output with `verify_engine` → APPROVE / CHANGES_REQUESTED. On failure it falls through to the free-tier subagent loop.
- everything else → `core.subagent.run_subagent` → `core.agent.run_role` (the L3 HTTP path via `core.provider.call`).

**The free-tier HTTP client — `core/provider.py` (~408 LoC).** OpenAI-compatible client for 8 lanes (groq / nvidia / cerebras / gemini / mistral / openrouter / hf_router / ollama). `call()` (L207) handles per-provider quirks (gemini thinking-token floor, nvidia `parallel_tool_calls=False`, openrouter Referer), retry/backoff on 429/502/503/504 honoring `Retry-After`, emits a circuit-breaker event on exhaustion, and records quota + cost ledger.

**The 9-step agent loop — `core/agent.py` (~614 LoC).** `run_role` (L172): system prompt = constitutional preamble + role prompt; per-iteration **5-shaper compaction** before `provider.call`; filters `Spawn` out of non-director roles; **in-loop cross-provider failover** via `provider_fallback` on failoverable errors; a bounded **deliverable-completion nudge** driven by `verify_engine`; writes `session_exit.md`, runs evaluator/consolidator in `finally`, and overwrites hallucinated telemetry into the telemetry sink.

**Subagent wrapper — `core/subagent.py` (~1428 LoC).** `run_subagent` (L669): validate `DispatchSpec` (jsonschema + KNOWN_ROLES) → `run_role(is_subagent=True, max_iter=20)` → **post-loop `verify_engine.verify_artifact` whose verdict OVERRIDES the agent's self-reported BUILD_PASS/FAIL** → schema-correction retry via `core.decode` → real-telemetry overwrite → cross-family Evaluator routing (`MODEL_FAMILIES`, `EVAL_SLOTS`, `pick_evaluator_model` L426, P-VS-02 picks a different family than the producer).

**Python-native verification — `core/verify_engine.py` (~296 LoC).** `verify_artifact` (L90) runs structured checks (existence / non-empty, `min_chars`, `required_headers`, required+forbidden patterns, optional `pytest_command` as list-argv, a `gaps.md` disclosure gate). `DEFAULT_SPEC` = 1500 chars + H1/3×H2 + citation pattern + no-placeholder. No shell injection. It is the single source of truth consumed by the agent (completion nudge), the subagent (verdict override), and the host-bridge grader.

**A cycle returns success** only if the result is schema-valid AND the verdict is not in `{BUILD_FAIL, REJECT, OTHER, CHANGES_REQUESTED}`. Prior findings thread into the next role's prompt; a cycle stops early on the first invalid result. The autonomous director loop (`tools/bert_run.py` L1189+, `core/director.py`) adds 3-strike / failure-cascade / pending termination guardrails.

---

## 3. Memory + retrieval (the subsystem the benchmarks identify as bert's real value)

A markdown corpus (`memories/` + `findings/`) is chunked, embedded into sqlite-vec, and retrieved through a multi-signal fusion pipeline, then reranked by a cross-encoder. Per-lab scoped via `lab_context`; runs offline/local on an M3 Pro. Pipeline: **ingest → index → hybrid_retrieve → rerank**, with `core/retrieval.py:hybrid_retrieve` as the single entry point.

**Ingest + dense index — `core/memory.py` (~484 LoC).** `ingest_corpus()` walks an external tree and writes each file as a `.md` shard under `findings/corpus/` via `create()` (gated to `memories/` or `findings/` only; atomic tmp+rename; also serves the `memory_create` MCP tool). `_index_corpus()` is lazy and mtime-driven: on each `search()` the corpus is re-walked, files newer than their `indexed_mtime` are re-chunked (paragraph-aware, 1500 chars / 100 overlap) and re-embedded (all-MiniLM-L6-v2, 384-dim, normalized → cosine via L2) into the `vec_chunks` sqlite-vec virtual table, with metadata in `chunks`. Orphan GC drops chunks for deleted/archived files; `archive/*` is excluded (demand-paging page-out). A 5s TTL cache skips the ~4ms walk when nothing changed. `HF_HUB_OFFLINE=1` is forced by default to avoid multi-minute hangs.

**Hybrid fusion — `core/retrieval.py` (~457 LoC).** `hybrid_retrieve` pulls `_vector_candidates` (`memory.search`), `_bm25_candidates` (`bm25.search`), and `_graph_candidates` (only if `seed_ids` given), fuses them via Reciprocal Rank Fusion (`k=60`), takes `top_n*5` as a rerank pool, applies the cross-encoder `rerank_fn`, sorts by final score, trims to `top_n`. Per-stage timings + per-signal top-K are emitted to `state/observability/retrieval.jsonl`. **PPR and the semantic cache were removed from fusion on empirical evidence** (PPR never fired on arbitrary corpora; the cache ordered by recency not relevance); only vector + bm25 (+ graph) fuse now.

**Sparse signal — `core/bm25.py` (~427 LoC).** `rank_bm25` BM25Okapi over the same sqlite chunks; IR-quality tokenizer (stopword removal + light stemmer; lifted BEIR scifact 0.56 → 0.66 nDCG); mtime-gated incremental rebuild + a 3-layer process cache (payload / BM25Okapi instance / freshness signature) fixing a profiled 22ms JSON re-parse + 24ms rebuild per call.

**Rerank — `core/reranker.py` (~245 LoC).** `BAAI/bge-reranker-v2-m3` cross-encoder scoring `(query, passage)` pairs with full attention; thread-safe lazy singleton with a 30s load timeout, MiniLM fallback model, cached-failure flag, and a `BERT_DISABLE_RERANKER` kill switch. On any failure `hybrid_retrieve` falls back to `default_cosine_reranker` (single-vector cosine via Ollama nomic-embed).

**Supporting:** `core/semantic_cache.py` (LLM-dispatch dedup cache, **distinct from retrieval**: nomic-embed 768-dim, 0.90 cosine + an anchor-term guard to defeat the embedder's topic-suffix collapse; strict `CACHEABLE_ROLES` allow-list excludes all verdict roles). `core/grader.py` (4-judge median+variance artifact grader over a free-tier provider cascade; `aggregate()` is a pure LLM-free function). `core/prewarm.py` (background daemon pins embedder + reranker resident at MCP-server start).

**THE RECENT FIX (the benchmark's highest-value output).** Two bugs in `core/retrieval.py` crushed recall to 0.10 nDCG:
- **(a) vector key mismatch** — `_vector_candidates` read `r['id']/['text']/['score']`, but `memory.search` returns `{path, chunk_idx, content, distance}`. Candidates got empty text + a bare index id, so the reranker saw nothing and the dense signal was silently zeroed in RRF (recall cratered *below* vector-only). Fixed to read `content` (falling back to `text`), derive id from `path:chunk_idx`, and convert `distance → similarity` as `1/(1+distance)`.
- **(b) 240-char truncation** — candidate text was excerpted to `[:240]`, dropping the answer span on long (1500-char) chunks before the reranker/reader ever saw it. Both `_vector_candidates` and `_bm25_candidates` now carry the **full** chunk content.

Together: **0.10 → 0.85 accuracy, 0.125 → 0.783 recall**. The old BEIR bench masked this by *reimplementing* retrieval instead of calling the production path.

---

## 4. Verification + proof packets

`core/verify_engine.py` (above) is the runtime verification gate. At finalize, `lab_finalize` runs the 4-judge grader (`core/grader.py`), and `packet_export` builds a portable, verifiable `.tar.gz` (`core/proof_packet.py`): OCI 1.1 manifest + SLSA v1.1 in-toto DSSE + a Sigstore-shaped bundle (local-dev mode), a separately-signed `failures.md`, `reproduce.sh`, and `HASHES.txt`.

> **Release-critical:** the proof schema strings in `core/proof_packet.py` are **load-bearing**, not cosmetic. `SCHEMA_VERSION = "bert.proof.v1"` (L90), `PREDICATE_TYPE = "https://bert.dev/cycle/v1"` (L91), and 10 more (`buildType` L435, `builder` L447, key hints L489/L946, `artifactType` + layer mediaTypes L517–525, annotation keys L533–534, trusted-root mediaType L944) are embedded in every emitted packet and are what `bert verify`/cosign match on. Renaming them is a **schema migration**, not a string scrub.

---

## 5. Benchmark program — complete metrics (honest)

Three suites (B7 infra-value, B8 efficiency, B9 long-context RAG), run to answer *where is bert actually good, and where is it not?* The discipline was to **falsify bert's own claims**. Source of truth: `benchmarks/BENCHMARK_SYNTHESIS.md`, `benchmarks/results/B9_RAG_RESULT.md`, and the raw JSON.

### B7 — Infrastructure value (model held constant on Opus, non-Claude 4-judge grader)

What the data **DISPROVED**:

| claim | test | result |
|---|---|---|
| orchestration improves quality on a frontier model | bert-Opus vs bare-Opus | **No** — ≈0 gain (within noise), *hurts* on trivia, at **17–47× tokens** |
| the harness lets a cheaper model match the frontier | harness-lift: bert-Sonnet vs bare-Opus/Sonnet | **No** — see table below |

Harness-lift study (raw: `benchmarks/results/b7_harness_lift_20260603T004921.json`):

| arm | score |
|---|---|
| bare-Opus | **0.886** |
| bare-Sonnet | 0.867 |
| bert-Sonnet | **0.790** |
| harness_lift (bert-Sonnet − bare-Sonnet) | **−0.077** (same negative sign on all 3 tasks) |
| raw tier_gap (bare-Opus − bare-Sonnet) | +0.019 |
| bert-Sonnet vs Opus | −0.096; pairwise = **tie / tie / loss (never beat Opus)** |

**Conclusion:** bert's orchestration does not improve single-deliverable quality at any tier; the decomposition/verification overhead slightly *degrades* a task a capable model handles in one shot. bert is **not a better reasoner**.

### B9 — Long-context RAG (the ONE confirmed value)

Corpus: httpx-0.28.1 + starlette (vendored), ~131K tokens, 57 files → 438 chunks. Reader: llama-3.3-70b (free, bert's runtime). Grader: mistral-large + deepseek-v4-pro (non-Claude). n=20 span-validated gold (single-hop 5 / multi-hop 7 / needle 8). Window = 15K (corpus ~8.7× window). Raw: `benchmarks/results/b9_rag_final_20260602T170111.json`.

| arm | accuracy | input tokens | recall@10 |
|---|---|---|---|
| A1 naive truncation (15K) | 0.10 | 15,000 | — |
| A2 smart truncation (manifest+heads) | 0.35 | 14,709 | — |
| A4 vector-RAG | 0.70 | 2,905 | 0.692 |
| **A3 hybrid-RAG (vector+BM25+rerank)** | **0.85** | **3,278** | **0.783** |

Retrieval holds quality flat at **~4.6× fewer input tokens** than truncation (15,000 / 3,278 ≈ 4.58; the synthesis rounds this to "~5×" — cite ~4.6× to the raw JSON or "~5×" to the synthesis, not a mix).

By tier (needle = the thesis, one line buried in 131K tokens):

| tier | A1 | A2 | A4 vector | A3 hybrid |
|---|---|---|---|---|
| single-hop | 0.00 | 0.40 | 0.80 | 0.80 |
| multi-hop | 0.29 | 0.43 | 0.71 | **0.86** |
| **needle** | **0.00** | **0.25** | 0.63 | **0.88** |

Truncation drops the needle (0.00–0.25); retrieval finds it (0.88). Hybrid beats plain vector on multi-hop (0.86 vs 0.71), as designed.

### B9 — The full-context WALL (Max-Opus 1M-window reader; corpus padded with numpy+sympy)

Raw: `benchmarks/results/b9_wall_20260602T173506.json`.

| corpus | A0 full-context | A1 truncation (15K) | A3 bert-RAG |
|---|---|---|---|
| 132K (fits 1M window) | **acc 1.00** @ 132,068 tok | — | 0.50 @ 3.5K tok (n=2) |
| **3.0M (exceeds window)** | **INFEASIBLE** (3.04M > 1M) | **0.00** @ 15K (n=8) | **0.75** @ 3.3K (n=8) |

**Conclusion:** below the window full-context is perfect (RAG unnecessary, sometimes worse); above it, full-context cannot run, truncation = 0.00, and retrieval is the **only** working option at a **flat ~3.3K input tokens regardless of corpus size (132K → 3M)**. Note the wall-regime RAG number is **0.75**, not 0.85 — never let the 3M regime imply 0.85. This is the regime (project > context window) where the product is *necessary*, not merely cheaper.

### B8 — Efficiency / effort-triage

Trivia: **8.8× cheaper / 7× faster** with **no accuracy loss** (difficulty-gated short-circuit) — measured in an internal effort-triage run; the raw run JSON is not included in this public copy, so treat this as indicative, not a committed-artifact result like B7/B9/B2/B10. Separately, root-caused a 17–47× input-token inflation = mostly re-counted prompt-cache reads, not fresh compute.

### Standard-benchmark anchors (recognized, comparable to published)

Beyond the custom suites, two recognized benchmarks anchor bert to the public landscape.

**B2 — BEIR scifact** (`b2_beir_scifact.py`, the standard IR benchmark; raw `benchmarks/results/B2_BEIR_RESULT.md`). bert's real stack (MiniLM + `core.bm25` + RRF + bge-reranker) on 5,183 docs / 300 queries:

| method | nDCG@10 |
|---|---|
| vector-only | 0.645 |
| BM25 | **0.658** (≈ published BM25 0.665) |
| hybrid (vector+BM25) | **0.684** (beats published BM25) |

(The reranker row OOM'd on 18 GB MPS — a hardware limit, not a bug.) Note BEIR's short passages can't exercise the wall — it measures retrieval-*stack quality* on standard data.

**B10 — Needle-in-a-Haystack** (`b10_niah.py`, the de-facto context-window test; raw `benchmarks/results/B10_NIAH_RESULT.md`). The standard NIAH method, extended past the window:

| | result |
|---|---|
| bert-RAG (5×5 depth×length grid) | **25/25 (100%)** incl. 2× the window |
| full-context | recall 1.0 ≤ window, **INFEASIBLE** at 2M (quota-bounded, sparse) |

Honest scope: **single-needle** (not RULER's multi-needle), and the full-context arm is one sample per cell (each is a real Opus call). Single-needle NIAH is *easy* for retrieval; B9 is the *hard*-distractor retrieval test (0.85, not 1.0).

### Three real bugs the eval caught + fixed in bert's own code (the highest-value output)

1. **Hybrid retriever silently broken** — wrong dict keys (`text/id/score` vs `path/chunk_idx/content/distance`) zeroed the RRF vector signal + a 240-char truncation dropped answer spans. **0.10 → 0.85 accuracy, 0.125 → 0.783 recall.** (`core/retrieval.py`)
2. **Gemini judge lane dead everywhere** — the provider key is `gemini`, not `google`; the grader's `DEFAULT_CASCADE` and the strong panel both used `google`, so all grading had silently been Mistral + nvidia-llama only. (`core/grader.py`)
3. **Token waste** — no difficulty gate; the full research ritual + multi-role roster ran on trivia (253K tokens to answer "what's the PostgreSQL port?"). Fixed by effort-triage. (`core/effort_triage.py`)

### Methodology / measurement integrity

- Judges must be **non-Claude** when arms are Claude-family (`assert_non_claude_cascade`); free-tier llama judges compress scores to 0.85–0.95 (use Mistral-large).
- Cost axis = **tokens + wall-clock**, never imputed Max-plan dollars.
- Pre-registered + frozen gold sets, every span verbatim-validated against the corpus, blind-authored + adversarially reviewed (selection bias), pairwise both-orders grading to de-compress.
- n=20 (B9) / n=3 (B7) are **directional, not significance-tested**; the free-llama reader caps absolute accuracy, so only the *contrast* (RAG vs truncation) is the result, not the absolute numbers.

---

## Where to look

| Topic | Files / docs |
|---|---|
| **What a user actually gets (11 tools)** | `tools/mcp/bert_lab.py` (11 `register_tool` @ L673–970) |
| **MCP transport / JSON-RPC framework / stdio loop / handshake / namespace / replay** | `core/mcp_server.py` (`serve_stdio`, `_qualified`/`_strip_ns` L104–108, `protocolVersion 2025-06-18`) |
| **`make_server` (model-free) vs `serve` (prewarm-then-serve)** | `tools/mcp/bert_lab.py`, `core/prewarm.py` |
| **MCP resources + prompts wiring** | `tools/mcp/bert_lab.py`, `core/mcp_server.py`, `core/library/features/` |
| **`lab_cycle` subprocess + Opus-via-`claude` bridge + budget/saturation** | `tools/mcp/bert_lab.py`, `tools/bert_run.py` |
| **Install recipes (Claude Desktop / Claude Code / Cursor)** | `README.md` (§ Install) |
| **Legacy single-lab A2A inspection servers + `lab.py mcp` dispatcher** | `tools/mcp/bert_{orchestrator,memory,search,mission,evaluator}.py`, `core/mcp_server.py:run` |
| **Permission-gated write/execute (`approver` / P-005)** | `tools/mcp/bert_queue.py`, `tools/mcp/bert_sandbox.py` |
| **Effort triage / trivial short-circuit (253K-token fix)** | `core/effort_triage.py`, `core/library/effort_lexicon.yaml`, `tools/bert_run.py` |
| **Tier resolution + host>BYO>free routing (live path)** | `core/router.py` (`resolve_model_for_dispatch` L270, `_host_model_for_tier` L253, `TIER_TO_PROVIDER_MODEL` L139), `core/role_registry.py`, `core/host_detector.py` |
| **RouteLLM stub (NOT wired live)** | `core/router.py` (`select_first_attempt_provider` L75) |
| **`claude -p` host-Opus bridge (subprocess, cache prefix, grading)** | `tools/bert_run.py` (`_dispatch_via_claude_cli`, `_grade_bridge_artifact`) |
| **Free-tier provider HTTP, quirks, retry/circuit-breaker** | `core/provider.py`, `core/provider_fallback.py`, `core/quota.py`, `core/cost_ledger.py` |
| **9-step agent loop (call, shapers, tools, permission, stop, failover)** | `core/agent.py`, `core/compact.py`, `core/permission.py`, `core/tool_registry.py` |
| **Subagent dispatch, schema validation, verdict override, cross-family eval** | `core/subagent.py`, `core/decode.py`, `schemas/dispatch_spec.json`, `schemas/result_packet.json` |
| **Python-native verification gate (source of truth over self-report)** | `core/verify_engine.py` |
| **The retrieval recall fix (0.10 → 0.85)** | `core/retrieval.py` (`_vector_candidates`, `_bm25_candidates`) |
| **Hybrid fusion entry point + RRF + rerank + what got REMOVED (PPR, cache)** | `core/retrieval.py` (`hybrid_retrieve`, `reciprocal_rank_fusion`) |
| **Chunk/embed/sqlite-vec indexing, lazy mtime reindex, TTL skip, orphan GC** | `core/memory.py` (`_index_corpus`) |
| **Per-lab scoping (`lab_context`)** | `core/memory.py`, `core/retrieval.py`, `core/bm25.py` |
| **BM25 sparse signal: IR tokenizer, 3-layer perf cache** | `core/bm25.py` |
| **Cross-encoder reranker: bge-reranker-v2-m3, fallbacks, kill switch** | `core/reranker.py`, `core/retrieval.py` |
| **Semantic dispatch cache (LLM dedup, not retrieval)** | `core/semantic_cache.py` |
| **4-judge median+variance grader + pure `aggregate()`** | `core/grader.py` |
| **Cold-start mitigation (prewarm)** | `core/prewarm.py` |
| **Proof packets (load-bearing schema strings)** | `core/proof_packet.py` (`SCHEMA_VERSION` L90, `PREDICATE_TYPE` L91, +10 more) |
| **Headline benchmark tables (start here)** | `benchmarks/BENCHMARK_SYNTHESIS.md`, `benchmarks/results/B9_RAG_RESULT.md` |
| **Raw JSON to verify every number** | `benchmarks/results/b7_harness_lift_20260603T004921.json`, `…/b9_rag_final_20260602T170111.json`, `…/b9_wall_20260602T173506.json` |
| **Methodology / rigor narrative** | `benchmarks/B7_INFRA_VALUE_METHODOLOGY.md`, `benchmarks/B8_EFFICIENCY_AND_RAG_PLAN.md` |
| **Publishable harness code (offline-tested)** | `benchmarks/b7_ab_infra.py`, `benchmarks/b7_stats.py`, `benchmarks/b9_rag.py`, `benchmarks/b9_rag_runner.py`, `benchmarks/b9_rag_stats.py` |
| **Autonomous director loop + termination guardrails** | `core/director.py`, `tools/bert_run.py` (L1189+), `core/lab_schema_io.py` |
