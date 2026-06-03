# bert

> Long-context project-memory and retrieval infrastructure for Claude Code / Cursor / Codex. When a project outgrows the model's context window, bert's hybrid retrieval keeps answer quality flat at near-constant input cost — where full-context stuffing becomes infeasible and naive truncation drops to zero.

bert is a local **MCP server** that gives an AI coding host a persistent, per-project memory + hybrid-retrieval layer. It exists for one specific, measured problem: **projects that outgrow the model's context window.** A frontier model with a large window can brute-force anything that fits inside it — but no one stuffs a 10M-token project into a prompt. When the corpus exceeds the window, full-context becomes infeasible, naive truncation drops the answer, and retrieval is the only thing that still works. That regime is the entirety of what bert claims.

Those claims come from a benchmark program built to **falsify** them. The honest result up front: **bert's orchestration does NOT make a model produce better single deliverables.** With the model held constant, orchestration showed ≈0 quality gain at 17–47× the token cost, and a cheaper-model-plus-harness arm (0.79) scored *below* the same bare model (0.87) and never beat the bare frontier model (0.89). What the data *does* support is the long-context retrieval value below. The credibility here is the rigor and the published nulls, not a leaderboard claim.

## What bert is / is NOT

**IS:** a local stdio MCP server; a hybrid retrieval engine (dense vector + BM25 + cross-encoder rerank, fused by RRF) over a per-project corpus; a free-tier dispatch + verification harness.

**is NOT:** a better agent/reasoner (disproved — see below); "a cheaper model + harness that matches the frontier" (disproved); an autonomous lab that beats frontier models; a SaaS (it's a single-tenant local process, stdio only).

## Results (honest, falsification-first)

**Long-context RAG — the one confirmed value** (httpx+starlette corpus, free-llama reader, non-Claude judge):

| arm | accuracy | input tokens | needle-tier |
|---|---|---|---|
| naive truncation (15K) | 0.10 | 15,000 | 0.00 |
| smart truncation (15K) | 0.35 | 14,709 | 0.25 |
| **bert hybrid-RAG** | **0.85** | **3,278** | **0.88** |

**The full-context wall** (1M-token model): at 132K (fits) full-context scores 1.00; at **3.0M (exceeds the window) full-context is INFEASIBLE**, truncation scores **0.00**, and bert-RAG holds **0.75 at a flat ~3.3K input tokens**. Retrieval is the *only* option above the window.

**What was disproved** (reported as the headline, not buried): orchestration on a frontier model — ≈0 gain at 17–47× tokens; cheaper-model-plus-harness — bert-Sonnet 0.79 < bare-Sonnet 0.87 < bare-Opus 0.89, never won.

**Industry-standard anchors** (recognized benchmarks, comparable to published baselines):
- **BEIR scifact** (the standard IR benchmark, nDCG@10): bert's BM25 **0.658** ≈ published 0.665; **hybrid 0.684 beats it**. See [`benchmarks/results/B2_BEIR_RESULT.md`](benchmarks/results/B2_BEIR_RESULT.md).
- **Needle-in-a-Haystack** (the de-facto context-window test): bert-RAG **25/25** across a depth×length grid *including 2× the window*, where full-context is infeasible. (Single-needle NIAH, not RULER; the full-context arm is quota-bounded.) See [`benchmarks/results/B10_NIAH_RESULT.md`](benchmarks/results/B10_NIAH_RESULT.md).

Full methodology, results, and limitations: [`benchmarks/BENCHMARK_SYNTHESIS.md`](benchmarks/BENCHMARK_SYNTHESIS.md).

## Install (MCP server)

bert is a local stdio MCP server. **The retrieval layer needs no LLM and no API keys** — it embeds locally (`all-MiniLM-L6-v2`, 22 MB) + BM25 + a local cross-encoder reranker (`bge-reranker-v2-m3`, ~568 MB), both downloaded once by `pip`/HuggingFace. The *answering* is done by your **host model** (Claude Code / Cursor / Codex): the host calls `memory_search`, bert returns the relevant chunks, and the host's own model reasons over them. No Ollama, no llama — the free-tier llama in the benchmarks was only a controlled reader to isolate retrieval quality.

```bash
git clone <your-fork-url> bert && cd bert
python -m venv .venv && .venv/bin/pip install -e .     # pulls sentence-transformers, sqlite-vec, etc.
```

Register with Claude Code:
```bash
claude mcp add bert -- /abs/path/to/bert/.venv/bin/python -m tools.mcp.bert_lab
```

Or in `claude_desktop_config.json` / Cursor (`mcpServers` block):
```json
{ "mcpServers": { "bert": {
    "command": "/abs/path/to/bert/.venv/bin/python",
    "args": ["-m", "tools.mcp.bert_lab"],
    "env": { "PYTHONPATH": "/abs/path/to/bert" } } } }
```

The host then gets tools to ingest/inspect a project corpus, **`memory_search`** (the retrieval layer — the value), and proof-packet export. Labs live under `~/.bert/labs/<name>/`.

**Optional — provider keys** (only for the autonomous lab-*cycle* feature, where bert dispatches its own model calls rather than letting the host reason): put any subset in **your own** `~/.bert-lab/credentials.json` (mode 600). Keys are never bundled and never reach the host model — they stay in bert's outbound HTTP calls. The core retrieval/memory product needs none of this.
```bash
mkdir -p ~/.bert-lab && printf '%s\n' '{ "GROQ_API_KEY": "...", "NVIDIA_API_KEY": "..." }' > ~/.bert-lab/credentials.json && chmod 600 ~/.bert-lab/credentials.json
```

## Architecture

A subsystem map (MCP layer → core dispatch → memory + retrieval → verification), the full benchmark tables, and a "where to look" index are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License

MIT — see [`LICENSE`](LICENSE).
