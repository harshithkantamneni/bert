#!/bin/bash
# RECOVERY after the gold/cache-drift bug. Gold is now FROZEN (v2_run reuses
# gold.json). This re-aligns the cache to the frozen 249 questions and re-runs the
# corrupted RAG cells, then the Claude tier + tokenomics + report. GPU (MPS) is
# used ONLY in the serial precompute; everything after is CPU/network + parallel.
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BERT_FACTORIAL_WORKERS=6
PY=.venv/bin/python; APY=/tmp/aider_venv/bin/python; L=/tmp/v2_overnight
say(){ echo "[$(date '+%H:%M:%S')] [recover2] $*" | tee -a "$L/recover2.log"; }

say "1/8 clean cache-corrupted rows"
$PY -u benchmarks/v2_clean_rows.py > "$L/r1_clean.log" 2>&1; say "  $(tail -1 "$L/r1_clean.log")"

say "2/8 re-precompute retrieval cache aligned to FROZEN gold (MPS, serial)"
BERT_EMBED_DEVICE=mps $PY -u benchmarks/v2_run.py --precompute > "$L/r2_precompute.log" 2>&1
say "  $(grep -aE 'retrieval_cache|FROZEN' "$L/r2_precompute.log" | tail -1)"

say "3/8 re-precompute aider cache (A6) for frozen gold"
$APY -u benchmarks/v2_precompute_aider.py > "$L/r3_aider.log" 2>&1; say "  $(tail -1 "$L/r3_aider.log")"

say "4/8 re-run factorial — RAG aligned + any missing reader cells (parallel)"
$PY -u benchmarks/v2_run.py --run > "$L/r4_factorial.log" 2>&1
say "  $(grep -aE 'factorial done|pending cells' "$L/r4_factorial.log" | tail -1)"

# CPU for all post-process bert ops so the parallel arms don't collide on the GPU
export BERT_EMBED_DEVICE=cpu BERT_RERANKER_DEVICE=cpu
export BERT_CLAUDE_WORKERS=4 BERT_LLAMA_WORKERS=5 BERT_MCP_WORKERS=3 BERT_TOKENOMICS_WORKERS=3

say "5/8 Tier-1 llama + bert-via-tool"
$PY -u benchmarks/v2_bert_tool_arm.py 12 > "$L/r5_berttool.log" 2>&1; say "  $(grep -a A_mcp_llama "$L/r5_berttool.log" | tail -1)"

say "6/8 Tier-2 Claude — frontier-reader (A0-A6)"
$PY -u benchmarks/v2_frontier_reader.py 12 sonnet > "$L/r6_frontier.log" 2>&1; say "  $(grep -aE 'A3 |frontier_reader.json' "$L/r6_frontier.log" | tail -1)"

say "7/8 Tier-2 Claude — live MCP arm  +  tokenomics (both tiers)"
$PY -u benchmarks/v2_mcp_arm.py 12 sonnet > "$L/r7_mcp.log" 2>&1; say "  $(grep -aE 'A_mcp|acc=' "$L/r7_mcp.log" | tail -1)"
$PY -u benchmarks/v2_tokenomics.py 6 > "$L/r7_tok.log" 2>&1

say "8/8 stats + FINAL report"
$PY -u benchmarks/v2_run.py --stats > "$L/r8_stats.log" 2>&1
$PY -u benchmarks/v2_report.py > "$L/r8_report.log" 2>&1
say "DONE -> benchmarks/V2_REPORT.md"
