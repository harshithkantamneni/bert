#!/bin/bash
# Runs after the free-tier factorial finishes. Adds: A5 real-bm25 redo, tokenomics,
# the all-Claude tier (frontier-reader proxy + literal MCP arm), then the final
# report with every track + both model tiers. Sequential (Claude arms must not
# overlap each other or the factorial's A7f; MCP arm spawns one MPS server at a time).
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BERT_FACTORIAL_WORKERS=6
# Run all post-process bert retrieval on CPU so the parallel arms don't collide on
# the GPU (concurrent MPS contexts deadlock). Retrieval is a small slice per call;
# parallelism more than compensates.
export BERT_EMBED_DEVICE=cpu BERT_RERANKER_DEVICE=cpu
export BERT_CLAUDE_WORKERS=4 BERT_LLAMA_WORKERS=5 BERT_MCP_WORKERS=3 BERT_TOKENOMICS_WORKERS=3
PY=.venv/bin/python
L=/tmp/v2_overnight
say(){ echo "[$(date '+%H:%M:%S')] [post] $*" | tee -a "$L/postprocess.log"; }

say "waiting for factorial finisher to complete…"
for i in $(seq 1 720); do
  grep -q "DONE -> benchmarks/V2_REPORT" "$L/finish.log" 2>/dev/null && { say "factorial done"; break; }
  sleep 60
done

say "1/6 A5 real-bm25 redo (refresh cache + delete stale A5 + re-run A5)"
$PY -u benchmarks/v2_fix_a5.py > "$L/p1_fixa5.log" 2>&1
$PY -u benchmarks/v2_run.py --run >> "$L/p1_fixa5.log" 2>&1
say "  $(grep -aE 'refreshed|deleted|factorial done' "$L/p1_fixa5.log" | tail -3 | tr '\n' ' ')"

say "2/6 Tier-1 llama + bert via tool (live MCP-style)"
$PY -u benchmarks/v2_bert_tool_arm.py 12 > "$L/p2_berttool.log" 2>&1
say "  $(grep -aE 'A_mcp_llama' "$L/p2_berttool.log" | tail -1)"

say "3/6 Tier-2 Claude — frontier-reader (Claude reads bert chunks, A0-A6)"
$PY -u benchmarks/v2_frontier_reader.py 12 sonnet > "$L/p3_frontier.log" 2>&1
say "  $(grep -aE 'frontier_reader.json|A3 ' "$L/p3_frontier.log" | tail -1)"

say "4/6 Tier-2 Claude — LITERAL MCP arm (Claude calls bert memory_search live)"
$PY -u benchmarks/v2_mcp_arm.py 12 sonnet > "$L/p4_mcp.log" 2>&1
say "  $(grep -aE 'A_mcp|acc=' "$L/p4_mcp.log" | tail -1)"

say "5/6 tokenomics — EVERY arm, BOTH tiers, real tokens"
$PY -u benchmarks/v2_tokenomics.py 6 > "$L/p5_tokenomics.log" 2>&1
say "  $(grep -aE 'tokenomics.json' "$L/p5_tokenomics.log" | tail -1)"

say "6/6 stats + FINAL report (all tracks + both tiers + tokenomics)"
$PY -u benchmarks/v2_run.py --stats > "$L/p6_stats.log" 2>&1
$PY -u benchmarks/v2_report.py > "$L/p6_report.log" 2>&1
say "DONE -> benchmarks/V2_REPORT.md"
