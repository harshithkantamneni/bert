#!/bin/bash
# v3 FINISH — precompute (steps 1-4) already done. Resumes the Claude code track
# at higher concurrency (the agentic arms are I/O-bound, not throttled, so more
# workers ~ linear speedup), then the semantic track, then the report.
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BERT_EMBED_DEVICE=cpu BERT_RERANKER_DEVICE=cpu BERT_CLAUDE_WORKERS=8
PY=.venv/bin/python; L=/tmp/v2_overnight; G=benchmarks/results/v2
say(){ echo "[$(date '+%H:%M:%S')] [v3f] $*" | tee -a "$L/v3.log"; }

say "5/7 Claude CODE-fact track — resume @ 8 workers (full frozen gold)"
$PY -u benchmarks/v2_claude_track.py --gold $G/gold.json --cache $G/retrieval_cache.json \
  --acache $G/aider_cache.json --out $G/claude_code_rows.jsonl >> "$L/v3_5_code.log" 2>&1
say "  $(grep -aE 'acc=' "$L/v3_5_code.log" | tail -9 | tr '\n' ' ')"

say "6/7 Claude SEMANTIC track — @ 8 workers"
$PY -u benchmarks/v2_claude_track.py --gold $G/semantic_gold.json --cache $G/retrieval_cache.json \
  --acache $G/aider_cache.json --out $G/claude_semantic_rows.jsonl >> "$L/v3_6_sem.log" 2>&1
say "  $(grep -aE 'acc=' "$L/v3_6_sem.log" | tail -9 | tr '\n' ' ')"

say "7/7 v3 report"
$PY -u -m benchmarks.v3_report >> "$L/v3_7_report.log" 2>&1
say "DONE -> benchmarks/V3_REPORT.md"
