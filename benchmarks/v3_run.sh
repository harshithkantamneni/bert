#!/bin/bash
# v3: SINGLE-MODEL (Claude) benchmark, correct (frozen+aligned gold), + semantic
# track. GPU (MPS) only in the serial precompute steps; Claude tracks run bert on
# CPU so the parallel arms don't collide on the GPU. Fully resumable per (id,arm).
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python; APY=/tmp/aider_venv/bin/python; L=/tmp/v2_overnight
G=benchmarks/results/v2; say(){ echo "[$(date '+%H:%M:%S')] [v3] $*" | tee -a "$L/v3.log"; }

say "1/7 precompute CODE retrieval cache (frozen gold, MPS serial)"
BERT_EMBED_DEVICE=mps $PY -u benchmarks/v2_run.py --precompute > "$L/v3_1_precompute.log" 2>&1
say "  $(grep -aE 'retrieval_cache|FROZEN' "$L/v3_1_precompute.log" | tail -1)"

say "2/7 precompute CODE aider cache (A6)"
$APY -u benchmarks/v2_precompute_aider.py > "$L/v3_2_aider.log" 2>&1; say "  $(tail -1 "$L/v3_2_aider.log")"

say "3/7 precompute SEMANTIC retrieval cache (merge into shared cache, MPS serial)"
BERT_EMBED_DEVICE=mps $PY -u benchmarks/v2_precompute_semantic.py > "$L/v3_3_sem_precompute.log" 2>&1
say "  $(tail -1 "$L/v3_3_sem_precompute.log")"

say "4/7 precompute SEMANTIC aider cache (merge)"
$APY -u benchmarks/v2_precompute_aider.py semantic_gold.json > "$L/v3_4_sem_aider.log" 2>&1; say "  $(tail -1 "$L/v3_4_sem_aider.log")"

# Claude tracks: bert on CPU (parallel-safe), 4 concurrent Claude calls
export BERT_EMBED_DEVICE=cpu BERT_RERANKER_DEVICE=cpu BERT_CLAUDE_WORKERS=4

say "5/7 Claude CODE-fact track — all 9 arms, full frozen gold"
$PY -u benchmarks/v2_claude_track.py --gold $G/gold.json --cache $G/retrieval_cache.json \
  --acache $G/aider_cache.json --out $G/claude_code_rows.jsonl > "$L/v3_5_code.log" 2>&1
say "  $(grep -aE 'acc=' "$L/v3_5_code.log" | tail -9 | tr '\n' ' ')"

say "6/7 Claude SEMANTIC track — all 9 arms"
$PY -u benchmarks/v2_claude_track.py --gold $G/semantic_gold.json --cache $G/retrieval_cache.json \
  --acache $G/aider_cache.json --out $G/claude_semantic_rows.jsonl > "$L/v3_6_sem.log" 2>&1
say "  $(grep -aE 'acc=' "$L/v3_6_sem.log" | tail -9 | tr '\n' ' ')"

say "7/7 v3 report"
$PY -u -m benchmarks.v3_report > "$L/v3_7_report.log" 2>&1
say "DONE -> benchmarks/V3_REPORT.md"
