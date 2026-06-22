#!/bin/bash
# Resume + finish the v2 run after parallelizing the factorial. Phases 1-4
# (corpora, Track A, precompute cache, aider cache) are already on disk; this
# does the parallel factorial (resumes from the checkpoint), then stats + report.
# No GPU needed — the factorial reads the precomputed retrieval caches.
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BERT_FACTORIAL_WORKERS="${BERT_FACTORIAL_WORKERS:-6}"
PY=.venv/bin/python
L=/tmp/v2_overnight
say(){ echo "[$(date '+%H:%M:%S')] [finish] $*" | tee -a "$L/finish.log"; }

say "PHASE 5 (parallel factorial resume, workers=$BERT_FACTORIAL_WORKERS)"
$PY -u benchmarks/v2_run.py --run >> "$L/5_factorial.log" 2>&1
say "  $(grep -aE 'factorial done|pending cells' "$L/5_factorial.log" | tail -1)"

say "PHASE 6 stats"
$PY -u benchmarks/v2_run.py --stats > "$L/6_stats.log" 2>&1

say "PHASE 7 report"
$PY -u benchmarks/v2_report.py > "$L/7_report.log" 2>&1
say "DONE -> benchmarks/V2_REPORT.md"
