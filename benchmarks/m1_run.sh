#!/bin/bash
# m1 memory-MCP benchmark driver: finish S (resume), run M, emit the crossover
# report. Single reader = Claude; bert/retrieval on CPU so the parallel arms +
# MCP servers don't collide on the GPU. Resumable per (id,arm,size).
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BERT_EMBED_DEVICE=cpu BERT_RERANKER_DEVICE=cpu M1_WORKERS=6
PY=.venv/bin/python; L=/tmp/v2_overnight
say(){ echo "[$(date '+%H:%M:%S')] [m1] $*" | tee -a "$L/m1.log"; }

say "1/3 S run (sanity — fits the window; resume from checkpoint)"
$PY -u benchmarks/m1_arms.py --size S >> "$L/m1_S.log" 2>&1
say "  S done: $(grep -aE '^  A[0-9]' "$L/m1_S.log" | tail -5 | tr '\n' ' ')"

say "2/3 M run (1.26M tokens — exceeds the window ~7x; the crossover)"
$PY -u benchmarks/m1_arms.py --size M >> "$L/m1_M.log" 2>&1
say "  M done: $(grep -aE '^  A[0-9]' "$L/m1_M.log" | tail -5 | tr '\n' ' ')"

say "3/3 crossover report"
$PY -u -m benchmarks.m1_report >> "$L/m1_report.log" 2>&1
say "DONE -> benchmarks/M1_REPORT.md"
