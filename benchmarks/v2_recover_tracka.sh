#!/bin/bash
# Recover the 3 Track A datasets that failed in the main overnight run, WITHOUT
# contending with it for the GPU. Waits until the main run reaches PHASE 5 (the
# QA factorial — LLM-bound, GPU free) or finishes, then runs the missing BEIR
# datasets with --skip-rerank (fast, can't hit the rerank timeout; the rerank
# comparison is already covered by scifact/nfcorpus/fiqa). Finally regenerates
# the report so the recovered datasets merge in.
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BERT_EMBED_DEVICE=mps BERT_EMBED_BATCH=128
PY=.venv/bin/python
L=/tmp/v2_overnight
say(){ echo "[$(date '+%H:%M:%S')] [recover] $*" | tee -a "$L/recover.log"; }

say "waiting for main run to free the GPU (PHASE 5 factorial, or DONE)…"
for i in $(seq 1 900); do   # up to 15h
  grep -q "PHASE 5" "$L/driver.log" 2>/dev/null && { say "PHASE 5 reached — GPU free"; break; }
  grep -q "overnight DONE" "$L/driver.log" 2>/dev/null && { say "main run DONE — GPU free"; break; }
  sleep 60
done

for ds in arguana scidocs cqadupstack-programmers; do
  say "-> $ds (skip-rerank)"
  timeout 2400 $PY -u benchmarks/b2_beir_multi.py --datasets "$ds" \
      --max-queries 200 --max-docs 20000 --skip-rerank > "$L/2_${ds}_recover.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "$ds FAILED rc=$rc :: $(tail -1 "$L/2_${ds}_recover.log" 2>/dev/null | cut -c1-120)"
  else
    say "$ds done: $(grep -aE 'vector_only|hybrid_no_rerank' "$L/2_${ds}_recover.log" | tail -1 | cut -c1-90)"
  fi
done

say "regenerating report to merge recovered Track A…"
$PY -u benchmarks/v2_report.py > "$L/7b_report_recover.log" 2>&1
say "DONE"
