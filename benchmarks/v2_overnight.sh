#!/bin/bash
# v2 SOTA benchmark — unattended overnight driver.
# Robust by design: GPU phases are strictly serial (concurrent bge jobs thrash an
# 18GB M3 Pro); each BEIR dataset runs in its own timeout-isolated process so one
# ir_datasets hang can't sink the run; every phase logs with timestamps; the
# python phases checkpoint internally so a re-run resumes.
set -u
cd ~/Desktop/bert-mcp || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# MPS is SAFE here as long as only ONE process touches it at a time. The earlier
# "MPS hangs" was CONTENTION: overlapping probe/preflight processes + kill -9
# mid-MPS-op deadlocked the Metal command queue. This driver runs all GPU phases
# STRICTLY SERIALLY (one process each, run to completion — never kill -9'd
# mid-op), so MPS is the right call: ~11 chunks/s ingest (6.5x CPU) and ~136/s
# raw encode. DO NOT run any other GPU/MPS process while this is going.
export BERT_EMBED_DEVICE="${BERT_EMBED_DEVICE:-mps}"
export BERT_EMBED_BATCH="${BERT_EMBED_BATCH:-128}"
PY=.venv/bin/python
APY=/tmp/aider_venv/bin/python
L=/tmp/v2_overnight; mkdir -p "$L"
# Max-feasible corpus on MPS: ~40M tokens ≈ ~107K chunks ≈ ~2.7h to index at
# ~11 chunks/s. ~40x a 1M context window — full-context + truncation both
# collapse; retrieval is the only option.
TARGET_TOKENS="${TARGET_TOKENS:-40000000}"
MAXDOCS="${MAXDOCS:-20000}"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$L/driver.log"; }

say "=== v2 overnight START (target_tokens=$TARGET_TOKENS maxdocs=$MAXDOCS) ==="

say "PHASE 1: big corpus acquisition"
$PY -u benchmarks/v2_big_corpus.py --target-tokens "$TARGET_TOKENS" > "$L/1_bigcorpus.log" 2>&1
say "  $(tail -1 "$L/1_bigcorpus.log")"

say "PHASE 2: Track A — established BEIR datasets (per-dataset, timeout-isolated)"
for ds in scifact nfcorpus fiqa scidocs arguana cqadupstack-programmers; do
  say "  -> $ds"
  timeout 5400 $PY -u benchmarks/b2_beir_multi.py --datasets "$ds" --max-queries 200 --max-docs "$MAXDOCS" > "$L/2_$ds.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "  $ds FAILED/TIMEOUT rc=$rc (continuing)"; else say "  $ds done: $(grep -aE 'hybrid_with_rerank|vector_only' "$L/2_$ds.log" | tail -1)"; fi
done

say "PHASE 3: precompute retrievals (hybrid/vector/bm25) incl. big corpus [GPU]"
$PY -u benchmarks/v2_run.py --precompute --per-corpus 80 > "$L/3_precompute.log" 2>&1
say "  $(grep -aE 'retrieval_cache|ingested' "$L/3_precompute.log" | tail -1)"

say "PHASE 4: A6/Aider precompute [aider venv]"
$APY -u benchmarks/v2_precompute_aider.py > "$L/4_aider.log" 2>&1
say "  $(tail -1 "$L/4_aider.log")"

say "PHASE 5: QA factorial (parallel reader calls; A7f frontier subset) [providers]"
$PY -u benchmarks/v2_run.py --run > "$L/5_factorial.log" 2>&1
say "  $(grep -aE 'factorial done|ran [0-9]+ new' "$L/5_factorial.log" | tail -1)"

say "PHASE 6: stats (bootstrap CIs + McNemar + Holm)"
$PY -u benchmarks/v2_run.py --stats > "$L/6_stats.log" 2>&1

say "PHASE 7: REPORT.md"
$PY -u benchmarks/v2_report.py > "$L/7_report.log" 2>&1
say "  $(tail -1 "$L/7_report.log")"

say "=== v2 overnight DONE ==="
