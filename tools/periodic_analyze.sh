#!/bin/bash
# Run the observability analyzer periodically while gen accumulates data.
# Each run produces benchmarks/results/diverse_progress_HHMM.md so we
# can track Zipfian convergence.

cd /path/to/Desktop/bert-lab

while true; do
  STAMP=$(date +%H%M)
  COUNT=$(wc -l < state/observability/retrieval.jsonl)
  echo "[$(date +%H:%M:%S)] retrieval events: $COUNT — running analyzer"

  .venv/bin/python tools/analyze_observability.py \
    --output "benchmarks/results/diverse_progress_${STAMP}.md" 2>&1 | tail -5

  # Print just the key Zipfian + cache hit rate lines
  grep -E "top-1/top-5|hit rate K=10|Total calls" \
    "benchmarks/results/diverse_progress_${STAMP}.md" 2>/dev/null | head -5

  # Sleep 30 min between snapshots
  sleep 1800
done
