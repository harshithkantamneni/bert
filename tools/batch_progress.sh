#!/bin/bash
cd /Users/harshithkantamneni/Desktop/bert-lab
while true; do
  NOW_C=$(wc -l < state/observability/cycle_outcome.jsonl 2>/dev/null || echo 0)
  NOW_R=$(wc -l < state/observability/retrieval.jsonl 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) cycles=$NOW_C retrievals=$NOW_R"
  if ! ps -p $(cat /tmp/cycles_batch_pid.txt 2>/dev/null) >/dev/null 2>&1; then
    echo "BATCH_DONE"; break
  fi
  sleep 300
done
