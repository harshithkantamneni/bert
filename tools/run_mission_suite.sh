#!/bin/bash
# Run 3 missions sequentially against the lab.
# Captures: start/end timestamps per mission so we can slice cycle_outcome by mission.
#
# Usage: bash tools/run_mission_suite.sh

set -uo pipefail
cd /Users/harshithkantamneni/Desktop/bert-lab

CYCLES_PER_MISSION=${CYCLES_PER_MISSION:-5}

MISSIONS=(research build analysis)
MANIFEST=/tmp/mission_suite_manifest.jsonl
: > "$MANIFEST"

echo "=== Mission suite: $CYCLES_PER_MISSION cycles each ==="
echo "  missions: ${MISSIONS[*]}"
echo "  manifest: $MANIFEST"
echo ""

for m in "${MISSIONS[@]}"; do
  echo "=================================================================="
  echo "  MISSION: $m  ($(date +%H:%M:%S))"
  echo "=================================================================="

  cp "missions/${m}.md" lab/seed_brief.md

  START_TS=$(date -u +%Y-%m-%dT%H:%M:%S%z | sed 's/\([+-]\)\([0-9][0-9]\)\([0-9][0-9]\)$/\1\2:\3/')
  START_EPOCH=$(date +%s)
  echo "  start: $START_TS"

  .venv/bin/python tools/bert_run.py --max-cycles "$CYCLES_PER_MISSION" \
     > "/tmp/mission_${m}.log" 2>&1
  EXIT=$?

  END_TS=$(date -u +%Y-%m-%dT%H:%M:%S%z | sed 's/\([+-]\)\([0-9][0-9]\)\([0-9][0-9]\)$/\1\2:\3/')
  END_EPOCH=$(date +%s)
  ELAPSED=$((END_EPOCH - START_EPOCH))

  echo "  end:   $END_TS  ($((ELAPSED / 60))m ${ELAPSED}s total)"

  # Manifest entry
  echo "{\"mission\":\"$m\",\"start_ts\":\"$START_TS\",\"end_ts\":\"$END_TS\",\"elapsed_secs\":$ELAPSED,\"exit_code\":$EXIT,\"cycles\":$CYCLES_PER_MISSION}" >> "$MANIFEST"
  echo ""
done

echo "=== ALL MISSIONS DONE @ $(date +%H:%M:%S) ==="
echo ""
echo "Manifest:"
cat "$MANIFEST"
echo ""
echo "Run analyzer:"
echo "  .venv/bin/python tools/compare_mission_outcomes.py"
