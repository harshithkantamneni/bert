#!/usr/bin/env bash
# Canonical CI regression. Excludes live-only tests automatically.
# Usage:
#   ./tests/run_canonical.sh              # full canonical regression
#   ./tests/run_canonical.sh --include-live   # include live tests too
#
# BB.2 — no `set -e`. With it, the substitution `out=$(...)` triggered
# the script's own exit on any failing test, silently killing the
# regression and producing zero stdout. The loop manages its own
# per-test error handling; bash's strict-mode is unhelpful here.
#
# Also exports BERT_SKIP_INDEXER=1 to prevent CPU-bound sentence-
# transformer re-embedding of changed files (which was hanging the
# robustness test past the 60s budget — see BB.1).

set -u  # catch typos in variable names; do NOT set -e
cd "$(dirname "$0")/.."

# Tests don't need a fresh corpus index; the indexer is exercised
# directly by tests that care about it. This unblocks the robustness
# suite + cuts ~30s off the regression on a dirty corpus.
export BERT_SKIP_INDEXER=1

INCLUDE_LIVE=0
if [ "${1:-}" = "--include-live" ]; then
  INCLUDE_LIVE=1
fi

PASS=0
FAIL=0
SKIP=0
FAILED=()

for f in tests/_smoke_*.py; do
  name=$(basename "$f" .py)

  # Auto-skip live tests unless --include-live is set
  if [ $INCLUDE_LIVE -eq 0 ] && head -2 "$f" | grep -q '"""LIVE-TEST'; then
    SKIP=$((SKIP+1))
    continue
  fi

  # Important: capture rc immediately after the substitution. Under
  # set -e this would never reach the rc=$? line because the failing
  # assignment would terminate the script. Without set -e, rc captures
  # cleanly.
  out=$(timeout 60 .venv/bin/python -u "$f" 2>&1) || rc=$?
  rc=${rc:-0}
  if [ $rc -eq 124 ]; then
    FAIL=$((FAIL+1)); FAILED+=("TIMEOUT $name")
  elif echo "$out" | grep -qE "All [0-9]+ .*tests? passed\."; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1)); FAILED+=("$name (rc=$rc)")
  fi
  unset rc
done

echo "Canonical regression: $PASS pass, $FAIL fail, $SKIP skipped (live tests)"
echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failures:"
  for t in "${FAILED[@]}"; do echo "  $t"; done
  exit 1
fi
exit 0
