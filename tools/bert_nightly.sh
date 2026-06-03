#!/usr/bin/env bash
# bert · nightly automation script.
#
# The "keep running and updating accordingly" loop. Designed for cron or
# launchd. Runs every night and:
#   1. daily_quality_report.py for today
#   2. daily_history_compile.py
#   3. director_letter.py for today
#   4. If today is Friday: weekly_quality_report + weekly_history_compile
#
# Each step is logged to lab/state/nightly.log with timestamps. The
# script exits non-zero if any required step fails (cron then surfaces
# the failure via email or your monitoring of choice).
#
# Install on macOS (launchd):
#   .venv/bin/python tools/install_nightly.py --install
#
# Install on Linux/macOS (crontab):
#   crontab -e
#   # add: 0 23 * * *  /full/path/to/bert-lab/tools/bert_nightly.sh
#
# Manual run:
#   ./tools/bert_nightly.sh
#   ./tools/bert_nightly.sh --include-weekly   # force weekly even if not Friday
#   ./tools/bert_nightly.sh --dry-run          # report what would run

set -u  # NOT -e: we want to log every failure rather than die mid-run

INCLUDE_WEEKLY=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --include-weekly) INCLUDE_WEEKLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *) echo "[warn] unrecognized flag: $arg (continuing)" ;;
  esac
done

LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB_ROOT"

LOG="$LAB_ROOT/lab/state/nightly.log"
mkdir -p "$(dirname "$LOG")"

# Day-of-week check: 5 = Friday on macOS/Linux date(1)
DOW=$(date +%u)
TODAY=$(date +%F)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*" | tee -a "$LOG"
}

run_step() {
  local label="$1"
  shift
  log "  ▸ $label"
  if [ "$DRY_RUN" -eq 1 ]; then
    log "    (dry-run: would execute) $*"
    return 0
  fi
  local rc=0
  local output
  output=$("$@" 2>&1) || rc=$?
  if [ $rc -eq 0 ]; then
    log "    ✓ ok"
  else
    log "    ✗ FAIL (rc=$rc)"
    while IFS= read -r line; do log "      $line"; done <<< "$output"
  fi
  return $rc
}

log "=== bert nightly starting (date=$TODAY, dow=$DOW, dry_run=$DRY_RUN) ==="

VENV_PY="$LAB_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  log "[FATAL] .venv/bin/python not found at $VENV_PY"
  log "        run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 2
fi

FAILURES=0

# Step 1: daily quality report for today
run_step "daily_quality_report --date today" \
  "$VENV_PY" tools/daily_quality_report.py --date today || FAILURES=$((FAILURES+1))

# Step 2: daily history compile
run_step "daily_history_compile" \
  "$VENV_PY" tools/daily_history_compile.py || FAILURES=$((FAILURES+1))

# Step 3: director letter
run_step "director_letter" \
  "$VENV_PY" tools/director_letter.py || FAILURES=$((FAILURES+1))

# Step 3b: model-registry refresh (#31) — validate registry + surface
# deprecations (#39 warn / #32 remap) + stamp a freshness marker.
run_step "refresh_model_cards.py" \
  "$VENV_PY" tools/refresh_model_cards.py || FAILURES=$((FAILURES+1))

# Step 4 (Friday only, or with --include-weekly): weekly rollup
if [ "$DOW" -eq 5 ] || [ "$INCLUDE_WEEKLY" -eq 1 ]; then
  run_step "weekly_quality_report" \
    "$VENV_PY" tools/weekly_quality_report.py || FAILURES=$((FAILURES+1))
  run_step "weekly_history_compile" \
    "$VENV_PY" tools/weekly_history_compile.py || FAILURES=$((FAILURES+1))
else
  log "  ▸ weekly steps skipped (not Friday and --include-weekly not set)"
fi

log "=== bert nightly complete (failures=$FAILURES) ==="

if [ "$FAILURES" -gt 0 ]; then
  exit 1
fi
exit 0
