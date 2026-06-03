#!/bin/zsh
cd /path/to/Desktop/bert-lab
PASS=0; FAIL=0; TO=0; SKIP=0; RETRIED=0
FAILS=()

# Run one test file with a $2-second cap.
# Echoes a single outcome line: "PASS" | "TIMEOUT" | "FAIL<TAB><fail line>".
run_one() {
  local f=$1 t=$2 out rc
  out=$(gtimeout "$t" .venv/bin/python "$f" 2>/dev/null); rc=$?
  if [ $rc -eq 124 ]; then echo "TIMEOUT"; return; fi
  if echo "$out" | grep -qE "^All [0-9]+( [^ ]+){0,4} (tests|checks) passed\.?$"; then
    echo "PASS"; return
  fi
  local fl
  fl=$(echo "$out" | grep -m1 "^  FAIL " | head -c 250)
  [ -z "$fl" ] && fl="rc=$rc"
  printf 'FAIL\t%s\n' "$fl"
}

for f in tests/_smoke_*.py tests/_e2e_*.py tests/_acceptance_*.py tests/_soak_*.py tests/_security_*.py; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  case "$name" in
    *_live_*|*_walkthrough_*) SKIP=$((SKIP+1)); continue ;;
  esac
  # Skip files explicitly marked LIVE-TEST in their docstring — they require
  # real model-provider credentials + network and make live LLM dispatches
  # (slow + flaky + non-deterministic), so they don't belong in the offline
  # batch. Content-based detection is the robust rule (the _live_ filename
  # convention missed two that are named _smoke_* but ARE live).
  if head -3 "$f" | grep -q "LIVE-TEST"; then
    SKIP=$((SKIP+1)); continue
  fi
  # Per-test cap. Heavy tests (uvicorn/api startup, embedder load, 50-cycle
  # soak) get a generous cap so the batch's serial memory pressure doesn't
  # produce FALSE timeouts — these all finish well under 300s in isolation.
  case "$name" in
    *aa_mission*|*api_full*|*j1_endpoints*|*inspect_ai*|*retrieval*) timeout_s=300 ;;
    _soak_*) timeout_s=300 ;;
    _e2e_*|_security_*) timeout_s=150 ;;
    *) timeout_s=90 ;;
  esac
  res=$(run_one "$f" $timeout_s)
  outcome=${res%%$'\t'*}
  # Retry ONCE on a transient failure/timeout. A real failure fails both
  # times; a flake (uvicorn cold-start race, OOM stall under peak pressure)
  # passes on the lone re-run. This makes the gate reliable WITHOUT masking
  # real bugs — and it's reported (retried_ok) so flakiness stays visible.
  if [ "$outcome" != "PASS" ]; then
    res2=$(run_one "$f" $timeout_s)
    if [ "${res2%%$'\t'*}" = "PASS" ]; then
      RETRIED=$((RETRIED+1)); outcome="PASS"
    else
      res=$res2; outcome=${res2%%$'\t'*}
    fi
  fi
  case "$outcome" in
    PASS) PASS=$((PASS+1)) ;;
    TIMEOUT) TO=$((TO+1)); FAILS+=("$name TIMEOUT (failed x2)") ;;
    *) FAIL=$((FAIL+1)); FAILS+=("$name :: ${res#*$'\t'}") ;;
  esac
done
echo "=== RESULT: pass=$PASS  fail=$FAIL  timeout=$TO  skip=$SKIP  (retried_ok=$RETRIED) ==="
for f in "${FAILS[@]}"; do echo "  $f"; done
