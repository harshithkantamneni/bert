#!/bin/zsh
# Run all production-bar test suites in order. Each suite is
# independent — if one fails, the others still run.
#
# Output: one line per suite, plus final summary.
# Exit: 0 if all green, 1 if any suite fails.
#
# Suites:
#   tier-1 unit edge cases       (46 tests, ~10s)
#   tier-2 integration + race    (19 tests, ~10s)
#   tier-3 chaos + adversarial   (18 tests, ~5s)
#   E2E MCP full lifecycle       (26 tests, ~5s)
#   investor demo acceptance     (8 beats, ~1s)
#   50-cycle soak + resource     (10 tests, ~15s)
#   security boundaries          (15 tests, ~10s)
#   clean-install verify         (8 checks, ~30s)
#
# Total budget: ~90s on a warm system. First run pays cold-start
# (model downloads, etc.) — ~3-5min.

set -u
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd -P)}"
cd "$REPO"

GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

TOTAL_PASS=0
TOTAL_FAIL=0
SUITES_FAILED=()

run_suite() {
  local name="$1"
  local cmd="$2"
  local t0=$(date +%s)
  local out
  out=$(eval "$cmd" 2>&1)
  local rc=$?
  local elapsed=$(($(date +%s) - t0))
  # Extract "All N tests passed." line
  local final_line=$(echo "$out" | grep -E "All [0-9]+ tests passed\." | tail -1)
  if [ $rc -eq 0 ] && [ -n "$final_line" ]; then
    local n=$(echo "$final_line" | grep -oE "[0-9]+" | head -1)
    printf "  ${GREEN}PASS${RESET}  %-32s  %s/%s tests  (%ss)\n" "$name" "$n" "$n" "$elapsed"
    TOTAL_PASS=$((TOTAL_PASS + n))
  else
    # Extract the actual fail line
    local fail_summary=$(echo "$out" | grep -E "(pass=[0-9]+ fail=[0-9]+|FAIL [0-9]+|Install verify:)" | tail -1)
    printf "  ${RED}FAIL${RESET}  %-32s  %s  (%ss)\n" "$name" "$fail_summary" "$elapsed"
    SUITES_FAILED+=("$name")
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  bert-lab production test gauntlet"
echo "════════════════════════════════════════════════════════════════"
echo ""

run_suite "tier-1 unit edge"          ".venv/bin/python tests/_smoke_phase_abcde_edge.py"
run_suite "tier-2 integration+race"   ".venv/bin/python tests/_smoke_phase_abcde_pessimistic.py"
run_suite "tier-3 chaos+adversarial"  ".venv/bin/python tests/_smoke_phase_abcde_chaos.py"
run_suite "E2E MCP lifecycle"         ".venv/bin/python tests/_e2e_mcp_full_lifecycle.py"
run_suite "investor demo acceptance"  ".venv/bin/python tests/_acceptance_demo_flight.py"
run_suite "50-cycle soak+resource"    ".venv/bin/python tests/_soak_50_cycles.py"
run_suite "security boundaries"       ".venv/bin/python tests/_security_boundaries.py"
run_suite "clean-install verify"      "zsh tests/_install_verify.sh"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Total: ${TOTAL_PASS} tests passing across $((8 - TOTAL_FAIL))/8 suites"
if [ $TOTAL_FAIL -gt 0 ]; then
  echo "  ${RED}FAILED suites${RESET}:"
  for s in "${SUITES_FAILED[@]}"; do
    echo "    · $s"
  done
fi
echo "════════════════════════════════════════════════════════════════"

exit $TOTAL_FAIL
