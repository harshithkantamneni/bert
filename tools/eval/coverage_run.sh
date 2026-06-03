#!/bin/zsh
# G3 — Python code coverage via coverage.py.
#
# Runs the full smoke suite under coverage instrumentation. Skips
# only the live/walkthrough variants and known-slow heavy externals
# (robustness, inspect_ai, owasp_inspect, memoryagentbench) that
# blow runtime past CI budget without adding line-coverage signal.
#
# Gate: total ≥ 35% (research-lab realistic baseline; many event-
# producing modules are exercised only via live LLM dispatches we
# don't run in unit mode).

set -u
REPO=/Users/harshithkantamneni/Desktop/bert-lab
cd "$REPO"
.venv/bin/python -m coverage erase

SKIP_PATTERNS="(_live_|_walkthrough_|_smoke_robustness|_smoke_inspect_ai|_smoke_owasp_inspect|_smoke_memoryagentbench|_smoke_bb_phase|_smoke_verification_command|_smoke_h4_wiring|_smoke_j1_endpoints|_smoke_aa_mission|_smoke_spawn|_smoke_otel_wiring)"

count=0
for smoke in tests/_smoke_*.py; do
  name=$(basename "$smoke")
  if [[ "$name" =~ $SKIP_PATTERNS ]]; then continue; fi
  count=$((count+1))
  print -P "%F{cyan}cover %f $name" >&2
  # 120s (was 45s): a few smokes load the real embedder (sentence-transformers
  # MiniLM); its cold-start races a 45s budget on memory-constrained hosts,
  # which silently dropped ~600 lines from the total on unlucky runs and made
  # the gate non-deterministic (±3%). 120s lets the cached model load reliably.
  gtimeout 120 .venv/bin/python -m coverage run \
    --source=core,tools,api \
    --append "$smoke" > /dev/null 2>&1
done
print "ran $count smoke files under coverage"

print -P "\n%F{cyan}━━━ coverage report (top modules) ━━━%f"
.venv/bin/python -m coverage report --skip-empty 2>&1 | tail -30
.venv/bin/python -m coverage json -o /tmp/coverage.json 2>/dev/null
TOTAL=$(.venv/bin/python -c "
import json
d = json.load(open('/tmp/coverage.json'))
print(round(d['totals']['percent_covered']))
" 2>/dev/null || echo "0")
print
# Gate ratchets up as coverage rises (35→75 after the 2026-05-28 push from
# 54%→76%). Target remains 90% (exemplary); raise this floor each time a new
# stable plateau is reached. Stays a few points below the measured total so
# normal churn doesn't flake the build.
print -P "%F{cyan}TOTAL: ${TOTAL}%%%f  (gate: ≥ 79%)"
if [ "$TOTAL" -lt 79 ]; then
  print -P "%F{red}FAIL%f"
  exit 1
fi
print -P "%F{green}PASS%f"
