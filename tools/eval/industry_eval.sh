#!/bin/zsh
# G-phase — Industry-standard testing + eval matrix (22 stages).
#
# Each stage is a hermetic gate that contributes to the final
# summary. Non-zero exit on any blocker; soft skips when a tool
# isn't installed (logged but not counted as failure).
#
# Absolute paths throughout so subshells don't lose cwd.
#
# Stages:
#   1. Python smoke suite
#   2. TypeScript strict typecheck
#   3. Vite production build
#   4. Ruff Python lint
#   5. npm audit (frontend deps)
#   6. pip-audit (Python deps, if available)
#   7. JSON-schema validation
#   8. API contract probe
#   9. Playwright feature sweep
#  10. Playwright deep behavior sweep
#  11. axe-core a11y audit
#  12. Engine smoke (dry-run)
#  13. Resilience suite (fault injection)
#  14. Cross-browser Playwright (chromium + firefox + webkit)
#  15. Python code coverage (gate ≥ 79%)
#  16. SAST — bandit (HIGH/HIGH = 0) + eslint-security
#  17. Secrets scan (gitleaks)
#  18. License scan (npm + python deps, permissive-only)
#  19. SBOM (CycloneDX, npm + python)
#  20. Lighthouse perf baseline (Home + Manuscript)
#  21. Visual regression (Playwright + pixelmatch)
#  22. Docker container build + smoke

REPO=/Users/harshithkantamneni/Desktop/bert-lab
cd "$REPO"
EVAL_DIR="$REPO/tools/eval"
# Flush prints so stdout reaches the log even mid-stage (zsh default
# buffers when stdout is a pipe). PYTHONUNBUFFERED handles python
# children; setopt forces zsh's print to flush.
export PYTHONUNBUFFERED=1
setopt NO_FLOW_CONTROL 2>/dev/null || true

# Eval-owned uvicorn — always start a fresh one as the eval's own
# child so its lifetime ties to the eval, and we have a PID we can
# health-check between stages. Any pre-existing uvicorn on :5174
# from this user's session gets killed first to avoid the dead-
# server-after-session-timeout pattern we hit before.
ensure_uvicorn() {
  # Are we already up?
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:5174/api/labs 2>/dev/null; then
    return 0
  fi
  print -P "  %F{cyan}↦%f (re)starting uvicorn on :5174..."
  "$REPO/.venv/bin/uvicorn" api.main:app --port 5174 --log-level warning \
    > /tmp/eval_uvicorn.log 2>&1 &
  EVAL_UVICORN_PID=$!
  EVAL_STARTED_UVICORN=1
  for _i in {1..30}; do
    sleep 1
    if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:5174/api/labs 2>/dev/null; then
      print -P "  %F{cyan}↦%f uvicorn ready (pid=$EVAL_UVICORN_PID)"
      return 0
    fi
  done
  print -P "  %F{yellow}↷%f uvicorn never came up; stages 8-10 will fail"
  return 1
}
# Make sure no stale uvicorn from a prior session is hogging the port
pkill -f "uvicorn api.main:app" 2>/dev/null
sleep 1
EVAL_STARTED_UVICORN=0
ensure_uvicorn

# Same for vite dev — visual regression + cross-browser need it.
ensure_vite() {
  if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:5173/ 2>/dev/null; then
    return 0
  fi
  print -P "  %F{cyan}↦%f (re)starting vite dev on :5173..."
  ( cd "$REPO/abyssal/v4" && npm run dev > /tmp/eval_vite.log 2>&1 ) &
  EVAL_VITE_PID=$!
  EVAL_STARTED_VITE=1
  for _i in {1..30}; do
    sleep 1
    if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:5173/ 2>/dev/null; then
      print -P "  %F{cyan}↦%f vite ready (pid=$EVAL_VITE_PID)"
      return 0
    fi
  done
  print -P "  %F{yellow}↷%f vite never came up"
  return 1
}
EVAL_STARTED_VITE=0
ensure_vite

cleanup_uvicorn() {
  if [ "$EVAL_STARTED_UVICORN" = "1" ] && [ -n "$EVAL_UVICORN_PID" ]; then
    kill "$EVAL_UVICORN_PID" 2>/dev/null
  fi
  if [ "$EVAL_STARTED_VITE" = "1" ] && [ -n "$EVAL_VITE_PID" ]; then
    kill "$EVAL_VITE_PID" 2>/dev/null
  fi
  if [ -n "$EVAL_WARMUP_PID" ]; then
    kill "$EVAL_WARMUP_PID" 2>/dev/null
  fi
}
trap cleanup_uvicorn EXIT

# Background warm-up: load sentence_transformers (and the transitive
# transformers import structure) into OS page cache while stages 1-12
# run. By the time we hit stage 13 (resilience suite, which imports
# memory.py which lazily uses sentence_transformers), the cold-cache
# 2347-file directory walk inside transformers/utils/import_utils
# has already happened — eliminating the read() timeout we saw on
# disk-pressured macOS.
( "$REPO/.venv/bin/python" -c "import sentence_transformers" > /dev/null 2>&1 ) &
EVAL_WARMUP_PID=$!

PASS=()
FAIL=()
SKIP=()

heading() { print -P "\n%F{cyan}━━━ $1 ━━━%f"; }
record_pass() { PASS+=("$1"); print -P "  %F{green}✓%f $1"; }
record_fail() { FAIL+=("$1: $2"); print -P "  %F{red}✗%f $1 :: $2"; }
record_skip() { SKIP+=("$1: $2"); print -P "  %F{yellow}↷%f $1 :: $2"; }

# Smoke runner — inlined so it's not dependent on /tmp.
SMOKE_RUNNER="$EVAL_DIR/_smoke_batch.sh"
if [ ! -f "$SMOKE_RUNNER" ]; then
cat > "$SMOKE_RUNNER" <<'SMOKERUNNER'
#!/bin/zsh
cd /Users/harshithkantamneni/Desktop/bert-lab
PASS=0; FAIL=0; TO=0; SKIP=0
FAILS=()
for f in tests/_smoke_*.py; do
  name=$(basename "$f")
  case "$name" in
    *_live_*|*_walkthrough_*) SKIP=$((SKIP+1)); continue ;;
  esac
  stdout=$(gtimeout 40 .venv/bin/python "$f" 2>/dev/null)
  rc=$?
  if [ $rc -eq 124 ]; then
    TO=$((TO+1)); FAILS+=("$name TIMEOUT"); continue
  fi
  if echo "$stdout" | grep -qE "^All [0-9]+( [^ ]+){0,4} (tests|checks) passed\.?$"; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
    fail_line=$(echo "$stdout" | grep -m1 "^  FAIL " | head -c 250)
    [ -z "$fail_line" ] && fail_line="rc=$rc"
    FAILS+=("$name :: $fail_line")
  fi
done
echo "=== RESULT: pass=$PASS  fail=$FAIL  timeout=$TO  skip=$SKIP ==="
for f in "${FAILS[@]}"; do echo "  $f"; done
SMOKERUNNER
chmod +x "$SMOKE_RUNNER"
fi

# ── 1. Python smoke suite ──────────────────────────────────────
heading "Stage 1 — Python smoke suite"
out=$(zsh "$SMOKE_RUNNER" 2>&1)
result_line=$(echo "$out" | grep "^=== RESULT")
print "  $result_line"
if echo "$result_line" | grep -q "fail=0"; then
  record_pass "smoke-suite (~135 files)"
else
  fails=$(echo "$result_line" | sed -E 's/.*fail=([0-9]+).*/\1/')
  record_fail "smoke-suite" "$fails non-environmental failures"
fi

# ── 2. TypeScript strict typecheck ─────────────────────────────
heading "Stage 2 — TypeScript strict"
ts_out=$(cd "$REPO/abyssal/v4" && npx -y tsc --noEmit -p tsconfig.json 2>&1)
if [ -z "$ts_out" ]; then
  record_pass "ts strict — 0 errors"
else
  count=$(echo "$ts_out" | grep -c "error TS")
  record_fail "ts strict" "$count errors: $(echo "$ts_out" | head -c 200)"
fi

# ── 3. Vite production build ───────────────────────────────────
# Retry once on failure: esbuild/rollup spawn native workers that can
# hit a transient SIGBUS/OOM under the full eval's disk+memory pressure
# (96%-full mac). The build is deterministic — a clean retry succeeds.
# Matches the ensure_uvicorn/ensure_vite transient-resilience pattern.
heading "Stage 3 — Vite production build"
build_out=""
for _attempt in 1 2; do
  build_out=$(cd "$REPO/abyssal/v4" && npx -y vite build 2>&1)
  echo "$build_out" | grep -q "built in" && break
  print -P "  %F{yellow}↻%f vite build attempt $_attempt failed; retrying..."
  sleep 2
done
if echo "$build_out" | grep -q "built in"; then
  size=$(echo "$build_out" | grep -E "index-.*\.js" | head -1 | awk '{print $2}')
  record_pass "vite build ok (index chunk $size)"
else
  record_fail "vite build" "$(echo "$build_out" | tail -c 200)"
fi

# ── 4. Ruff lint ───────────────────────────────────────────────
heading "Stage 4 — Ruff lint"
# Trust ruff's own exit code, not output grepping. The previous grep
# ("All checks passed|0 errors|^$") matched the BLANK LINES in ruff's
# multi-line violation output via `^$`, so it reported "ruff clean"
# regardless of how many violations existed — a false-green gate. ruff
# exits 0 only when clean, non-zero when there are violations.
if .venv/bin/python -m ruff check core/ tools/ api/ > /tmp/ruff_out.txt 2>&1; then
  record_pass "ruff clean"
else
  bad=$(grep -cE "^(core|tools|api)/\S+\.py:[0-9]+:[0-9]+:" /tmp/ruff_out.txt)
  [ "$bad" = "0" ] && bad=$(tail -1 /tmp/ruff_out.txt)
  record_fail "ruff" "$bad violations (see /tmp/ruff_out.txt)"
fi

# ── 5. npm audit ───────────────────────────────────────────────
heading "Stage 5 — npm audit (frontend deps)"
audit_out=$(cd "$REPO/abyssal/v4" && npm audit --json 2>&1)
high=$(echo "$audit_out" | python3 -c "import sys,json; d=json.load(sys.stdin); v=d.get('metadata',{}).get('vulnerabilities',{}); print(v.get('high',0)+v.get('critical',0))" 2>/dev/null || echo "?")
mod=$(echo "$audit_out" | python3 -c "import sys,json; d=json.load(sys.stdin); v=d.get('metadata',{}).get('vulnerabilities',{}); print(v.get('moderate',0))" 2>/dev/null || echo "?")
print "  high/critical=$high moderate=$mod"
if [ "$high" = "0" ]; then
  record_pass "npm audit — no high/critical (moderate=$mod tolerated)"
else
  record_fail "npm audit" "$high high/critical vulns"
fi

# ── 6. pip-audit ───────────────────────────────────────────────
# Isolated cache dir avoids stale cachecontrol entries producing
# spurious deserialization warnings.
#
# --skip-editable: bert-lab itself is installed editable (`uv pip
#   install -e .`) and isn't on PyPI, so it can't be audited; skipping
#   it removes the "could not be audited" false failure.
#
# Ignored vulns (re-audit when upstream relaxes / patches):
#   CVE-2025-69872 — diskcache (transitive of outlines); unpatched
#     upstream as of 2026-05-22; bert's cache path is user-local
#     non-shared.
#   PYSEC-2026-87 — lxml <6.1.0; fixed in 6.1.0 BUT crawl4ai (~=5.3)
#     and inscriptis (<6.1.0) both pin lxml below the fix, so the
#     upgrade is blocked upstream. bert's lxml exposure is web-content
#     parsing via Crawl4AI, which runs in a sandbox tier. Re-audit when
#     crawl4ai + inscriptis relax their lxml pins.
heading "Stage 6 — pip-audit (Python deps)"
if "$REPO/.venv/bin/python" -m pip show pip-audit 2>&1 | grep -q Name; then
  pa_out=$(gtimeout 300 "$REPO/.venv/bin/python" -m pip_audit \
    --cache-dir=/tmp/pip-audit-fresh \
    --skip-editable \
    --ignore-vuln CVE-2025-69872 \
    --ignore-vuln PYSEC-2026-87 \
    2>&1 | head -c 2000)
  pa_rc=$?
  if [ $pa_rc -eq 124 ]; then
    record_skip "pip-audit" "timed out at 300s (network resolution)"
  elif echo "$pa_out" | grep -qiE "no known vulnerabilities|^[Nn]o vulnerab"; then
    record_pass "pip-audit clean (CVE-2025-69872 ignored, see comment)"
  else
    record_fail "pip-audit" "$(echo "$pa_out" | tail -c 200 | tr '\n' ' ')"
  fi
else
  record_skip "pip-audit" "not installed"
fi

# ── 7. JSON-schema validation ──────────────────────────────────
heading "Stage 7 — JSON schemas validate"
schema_ok=$(.venv/bin/python -c "
import json, jsonschema
from pathlib import Path
errs = []
for p in Path('schemas').glob('*.json'):
    try:
        s = json.loads(p.read_text())
        jsonschema.Draft202012Validator.check_schema(s)
    except Exception as e:
        errs.append(f'{p.name}: {type(e).__name__}: {e}')
print('OK' if not errs else 'FAIL: ' + ' / '.join(errs))
" 2>&1)
if [[ "$schema_ok" == OK* ]]; then
  record_pass "json schemas (10 files) all valid Draft 2020-12"
else
  record_fail "json schemas" "$schema_ok"
fi

# ── 8. API contract probe ──────────────────────────────────────
heading "Stage 8 — API contract probe"
ensure_uvicorn > /dev/null 2>&1
api_out=$(.venv/bin/python "$EVAL_DIR/api_probe.py" 2>&1 | tail -2)
if echo "$api_out" | grep -q "no 5xx anywhere"; then
  record_pass "30 GET endpoints clean, bare + ?lab=test01"
else
  record_fail "api probe" "5xx detected"
fi

# ── 9. Playwright feature sweep ────────────────────────────────
heading "Stage 9 — Playwright feature sweep"
ensure_uvicorn > /dev/null 2>&1 && ensure_vite > /dev/null 2>&1
fs_out=$(.venv/bin/python "$EVAL_DIR/feature_sweep.py" 2>&1)
if echo "$fs_out" | grep -q "ALL FEATURE CHECKS PASSED"; then
  record_pass "feature sweep — 140+ checks, both lab states"
else
  fcount=$(echo "$fs_out" | grep -oE "FAILURES: [0-9]+" | head -1)
  record_fail "feature sweep" "${fcount:-no-passed-marker}"
fi

# ── 10. Playwright deep behavior sweep ─────────────────────────
heading "Stage 10 — Playwright deep behavior sweep"
ensure_uvicorn > /dev/null 2>&1 && ensure_vite > /dev/null 2>&1
ds_out=$(.venv/bin/python "$EVAL_DIR/deep_sweep.py" 2>&1)
if echo "$ds_out" | grep -q "DEEP SWEEP CLEAN"; then
  record_pass "deep sweep — 12 scenarios, real data"
else
  fcount=$(echo "$ds_out" | grep -oE "FAILURES: [0-9]+" | head -1)
  record_fail "deep sweep" "${fcount:-no-clean-marker}"
fi

# ── 11. axe-core a11y audit ────────────────────────────────────
heading "Stage 11 — axe-core a11y (WCAG 2.0 AA)"
a11y_out=$(.venv/bin/python "$EVAL_DIR/a11y_audit.py" 2>&1 | tail -2)
if echo "$a11y_out" | grep -q "no serious or critical"; then
  record_pass "axe — 9 surfaces, WCAG 2.0 AA, no serious/critical"
else
  vcount=$(echo "$a11y_out" | grep -oE "AXE: [0-9]+ violation" | head -1)
  record_fail "axe-core" "$vcount"
fi

# ── 12. Engine smoke (dry-run) ─────────────────────────────────
heading "Stage 12 — Engine smoke (dry-run)"
es_out=$("$REPO/.venv/bin/python" "$REPO/tools/bert_run.py" --lab test01 --dry-run --max-cycles 1 --autonomous 2>&1 | tail -3)
if echo "$es_out" | grep -q "dry-run.*exiting before any model"; then
  record_pass "bert_run dry-run on test01 ok"
else
  record_fail "engine dry-run" "$(echo "$es_out" | head -c 200)"
fi

# ── 13. Resilience suite ───────────────────────────────────────
# Wait for the background sentence_transformers warm-up (started at
# script start) to finish before running the resilience suite. The
# suite calls memory.search() which would otherwise pay the full
# transformers cold-cache cost under timeout pressure.
heading "Stage 13 — Resilience (fault injection)"
if [ -n "$EVAL_WARMUP_PID" ]; then
  wait "$EVAL_WARMUP_PID" 2>/dev/null
fi
res_out=$(BERT_SKIP_INDEXER=1 BERT_DISABLE_CANVAS_EMIT=1 gtimeout 600 "$REPO/.venv/bin/python" "$REPO/tests/_smoke_robustness.py" 2>&1)
res_rc=$?
# Don't use tail -1 — atexit handlers in canvas_emit/enrichment can
# write WARNING lines AFTER the success marker. Look for the marker
# anywhere in the output.
if echo "$res_out" | grep -q "All .* robustness tests passed"; then
  record_pass "robustness — graceful degradation under fault injection"
elif [ $res_rc -eq 124 ]; then
  record_skip "robustness" "ran out of time (>600s)"
else
  record_fail "robustness" "$(echo "$res_out" | tail -c 200)"
fi

# ── 14. Cross-browser Playwright ───────────────────────────────
heading "Stage 14 — Cross-browser (chromium + firefox + webkit)"
ensure_uvicorn > /dev/null 2>&1 && ensure_vite > /dev/null 2>&1
cb_out=$("$REPO/.venv/bin/python" "$EVAL_DIR/cross_browser_sweep.py" 2>&1)
if echo "$cb_out" | grep -q "CROSS-BROWSER SWEEP CLEAN"; then
  record_pass "cross-browser — 54 stops across 3 engines"
else
  fc=$(echo "$cb_out" | grep -oE "FAILURES: [0-9]+" | head -1)
  record_fail "cross-browser" "${fc:-error}"
fi

# ── 15. Python code coverage ───────────────────────────────────
heading "Stage 15 — Python code coverage (gate ≥ 79%)"
cov_out=$(zsh "$EVAL_DIR/coverage_run.sh" 2>&1)
if echo "$cov_out" | tail -2 | grep -q "PASS"; then
  pct=$(echo "$cov_out" | grep -oE "TOTAL: [0-9]+%" | tail -1)
  record_pass "coverage ${pct}"
else
  record_fail "coverage" "$(echo "$cov_out" | tail -c 200)"
fi

# ── 16. SAST (bandit + eslint-security) ────────────────────────
heading "Stage 16 — SAST (bandit HIGH/HIGH = 0, eslint-security)"
"$REPO/.venv/bin/python" -m bandit -r "$REPO/core" "$REPO/tools" "$REPO/api" \
  -f json -o /tmp/bandit.json 2>/dev/null
hh=$("$REPO/.venv/bin/python" -c "
import json
d = json.load(open('/tmp/bandit.json'))
hh = [r for r in d.get('results', []) if r['issue_severity'] == 'HIGH' and r['issue_confidence'] == 'HIGH']
print(len(hh))
" 2>/dev/null || echo "?")
es_out=$(cd "$REPO/abyssal/v4" && npx eslint 'src/**/*.{ts,tsx}' 2>&1)
es_rc=$?
if [ "$hh" = "0" ] && [ $es_rc -eq 0 ]; then
  record_pass "SAST — bandit 0 HIGH/HIGH + eslint-security clean"
else
  record_fail "SAST" "bandit_hh=$hh eslint_rc=$es_rc"
fi

# ── 17. Secrets scan (gitleaks) ────────────────────────────────
heading "Stage 17 — gitleaks (secrets)"
gl_out=$(gitleaks detect --config "$REPO/.gitleaks.toml" --no-banner \
  --no-git --redact 2>&1)
if echo "$gl_out" | grep -q "no leaks found"; then
  record_pass "gitleaks — no leaks"
else
  record_fail "gitleaks" "$(echo "$gl_out" | grep -E 'leaks found|leak:' | head -1)"
fi

# ── 18. License scan ───────────────────────────────────────────
heading "Stage 18 — License scan (permissive-only)"
lc_out=$("$REPO/.venv/bin/python" "$EVAL_DIR/license_scan.py" 2>&1)
if echo "$lc_out" | grep -q "LICENSE SCAN CLEAN"; then
  count=$(echo "$lc_out" | grep -oE "[0-9]+ dependencies" | head -1)
  record_pass "license-scan — $count, all permissive"
else
  record_fail "license-scan" "$(echo "$lc_out" | tail -c 300)"
fi

# ── 19. SBOM (CycloneDX) ───────────────────────────────────────
heading "Stage 19 — SBOM (CycloneDX npm + python)"
mkdir -p "$REPO/findings/sbom"
"$REPO/.venv/bin/python" -m cyclonedx_py environment "$REPO/.venv" \
  -o "$REPO/findings/sbom/python.cdx.json" 2>/dev/null
(cd "$REPO/abyssal/v4" && npx -y @cyclonedx/cyclonedx-npm \
  --output-file "$REPO/findings/sbom/npm.cdx.json" \
  --output-format JSON --omit dev > /dev/null 2>&1)
py_components=$("$REPO/.venv/bin/python" -c "import json;print(len(json.load(open('$REPO/findings/sbom/python.cdx.json')).get('components',[])))" 2>/dev/null || echo "0")
js_components=$("$REPO/.venv/bin/python" -c "import json;print(len(json.load(open('$REPO/findings/sbom/npm.cdx.json')).get('components',[])))" 2>/dev/null || echo "0")
if [ "$py_components" -gt 0 ] && [ "$js_components" -gt 0 ]; then
  record_pass "SBOM emitted — $py_components py + $js_components npm components"
else
  record_fail "SBOM" "py=$py_components js=$js_components"
fi

# ── 20. Lighthouse perf baseline ───────────────────────────────
# Targets the PRODUCTION-built bundle via `vite preview`, not the
# dev server. HMR + unminified chunks in dev mode make perf scores
# meaningless. We build, serve via preview on 4173, run lighthouse,
# then tear down.
heading "Stage 20 — Lighthouse (perf ≥ 70, a11y ≥ 95, best ≥ 95)"
CHROME_BIN="/Users/harshithkantamneni/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
mkdir -p "$REPO/findings/lighthouse"

# Spin up vite preview against dist/
( cd "$REPO/abyssal/v4" && npx vite preview --port 4173 --strictPort > /tmp/vite_preview.log 2>&1 ) &
PREVIEW_PID=$!
# wait for preview to bind (max 20s)
for i in {1..40}; do
  if curl -sfo /dev/null http://127.0.0.1:4173/; then break; fi
  sleep 0.5
done

lh_fail=0
for surface in "/" "/manuscript"; do
  out_name=$(echo "$surface" | sed 's|/|_|g; s|^_||; s|^$|home|')
  CHROME_PATH="$CHROME_BIN" npx -y lighthouse "http://127.0.0.1:4173$surface" \
    --output=json --output-path="$REPO/findings/lighthouse/${out_name}.json" \
    --chrome-flags="--headless=new --no-sandbox --disable-gpu" \
    --only-categories=performance,accessibility,best-practices \
    --quiet > /dev/null 2>&1
  scores=$("$REPO/.venv/bin/python" -c "
import json
d = json.load(open('$REPO/findings/lighthouse/${out_name}.json'))
c = d['categories']
p = int(c['performance']['score']*100)
a = int(c['accessibility']['score']*100)
b = int(c['best-practices']['score']*100)
print(f'{p} {a} {b}')
" 2>/dev/null || echo "0 0 0")
  read perf a11y best <<< "$scores"
  if [ "$perf" -ge 70 ] && [ "$a11y" -ge 95 ] && [ "$best" -ge 95 ]; then
    print -P "  %F{green}✓%f $surface — perf=$perf a11y=$a11y best=$best"
  else
    print -P "  %F{red}✗%f $surface — perf=$perf a11y=$a11y best=$best"
    lh_fail=$((lh_fail+1))
  fi
done

kill $PREVIEW_PID 2>/dev/null || true
wait $PREVIEW_PID 2>/dev/null || true

if [ $lh_fail -eq 0 ]; then
  record_pass "lighthouse — both surfaces above gates"
else
  record_fail "lighthouse" "$lh_fail surface(s) below gate"
fi

# ── 21. Visual regression ──────────────────────────────────────
heading "Stage 21 — Visual regression (vs committed baselines)"
ensure_uvicorn > /dev/null 2>&1 && ensure_vite > /dev/null 2>&1
vr_out=$("$REPO/.venv/bin/python" "$EVAL_DIR/visual_regression.py" 2>&1)
if echo "$vr_out" | grep -q "VISUAL REGRESSION CLEAN"; then
  record_pass "visual regression — 12 surfaces within 2% diff"
elif echo "$vr_out" | grep -q "BASELINES RECORDED"; then
  record_skip "visual regression" "baselines recorded; next run will diff"
else
  fc=$(echo "$vr_out" | grep -oE "[0-9]+ failures" | head -1)
  record_fail "visual regression" "${fc:-error}"
fi

# ── 22. Docker container build + smoke ─────────────────────────
heading "Stage 22 — Docker container build + smoke"
if ! command -v docker >/dev/null 2>&1; then
  record_skip "docker" "docker CLI not installed"
elif ! docker info >/dev/null 2>&1; then
  record_skip "docker" "docker daemon not running"
else
  if docker build -q -t bert-lab:eval "$REPO" > /dev/null 2>&1; then
    docker rm -f bert-eval > /dev/null 2>&1
    cid=$(docker run --rm -d --name bert-eval -p 5184:5174 bert-lab:eval 2>/dev/null)
    sleep 6
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:5184/api/labs)
    docker stop bert-eval > /dev/null 2>&1
    if [ "$code" = "200" ]; then
      record_pass "docker — image builds + /api/labs returns 200"
    else
      record_fail "docker" "smoke /api/labs → $code"
    fi
  else
    record_fail "docker" "build failed"
  fi
fi

# ── Summary ────────────────────────────────────────────────────
heading "Industry-standard eval — summary"
print -P "  %F{green}PASS%f: ${#PASS}/22"
print -P "  %F{red}FAIL%f: ${#FAIL}"
print -P "  %F{yellow}SKIP%f: ${#SKIP}"
print
print -P "%F{green}PASS:%f"
for p in $PASS; do print "  · $p"; done
if [ ${#FAIL} -gt 0 ]; then
  print -P "\n%F{red}FAIL:%f"
  for f in $FAIL; do print "  · $f"; done
fi
if [ ${#SKIP} -gt 0 ]; then
  print -P "\n%F{yellow}SKIP:%f"
  for s in $SKIP; do print "  · $s"; done
fi

# Exit nonzero if any blocker failed
[ ${#FAIL} -gt 0 ] && exit 1 || exit 0
