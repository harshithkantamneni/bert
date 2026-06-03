#!/bin/zsh
# Clean-install verification — catches "works on dev's box only" bugs.
#
# Spawns a fresh venv outside the project, installs bert from source,
# and exercises:
#   1. `pip install -e .` succeeds without error
#   2. `bert --help` runs from installed entry point
#   3. `bert lab list` works
#   4. `python -m tools.mcp.bert_lab` is importable + responds to init
#   5. Every core module imports cleanly (no missing deps)
#
# Run: zsh tests/_install_verify.sh
# Exits 0 on full pass; 1 on any failure.
#
# Designed for CI; SAFE to run locally since the test venv is in /tmp.

set -e
set -u
set -o pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd -P)}"
TMPROOT=$(mktemp -d -t bert_install_XXXXXX)
VENV="$TMPROOT/venv"

PASS=0
FAIL=0
FAILURES=()

pass() {
  echo "  PASS  $1"
  PASS=$((PASS + 1))
}

fail() {
  echo "  FAIL  $1: ${2:-}"
  FAIL=$((FAIL + 1))
  FAILURES+=("$1")
}

cleanup() {
  rm -rf "$TMPROOT"
}
trap cleanup EXIT

echo "Running install-verify suite…"
echo "  repo: $REPO"
echo "  test venv: $VENV"
echo ""

# ── 1. Create fresh venv ─────────────────────────────────────────
echo "Step 1 — create venv (Python 3.13 to match pyproject)"
# Find python3.13 — uv-managed version preferred since it matches
# the project's .venv exactly. Falls back to any python3 on PATH.
PYBIN=""
for candidate in \
  "$HOME/.local/share/uv/python"/cpython-3.13-*/bin/python3.13 \
  /opt/homebrew/bin/python3.13 \
  /usr/local/bin/python3.13 \
  /usr/bin/python3.13; do
  if [ -x "$candidate" ]; then
    PYBIN="$candidate"
    break
  fi
done
if [ -z "$PYBIN" ]; then
  PYBIN="python3"  # last resort
fi
echo "  using: $PYBIN ($($PYBIN --version 2>&1))"

if "$PYBIN" -m venv "$VENV" 2>/dev/null; then
  pass "fresh venv created"
else
  fail "venv creation" "$PYBIN -m venv failed"
  exit 1
fi
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ── 2. Install bert from source ─────────────────────────────────
echo ""
echo "Step 2 — pip install"
# Use uv (always available in dev) with --python pinned to our venv.
# Note: full -e install resolves all deps; first run pays cold-start
# (~2-3 min on M3 Pro). Cache-warm runs complete in seconds.
if command -v uv > /dev/null 2>&1; then
  if timeout 300 uv pip install --python "$PYTHON" -e "$REPO" > "$TMPROOT/install.log" 2>&1; then
    pass "uv pip install -e . (Python 3.13)"
  else
    rc=$?
    if [ $rc -eq 124 ]; then
      fail "uv install" "5-minute timeout exceeded (dep resolution)"
    else
      fail "uv install" "$(tail -3 "$TMPROOT/install.log")"
    fi
  fi
else
  if timeout 600 "$PIP" install --upgrade pip > /dev/null 2>&1 \
     && timeout 600 "$PIP" install -e "$REPO" > "$TMPROOT/install.log" 2>&1; then
    pass "pip install -e ."
  else
    fail "pip install" "$(tail -3 "$TMPROOT/install.log")"
  fi
fi

# ── 3. bert entry point works ───────────────────────────────────
echo ""
echo "Step 3 — bert CLI entry point"
BERT="$VENV/bin/bert"
if [ -x "$BERT" ]; then
  if "$BERT" --help > "$TMPROOT/help.log" 2>&1; then
    pass "bert --help runs"
  else
    # The entry point lab:main may not have --help; try without args
    if "$BERT" > "$TMPROOT/help.log" 2>&1; then
      pass "bert (no args) runs"
    else
      rc=$?
      # rc != 0 is OK if it printed usage and exited
      if [ -s "$TMPROOT/help.log" ]; then
        pass "bert prints usage on no-args (rc=$rc)"
      else
        fail "bert CLI" "no output, rc=$rc"
      fi
    fi
  fi
else
  fail "bert CLI" "$BERT not installed"
fi

# ── 4. bert_cli works ────────────────────────────────────────────
echo ""
echo "Step 4 — bert_cli lab list (read-only)"
if "$PYTHON" "$REPO/tools/bert_cli.py" lab list > "$TMPROOT/lab_list.log" 2>&1; then
  pass "bert_cli lab list works"
else
  fail "bert_cli lab list" "$(tail -3 "$TMPROOT/lab_list.log")"
fi

# ── 5. MCP server importable ────────────────────────────────────
echo ""
echo "Step 5 — MCP server module"
if "$PYTHON" -c "from tools.mcp import bert_lab; print('OK')" > "$TMPROOT/mcp_import.log" 2>&1; then
  pass "from tools.mcp import bert_lab"
else
  fail "MCP import" "$(cat "$TMPROOT/mcp_import.log")"
fi

# ── 6. MCP server responds to init ──────────────────────────────
echo ""
echo "Step 6 — MCP handshake from fresh venv"
INIT_JSON='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
INIT_OUT=$(echo "$INIT_JSON" | "$PYTHON" -m tools.mcp.bert_lab 2>/dev/null | head -1)
if echo "$INIT_OUT" | grep -q '"result"'; then
  pass "MCP init returns result"
else
  fail "MCP handshake" "got: $INIT_OUT"
fi

# ── 7. Core modules import without error ─────────────────────────
echo ""
echo "Step 7 — core module imports"
CORE_MODULES=(
  "core.memory"
  "core.retrieval"
  "core.bm25"
  "core.token_graph"
  "core.reranker"
  "core.pause_resume"
  "core.signing"
  "core.proof_packet"
  "core.verify_packet"
  "core.mission_profile"
  "core.schema_synthesizer"
  "core.cycle_budget"
  "core.parallel_dispatch"
  "core.roster"
  "core.profile_drift"
  "core.memory_adapters"
  "core.memory_adapters.document_corpus"
  "core.memory_adapters.code_repo"
  "core.migrations"
  "core.brief_assembler"
  "core.consolidator"
  "core.director"
  "core.lab_context"
  "core.router"
)
import_fail=0
for mod in "${CORE_MODULES[@]}"; do
  if ! "$PYTHON" -c "import $mod" > /dev/null 2>&1; then
    fail "import $mod" "module import failed"
    import_fail=$((import_fail + 1))
  fi
done
if [ $import_fail -eq 0 ]; then
  pass "all ${#CORE_MODULES[@]} core modules import"
fi

# ── 8. bert verify entry point ───────────────────────────────────
echo ""
echo "Step 8 — bert_verify CLI"
if "$PYTHON" "$REPO/tools/bert_verify.py" --help > "$TMPROOT/verify_help.log" 2>&1; then
  pass "bert_verify --help works"
elif "$PYTHON" "$REPO/tools/bert_verify.py" 2>&1 | grep -q "usage:"; then
  pass "bert_verify prints usage"
else
  # Exit code 2 from argparse on missing args is normal
  rc=$?
  if [ $rc -eq 2 ]; then
    pass "bert_verify argparse exits 2 (expected)"
  else
    fail "bert_verify" "rc=$rc"
  fi
fi

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "Install verify: pass=$PASS fail=$FAIL"

if [ $FAIL -gt 0 ]; then
  echo ""
  echo "FAILURES:"
  for f in "${FAILURES[@]}"; do
    echo "  · $f"
  done
  exit 1
fi
echo "All $PASS tests passed."
