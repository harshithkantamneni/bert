"""Smoke test for R.2: --with-live-cycle orchestrator flag.

Validates plumbing without firing any model calls. The actual live
cycle (tools/bert_demo_cycle.py without --dry-run) is a LIVE-TEST
class — it needs real provider keys.

Depth: this is the kind of smoke that catches "I shipped a flag the
script doesn't actually parse" — common bug class in long bash scripts.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
RUN_SH = LAB_ROOT / "findings" / "investor" / "demo_recording" / "demo_run.sh"
NARRATION = LAB_ROOT / "findings" / "investor" / "demo_recording" / "narration.md"
DEMO_CYCLE = LAB_ROOT / "tools" / "bert_demo_cycle.py"
VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"


def test_bert_demo_cycle_exists() -> None:
    assert DEMO_CYCLE.exists(), "tools/bert_demo_cycle.py missing"


def test_demo_cycle_dry_run_works() -> None:
    """--dry-run validates the plumbing (parses corpus, identifies the
    scenario, prints intent) without firing any model dispatches. Should
    exit 0 with no provider keys set."""
    result = subprocess.run(
        [str(VENV_PY), str(DEMO_CYCLE), "--scenario", "1", "--dry-run"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},  # explicitly no API keys
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 0, \
        f"--dry-run should exit 0; got {result.returncode}; stderr={result.stderr[:300]}"
    out = result.stdout + result.stderr
    assert "dry-run" in out.lower(), "dry-run mode must announce itself"
    assert "scenario 1" in out.lower() or "S1" in out, \
        "dry-run must reference the chosen scenario"


def test_demo_cycle_aborts_without_keys() -> None:
    """Without --dry-run AND without any provider keys in env, the script
    must abort cleanly with exit 2 and a stage-safe error message rather
    than letting the dispatches 401 mid-demo."""
    result = subprocess.run(
        [str(VENV_PY), str(DEMO_CYCLE), "--scenario", "1", "--cycle", "9999"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},  # explicitly no API keys
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 2, \
        f"no-keys path should exit 2; got {result.returncode}"
    out = result.stdout + result.stderr
    assert "ABORT" in out or "no provider keys" in out.lower(), \
        "no-keys abort must surface a clear message"


def test_demo_cycle_rejects_unknown_scenario() -> None:
    """Out-of-range scenario must abort cleanly, not crash."""
    result = subprocess.run(
        [str(VENV_PY), str(DEMO_CYCLE), "--scenario", "99", "--dry-run"],
        capture_output=True, text=True, timeout=15,
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(LAB_ROOT),
    )
    assert result.returncode == 2, \
        f"unknown scenario should exit 2; got {result.returncode}"
    out = result.stdout + result.stderr
    assert "not in corpus" in out.lower() or "ABORT" in out, \
        "unknown-scenario error must be informative"


def test_orchestrator_bash_syntax_valid() -> None:
    """bash -n catches syntax errors in long shell scripts before stage."""
    result = subprocess.run(
        ["bash", "-n", str(RUN_SH)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, \
        f"demo_run.sh bash syntax invalid: {result.stderr}"


def test_orchestrator_parses_with_live_cycle_flag() -> None:
    """Flag must appear in the script's argument parser block AND
    control the conditional live-cycle segment."""
    text = RUN_SH.read_text()
    assert "--with-live-cycle" in text, \
        "demo_run.sh must recognize --with-live-cycle"
    assert "WITH_LIVE_CYCLE" in text, \
        "demo_run.sh must use the WITH_LIVE_CYCLE variable"
    # The conditional block must guard the live-cycle step
    assert 'if [ "$WITH_LIVE_CYCLE" -eq 1 ]' in text or \
           'if [[ "$WITH_LIVE_CYCLE" -eq 1 ]]' in text, \
           "live-cycle step must be conditional on the flag"


def test_orchestrator_help_flag_works() -> None:
    """-h / --help must print usage and exit 0 without booting uvicorn."""
    result = subprocess.run(
        ["bash", str(RUN_SH), "--help"],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, \
        f"--help should exit 0; got {result.returncode}"
    out = result.stdout + result.stderr
    assert "Usage" in out or "usage" in out, "help must include 'Usage'"
    assert "--with-live-cycle" in out, "help must document the new flag"


def test_orchestrator_invokes_bert_demo_cycle() -> None:
    """The live-cycle segment must shell out to bert_demo_cycle.py."""
    text = RUN_SH.read_text()
    assert "bert_demo_cycle.py" in text, \
        "live-cycle step must invoke tools/bert_demo_cycle.py"


def test_narration_includes_live_cycle_segment() -> None:
    """narration.md must document the optional segment so the founder
    knows what to say when --with-live-cycle is used."""
    text = NARRATION.read_text()
    assert "LIVE CYCLE" in text or "live cycle" in text.lower(), \
        "narration.md must include the live-cycle segment"
    assert "--with-live-cycle" in text, \
        "narration.md must reference the flag"
    assert "60" in text or "ninety" in text.lower() or "90" in text, \
        "narration must set the time expectation (60-90s wait)"


def test_orchestrator_warns_on_unknown_flag() -> None:
    """If founder mistypes the flag, the orchestrator should warn and
    continue — not silently ignore. Stage-safety: a mistyped flag
    silently dropping the live cycle would be a surprise the founder
    discovers on stage."""
    text = RUN_SH.read_text()
    assert "unrecognized flag" in text or "unknown" in text.lower(), \
        "demo_run.sh must warn on unknown flags"


# ── S.4 depth audit: runtime conditional + structural verification ──

def test_step_order_is_verify_then_live_cycle_then_ui() -> None:
    """The orchestrator's segments must run in the locked flight-plan
    order: bert verify → (optional live cycle) → bert FirstLight.
    Reordering changes the narration timing marks downstream."""
    lines = RUN_SH.read_text().splitlines()
    verify_line = None
    live_cycle_line = None
    firstlight_line = None
    for i, line in enumerate(lines, 1):
        if "Step 2: bert verify" in line and verify_line is None:
            verify_line = i
        if "Step 2.5" in line and "live cycle" in line.lower() and live_cycle_line is None:
            live_cycle_line = i
        if "Step 3:" in line and "FirstLight" in line.lower() or \
           "Step 3:" in line and "browser" in line.lower():
            if firstlight_line is None:
                firstlight_line = i
    assert verify_line is not None, "Step 2 (bert verify) marker not found"
    assert live_cycle_line is not None, "Step 2.5 (live cycle) marker not found"
    assert firstlight_line is not None, "Step 3 (FirstLight/browser) marker not found"
    assert verify_line < live_cycle_line < firstlight_line, (
        f"step order broken: verify@{verify_line} live-cycle@{live_cycle_line} "
        f"firstlight@{firstlight_line}"
    )


def test_live_cycle_invocation_inside_conditional_block() -> None:
    """The bert_demo_cycle.py invocation MUST live inside the
    `if [ "$WITH_LIVE_CYCLE" -eq 1 ]` block. Outside the conditional
    would mean we'd fire a live cycle on every demo run — a 60-120s
    surprise the founder would discover on stage."""
    lines = RUN_SH.read_text().splitlines()
    invoke_idx = None
    for i, line in enumerate(lines):
        if "bert_demo_cycle.py" in line:
            invoke_idx = i
            break
    assert invoke_idx is not None, "bert_demo_cycle.py invocation line not found"

    # Walk UPWARD from the invocation to find the nearest enclosing 'if'
    enclosing_if = None
    for i in range(invoke_idx, -1, -1):
        line = lines[i].strip()
        if line.startswith("if ") and "WITH_LIVE_CYCLE" in line:
            enclosing_if = i
            break
        # If we hit a closing 'fi' or the start of another step, we're
        # already outside any conditional that should wrap us.
        if line == "fi" or line.startswith("# ── Step "):
            break
    assert enclosing_if is not None, (
        f"bert_demo_cycle.py at line {invoke_idx+1} is NOT inside a "
        f"WITH_LIVE_CYCLE conditional — would fire on every demo run"
    )

    # And the corresponding 'fi' must come AFTER the invocation
    fi_idx = None
    depth = 1
    for i in range(enclosing_if + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("if "):
            depth += 1
        elif line == "fi":
            depth -= 1
            if depth == 0:
                fi_idx = i
                break
    assert fi_idx is not None and fi_idx > invoke_idx, (
        "matching 'fi' not found after the bert_demo_cycle.py invocation"
    )


def test_orchestrator_uses_if_not_for_demo_cycle_failure() -> None:
    """Stage-safety: live-cycle returning non-zero (partial dispatch
    failure) MUST NOT kill the orchestrator via set -e. The if-not-then
    wrapper is the idiomatic guard. Without it, a single dispatch
    INVALID would terminate the whole demo mid-presentation."""
    text = RUN_SH.read_text()
    # Pattern: `if ! "$BERT_LAB_ROOT/.venv/bin/python" "$BERT_LAB_ROOT/tools/bert_demo_cycle.py"`
    assert re.search(
        r"if\s*!\s*[\"'$]*[^\"'$]*python[\"'$]*\s+[\"'$]*[^\"'$]*bert_demo_cycle\.py",
        text,
    ), "bert_demo_cycle.py must be wrapped in `if ! ... ; then` for set -e safety"


def test_unknown_flag_warning_fires_at_runtime() -> None:
    """Runtime check: passing an unknown flag actually triggers the
    warning. Stops the case where someone edits the warn message but
    breaks the case statement above it."""
    # Use --help after --bogus so the script exits without booting uvicorn
    result = subprocess.run(
        ["bash", str(RUN_SH), "--bogus-flag", "--help"],
        capture_output=True, text=True, timeout=5,
    )
    combined = result.stdout + result.stderr
    assert "unrecognized" in combined or "unknown" in combined.lower(), \
        f"unknown-flag warning didn't fire at runtime; output: {combined[:300]}"
    assert "--bogus-flag" in combined, \
        "warning should echo the offending flag back to the user"


def test_bert_demo_cycle_module_imports_clean() -> None:
    """Importing bert_demo_cycle must not (a) raise, (b) call a model,
    or (c) hit the filesystem in destructive ways. The smoke-importable
    discipline matters for IDEs + future test reuse."""
    result = subprocess.run(
        [str(VENV_PY), "-c",
         "import sys; sys.path.insert(0, '.'); "
         "import tools.bert_demo_cycle as m; "
         "assert hasattr(m, 'run_demo_cycle'); "
         "assert hasattr(m, 'main'); "
         "print('OK')"],
        capture_output=True, text=True, timeout=10,
        cwd=str(LAB_ROOT),
        env={"PATH": "/usr/bin:/bin"},  # no API keys present
    )
    assert result.returncode == 0, \
        f"bert_demo_cycle import failed: stdout={result.stdout!r}, stderr={result.stderr!r}"
    assert "OK" in result.stdout, "import sanity check didn't print OK"


def test_bert_demo_cycle_exposes_documented_exit_codes() -> None:
    """The module's docstring promises three exit codes (0/1/2). Verify
    the constants/logic exist so the orchestrator can rely on them."""
    text = (LAB_ROOT / "tools" / "bert_demo_cycle.py").read_text()
    # Module must document exit codes
    assert "exit code" in text.lower() or "Returns shell exit code" in text, \
        "bert_demo_cycle must document its exit codes"
    # All three codes must appear in return statements somewhere
    assert "return 0" in text, "must have a success path (exit 0)"
    assert "return 1" in text, "must have a partial-success path (exit 1)"
    assert "return 2" in text, "must have a hard-failure path (exit 2)"


def test_orchestrator_help_documents_with_live_cycle_flag() -> None:
    """The --help output (from the header comment) must reach the user
    when --help is passed. Catches the case where someone reformats the
    header and breaks the sed-driven help."""
    result = subprocess.run(
        ["bash", str(RUN_SH), "--help"],
        capture_output=True, text=True, timeout=5,
    )
    out = result.stdout + result.stderr
    assert "--with-live-cycle" in out, \
        "--help output must document --with-live-cycle"
    assert "GROQ_API_KEY" in out or "API key" in out.lower() or "key" in out.lower(), \
        "--help should mention provider key requirement"


def main() -> int:
    tests = [
        test_bert_demo_cycle_exists,
        test_demo_cycle_dry_run_works,
        test_demo_cycle_aborts_without_keys,
        test_demo_cycle_rejects_unknown_scenario,
        test_orchestrator_bash_syntax_valid,
        test_orchestrator_parses_with_live_cycle_flag,
        test_orchestrator_help_flag_works,
        test_orchestrator_invokes_bert_demo_cycle,
        test_narration_includes_live_cycle_segment,
        test_orchestrator_warns_on_unknown_flag,
        # ── S.4 runtime + structural depth ──
        test_step_order_is_verify_then_live_cycle_then_ui,
        test_live_cycle_invocation_inside_conditional_block,
        test_orchestrator_uses_if_not_for_demo_cycle_failure,
        test_unknown_flag_warning_fires_at_runtime,
        test_bert_demo_cycle_module_imports_clean,
        test_bert_demo_cycle_exposes_documented_exit_codes,
        test_orchestrator_help_documents_with_live_cycle_flag,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
